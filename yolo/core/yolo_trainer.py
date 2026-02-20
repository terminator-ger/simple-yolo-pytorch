"""
Create by:  zh320
Date:       2024/06/15
"""

import os
from token import AT
import torch
from tqdm import tqdm
from torch.cuda import amp
from torchvision.ops import nms, batched_nms, box_iou
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, AUROC, F1Score, ConfusionMatrix, Precision, Recall, PrecisionRecallCurve

from .base_trainer import BaseTrainer
from yolo.utils import (get_det_metrics, sampler_set_epoch, xywh_to_xyxy)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class YOLOTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        if config.task in ['train', 'val']:
            self.mAP = get_det_metrics().to(self.device)
            self.precision = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)

            # Classification metrics with background class (num_classes + 1)
            num_classes_with_bg = config.num_class + 1
            self.accuracy = Accuracy(task='multiclass', num_classes=num_classes_with_bg, average='none').to(self.device)
            self.f1 = F1Score(task='multiclass', num_classes=num_classes_with_bg, average='none').to(self.device)
            self.precision_metric = Precision(task='multiclass', num_classes=num_classes_with_bg, average='none').to(self.device)
            self.recall = Recall(task='multiclass', num_classes=num_classes_with_bg, average='none').to(self.device)
            self.cm = ConfusionMatrix(task='multiclass', num_classes=num_classes_with_bg).to(self.device)
            self.auroc = AUROC(task='multiclass', num_classes=num_classes_with_bg).to(self.device)
            self.pr_curve = PrecisionRecallCurve(task='multiclass', num_classes=num_classes_with_bg).to(self.device)

        if config.task == 'debug':
            from .loss import get_loss_fn
            self.loss_fn = get_loss_fn(config)
        self.config = config

    def train_one_epoch(self, config):
        self.model.train()

        sampler_set_epoch(config, self.train_loader, self.cur_epoch) 

        pbar = tqdm(self.train_loader) if self.main_rank else self.train_loader

        for cur_itrs, (images, bboxes, classes) in enumerate(pbar):
            self.cur_itrs = cur_itrs
            self.train_itrs += 1

            images = images.to(self.device, dtype=torch.float32)
            bboxes = bboxes.to(self.device, dtype=torch.float32)    
            classes = classes.to(self.device, dtype=torch.float32)    

            self.optimizer.zero_grad()

            # Forward path
            with amp.autocast(enabled=config.amp_training):
                preds = self.model(images)
                loss, (conf_loss, iou_loss, class_loss, num_pos) = self.loss_fn(preds, bboxes, classes)
            if torch.isnan(loss):
                print(f"NaN detected in loss at iteration {cur_itrs} of epoch {self.cur_epoch}. Skipping backward pass.")
                continue
            loss = loss.clamp(min=0, max=10)  # Optional: Clamp loss to prevent extreme values

            if config.use_tb and self.main_rank:
                self.writer.add_scalar('num_pos', num_pos, self.train_itrs)
                self.writer.add_scalar('train/loss', loss.detach(), self.train_itrs)
                self.writer.add_scalar('train/conf_loss', conf_loss, self.train_itrs)
                self.writer.add_scalar('train/iou_loss', iou_loss, self.train_itrs)
                self.writer.add_scalar('train/class_loss', class_loss, self.train_itrs)

            # Backward path
            self.scaler.scale(loss).backward()

            # # Clip the gradients
            # self.scaler.unscale_(self.optimizer)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.ema_model.update(self.model, self.train_itrs)

            if self.main_rank:
                pbar.set_description(('%s'*6) % 
                                (f'Epoch:{self.cur_epoch}/{config.total_epoch}{" "*4}|',
                                f'Loss:{loss.item():4.4g}{" "*4}|',
                                f'Conf Loss:{conf_loss:4.4g}{" "*4}|',
                                f'IoU Loss:{iou_loss:4.4g}{" "*4}|',
                                f'Class Loss:{class_loss:4.4g}{" "*4}|',
                                f'Num Pos:{num_pos:4.2g}{" "*2}|',
                                ))
        return

    @torch.no_grad()
    def validate(self, config, val_best=False):
        empty_tensor = torch.tensor([]).to(self.device)
        pbar = tqdm(self.val_loader) if self.main_rank else self.val_loader
        for (images, bboxes, classes) in pbar:
            images = images.to(self.device, dtype=torch.float32)
            bboxes = bboxes.to(self.device, dtype=torch.float32)
            classes = classes.to(self.device, dtype=torch.long)

            preds = self.ema_model.ema(images, is_training=False)
            raw_preds = preds

            _, _, height, width = images.shape
            outputs, targets = [], []
            matched_preds, matched_targets, matched_probs = [], [], []
    
            for i, pred in enumerate(preds):
                raw_pred = raw_preds[i]
                pred_conf = pred[:, 0]

                pred_boxes = xywh_to_xyxy(pred[:, 1:5])
                pred_boxes[:, 0::2].clamp_(0, width)
                pred_boxes[:, 1::2].clamp_(0, height)

                cls_logits, pred_cls = preds[i][:, 5:].max(dim=1)
                pred_conf *= cls_logits
                pred = torch.cat([pred_conf.unsqueeze(1), pred_boxes, pred_cls.unsqueeze(1)], dim=1)
                output = pred[pred_conf > config.conf_thrs]

                # NMS per image
                kept_indices = self.nms(output, max_nms_num=config.max_nms_num, nms_iou=config.val_iou)

                if kept_indices is None:
                    outputs.append(dict(boxes=empty_tensor, scores=empty_tensor, labels=empty_tensor.long()))
                else:
                    outputs.append(dict(boxes=output[kept_indices][:, 1:5], scores=output[kept_indices][:, 0], 
                                        labels=output[kept_indices][:, 5].long()))

                if bboxes.shape[0]:
                    targets.append(dict(boxes=bboxes[bboxes[:, 0]==i][:, 1:], 
                                        labels=classes[classes[:, 0]==i][:, 1].long()))
                else:
                    targets.append(dict(boxes=empty_tensor, labels=empty_tensor.long()))

                # Matching for classification metrics - includes TP, FP, FN
                output = outputs[-1]
                target = targets[-1]

                matched_probs, matched_targets, matched_preds = self.evaluate(output, target, raw_pred)

                if False:
                    draw_gt(images[i], targets[-1], config, save_dir=config.save_dir, idx=i)
                    
            self.accuracy.update(matched_preds, matched_targets)
            self.f1.update(matched_preds, matched_targets)
            self.precision_metric.update(matched_preds, matched_targets)
            self.recall.update(matched_preds, matched_targets)
            self.cm.update(matched_preds, matched_targets)
            self.auroc.update(matched_probs, matched_targets)
            self.pr_curve.update(matched_probs, matched_targets)
            self.mAP.update(outputs, targets)


            if self.main_rank:
                pbar.set_description(('%s'*1) % (f'Validating:{" "*4}|',))

        if self.main_rank and config.use_tb and self.cur_epoch < config.total_epoch:
            # Compute scalar metrics
            accuracy = self.accuracy.compute()
            f1 = self.f1.compute()
            precision = self.precision_metric.compute()
            recall = self.recall.compute()
            auroc = self.auroc.compute()
            map = self.mAP.compute()
            map50 = map['map_50']
            map50_95 = map['map']

            # Log scalars
            self.writer.add_scalar('val/mAP@IoU=0.5', map50, self.cur_epoch+1)
            self.writer.add_scalar('val/mAP@IoU=0.5:0.95', map50_95, self.cur_epoch+1)
            self.writer.add_scalar('val/accuracy_0', accuracy[0], self.cur_epoch+1)
            self.writer.add_scalar('val/accuracy_1', accuracy[1], self.cur_epoch+1)
            self.writer.add_scalar('val/accuracy', accuracy.mean(), self.cur_epoch+1)
            self.writer.add_scalar('val/f1_0', f1[0], self.cur_epoch+1)
            self.writer.add_scalar('val/f1_1', f1[1], self.cur_epoch+1)
            self.writer.add_scalar('val/f1', f1.mean(), self.cur_epoch+1)
            self.writer.add_scalar('val/precision_0', precision[0], self.cur_epoch+1)
            self.writer.add_scalar('val/precision_1', precision[1], self.cur_epoch+1)
            self.writer.add_scalar('val/precision', precision.mean(), self.cur_epoch+1)
            self.writer.add_scalar('val/recall_0', recall[0], self.cur_epoch+1)
            self.writer.add_scalar('val/recall_1', recall[1], self.cur_epoch+1)
            self.writer.add_scalar('val/recall', recall.mean(), self.cur_epoch+1)
            self.writer.add_scalar('val/auroc', auroc, self.cur_epoch+1)

            os.makedirs(os.path.join(config.save_dir, 'plots'), exist_ok=True)
            # Plot using torchmetrics' built-in plotting if available
            fig_cm, ax_ = self.cm.plot()
            self.writer.add_figure('val/confusion_matrix', fig_cm, self.cur_epoch+1)
            fig_cm.savefig(os.path.join(config.save_dir,'plots', f'confusion_matrix_epoch{self.cur_epoch+1}.png'))

            fig_pr, ax_ = self.pr_curve.plot(score=True)
            self.writer.add_figure('val/pr_curve', fig_pr, self.cur_epoch+1)
            fig_pr.savefig(os.path.join(config.save_dir, 'plots', f'pr_curve_epoch{self.cur_epoch+1}.png')) 

            fig_f1, ax_ = self.f1.plot()
            self.writer.add_figure('val/f1_curve', fig_f1, self.cur_epoch+1)
            fig_f1.savefig(os.path.join(config.save_dir, 'plots', f'f1_curve_epoch{self.cur_epoch+1}.png'))

            fig_precision, ax_ = self.precision_metric.plot()
            self.writer.add_figure('val/precision_curve', fig_precision, self.cur_epoch+1)
            fig_precision.savefig(os.path.join(config.save_dir, 'plots', f'precision_curve_epoch{self.cur_epoch+1}.png'))
            fig_recall, ax_ = self.recall.plot()
            self.writer.add_figure('val/recall_curve', fig_recall, self.cur_epoch+1)
            fig_recall.savefig(os.path.join(config.save_dir, 'plots', f'recall_curve_epoch{self.cur_epoch+1}.png'))  
            
            fig_auroc, ax_ = self.auroc.plot()
            self.writer.add_figure('val/auroc_curve', fig_auroc, self.cur_epoch+1)      
            fig_auroc.savefig(os.path.join(config.save_dir, 'plots', f'auroc_curve_epoch{self.cur_epoch+1}.png'))    

            # Reset metrics for next use
            self.accuracy.reset()
            self.f1.reset()
            self.precision_metric.reset()
            self.recall.reset()
            self.cm.reset()
            self.auroc.reset()
            self.pr_curve.reset()
            self.mAP.reset()

        if self.main_rank:
            if val_best:
                self.logger.info(f'\n\nTrain {config.total_epoch} epochs finished.')
            else:
                self.logger.info(f' Epoch{self.cur_epoch} score: {map50:.4f}    | ' + 
                                 f'best val-score so far: {self.best_score:.4f}\n')

        if not isinstance(map50, torch.Tensor):
            map50 = torch.tensor(map50)
        return map50.to(self.device)


    @classmethod
    def nms(cls, output, class_agnostic=False, max_nms_num=300, nms_iou=0.6):
        if class_agnostic:
            kept_indices = nms(boxes=output[:, 1:5], scores=output[:, 0], iou_threshold=nms_iou)
        else:
            kept_indices = batched_nms(boxes=output[:, 1:5], scores=output[:, 0], idxs=output[:, 5], iou_threshold=nms_iou)

        if not kept_indices.shape[0]:
            return None

        if kept_indices.shape[0] > max_nms_num:
            kept_indices = kept_indices[:max_nms_num]

        return kept_indices

    @torch.no_grad()
    def predict(self, config):
        if config.DDP:
            raise ValueError('Predict mode currently does not support DDP.')

        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype("tools/ARIAL.TTF", 14)

        self.logger.info('\nStart predicting...\n')

        self.model.eval()
        for (images, images_aug, img_names) in tqdm(self.test_loader):
            images_aug = images_aug.to(self.device, dtype=torch.float32)

            preds = self.model(images_aug, is_training=False)

            height, width = images_aug.shape[2:]
            resize_ratio = torch.tensor(images.shape[1:3]) / torch.tensor(images_aug.shape[2:])

            os.makedirs(os.path.join(config.save_dir, 'imgs'), exist_ok=True)
            for i, pred in enumerate(preds):
                image = Image.fromarray(images[i].numpy())
                save_path = os.path.join(config.save_dir,'imgs', img_names[i])

                pred_conf = pred[:, 0]
                pred_boxes = xywh_to_xyxy(pred[:, 1:5])
                pred_boxes[:, 0::2].clamp_(0, width)
                pred_boxes[:, 1::2].clamp_(0, height)
                cls_logits, pred_cls = pred[:, 5:].max(dim=1)
                pred_conf *= cls_logits
                pred = torch.cat([pred_conf.unsqueeze(1), pred_boxes, pred_cls.unsqueeze(1)], dim=1)
                output = pred[pred_conf > config.test_conf_thrs]

                # NMS per image
                kept_indices = self.nms(output, nms_iou=config.test_iou)

                if kept_indices is not None:
                    output = output[kept_indices].cpu()
                    conf = output[:, 0]
                    bbox = output[:, 1:5]
                    cls = output[:, 5].long()

                    bbox[:, 0::2] *= resize_ratio[1]
                    bbox[:, 1::2] *= resize_ratio[0]
                    bbox = bbox.long()

                    draw = ImageDraw.Draw(image)

                    for j in range(conf.shape[0]):
                        #text = f'{config.class_map[cls[j].item()]}-{conf[j]:.2f}'
                        draw.rectangle(bbox[j].tolist(), outline=tuple(config.color_map[cls[j]]), width=2)
                        #draw.text(bbox[j,:2].tolist(), text, fill='white', font=font)

                image.save(save_path)

    @torch.no_grad()
    def debug(self, config):
        if config.DDP:
            raise ValueError('Debug mode currently does not support DDP.')

        from yolo.utils.plot import visualize_assignments

        self.logger.info(f'\nStart visualizing {config.label_assignment_method} label assignment results...\n')
        for cur_itrs, (images, bboxes, classes) in enumerate(tqdm(self.train_loader)):
            if cur_itrs >= config.num_debug_batch:
                break

            bboxes, classes = bboxes.float(), classes.float()

            assigned_labels = self.loss_fn.label_assignment(bboxes, classes, config.train_bs, len(config.anchor_boxes))

            visualize_assignments(assigned_labels=assigned_labels, bboxes=bboxes, anchor_boxes=config.anchor_boxes,
                                    batch_idx=cur_itrs, img_size=config.img_size, image_stride=(4, 8, 16, 32), save_dir=config.debug_dir)

        self.logger.info('Debug finished.\n')

    
    @torch.no_grad()
    def evaluate(self, output, target, raw_pred):
        matched_preds, matched_targets, matched_probs = [], [], []
        # Track which predictions and ground truths have been matched
        matched_pred_indices = set()
            
        if len(output['boxes']) > 0 and len(target['boxes']) > 0:
            ious = box_iou(output['boxes'], target['boxes'])
            
            # Match predictions to ground truths (greedy matching)
            for j, true_label in enumerate(target['labels']):
                iou_row = ious[:, j]
                if iou_row.max() > 0.5:
                    # True positive
                    best_idx = iou_row.argmax()
                    matched_preds.append(output['labels'][best_idx])
                    matched_targets.append(true_label)
                    # Probability vector including background class
                    class_probs = F.softmax(raw_pred[best_idx, 5:], dim=0)
                    # Pad with zero probability for background class
                    probs = torch.cat([class_probs, torch.tensor([0.0], device=self.device)])
                    matched_probs.append(probs)
                    matched_pred_indices.add(best_idx)
                else:
                    # False negative - ground truth without matching prediction
                    # Record as background class prediction
                    matched_preds.append(torch.tensor(self.config.num_class, device=self.device, dtype=torch.long))
                    matched_targets.append(true_label)
                    # Background class probability vector
                    bg_probs = torch.zeros(self.config.num_class + 1, device=self.device, dtype=torch.float)
                    bg_probs[self.config.num_class] = 1.0  # High confidence for background
                    matched_probs.append(bg_probs)

        elif len(target['boxes']) > 0 and len(output['boxes']) == 0:
            # No predictions but targets exist - all are false negatives
            for true_label in target['labels']:
                matched_preds.append(torch.tensor(self.config.num_class, device=self.device, dtype=torch.long))
                matched_targets.append(true_label)
                bg_probs = torch.zeros(self.config.num_class + 1, device=self.device, dtype=torch.float)
                bg_probs[self.config.num_class] = 1.0
                matched_probs.append(bg_probs)
            
        # False positives - predictions without matching ground truth (outside if block)
        if len(output['boxes']) > 0:
            for pred_idx in range(len(output['labels'])):
                if pred_idx not in matched_pred_indices:
                    # False positive prediction
                    matched_preds.append(output['labels'][pred_idx])
                    # Mark as background/no-class (use num_classes as background index)
                    matched_targets.append(torch.tensor(self.config.num_class, device=self.device, dtype=torch.long))
                    # Probability vector including background class
                    class_probs = F.softmax(raw_pred[pred_idx, 5:], dim=0)
                    probs = torch.cat([class_probs, torch.tensor([0.0], device=self.device)])
                    matched_probs.append(probs)

        # Update metric state
        matched_preds = torch.tensor(matched_preds).to(self.device)
        matched_targets = torch.tensor(matched_targets).to(self.device)
        matched_probs = torch.stack(matched_probs).to(self.device)


        return matched_probs, matched_targets, matched_preds


def draw_gt(image, target, config, save_dir='debug', idx=0):
    from PIL import ImageDraw
    os.makedirs(save_dir, exist_ok=True)
    draw = ImageDraw.Draw(image)
    for j in range(target['boxes'].shape[0]):
        cls_id = target['labels'][j].item()
        bbox = target['boxes'][j].tolist()
        draw.rectangle(bbox, outline=tuple(config.color_map[cls_id]), width=2)
    image.save(os.path.join(save_dir, f'gt_{idx}.png'))
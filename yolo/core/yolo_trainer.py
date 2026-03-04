"""
Create by:  zh320
Date:       2024/06/15
"""

import logging
import os
from token import AT
import torch
from tqdm import tqdm
from torch.cuda import amp
from torchvision.ops import nms, batched_nms

from .base_trainer import BaseTrainer
from yolo.utils import (get_det_metrics, sampler_set_epoch, xywh_to_xyxy)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from yolo.core.evaluate import Evaluator, get_corners_at_iou, get_matches_at_iou
import yolo.core.evaluate as E
from yolo.models.yolo import ModelOutputKeys

class YOLOTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        if config.task in ['train', 'val']:
            self.eval = Evaluator(E.eval_builder(config.evaluators), device=self.device)

        if config.task == 'debug':
            from .loss import get_loss_fn
            self.loss_fn = get_loss_fn(config)

        self.config = config

    def train_one_epoch(self, config):
        self.model.train()

        sampler_set_epoch(config, self.train_loader, self.cur_epoch) 

        pbar = tqdm(self.train_loader) if self.main_rank else self.train_loader

        for cur_itrs, data in enumerate(pbar):
            self.cur_itrs = cur_itrs
            self.train_itrs += 1

            images = data['pixel_values'].to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad()

            # Forward path
            with amp.autocast(enabled=config.amp_training):
                preds = self.model(images)
                loss, loss_dict = self.loss_fn(preds, data)

            if torch.isnan(loss):
                print(f"NaN detected in loss at iteration {cur_itrs} of epoch {self.cur_epoch}. Skipping backward pass.")
                continue
            loss = loss.clamp(min=0, max=10)  # Optional: Clamp loss to prevent extreme values

            if config.use_tb and self.main_rank:
                self.writer.add_scalar('pos_ratio_kp', loss_dict["pos_ratio_kp"], self.train_itrs)
                self.writer.add_scalar('pos_ratio_stone', loss_dict["pos_ratio_stone"], self.train_itrs)
                self.writer.add_scalar('train/loss', loss.detach(), self.train_itrs)
                self.writer.add_scalar('train/conf_loss',  loss_dict["conf"], self.train_itrs)
                self.writer.add_scalar('train/iou_loss',   loss_dict["iou"], self.train_itrs)
                self.writer.add_scalar('train/class_loss', loss_dict["class"], self.train_itrs)
                if "kp" in loss_dict and loss_dict["kp"] is not None:
                    self.writer.add_scalar('train/kp_loss', loss_dict["kp"], self.train_itrs)

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
                                f'Conf Loss:{loss_dict["conf"]:4.4g}{" "*4}|',
                                f'IoU Loss:{loss_dict["iou"]:4.4g}{" "*4}|',
                                f'Class Loss:{loss_dict["class"]:4.4g}{" "*4}|',
                                f'Pos Ratio:{loss_dict["pos_ratio_stone"]:4.2g}{" "*2}|',
                                ))
        return

    @torch.no_grad()
    def validate(self, config, val_best=False):
        def _select_by_index(x: torch.Tensor, idx: int) -> torch.tensor:
            select = x[:,0] == idx
            out = x[select][:,1:]
            return out

        pbar = tqdm(self.val_loader) if self.main_rank else self.val_loader
        for data in pbar:
            images = data['pixel_values'].to(self.device, dtype=torch.float32)

            batch_predictions = self.ema_model.ema(images, is_training=False)

            _, _, height, width = images.shape

            # iterate over all output fields and batches 
            for k, prediction in batch_predictions.items():
                for batch_idx, p in enumerate(prediction):

                    pred, target = {},{}
                    if k == ModelOutputKeys.Keypoints: 
                        pred_conf = p[:,0]
                        pred_classes = torch.ones((p.shape[0], 1), dtype=torch.float32, device=p.device)
                        pred_boxes = p[:, 1:5]
                        target_boxes   = data['keypoints_bboxes'][data['keypoints_bboxes'][:,0]==batch_idx][:,1:]
                        target_classes = data['keypoints_classes'][data['keypoints_classes'][:,0]==batch_idx][:,1:]
                        num_class = 1
                    else:
                        pred_conf = p[:,0]
                        pred_classes = p[:,5:]
                        pred_boxes = p[:,1:5]
                        target_boxes   = data['bboxes'][data['bboxes'][:,0]==batch_idx][:,1:]
                        target_classes = data['classes'][data['classes'][:,0]==batch_idx][:,1:]
                        num_class = self.config.num_class

                    matched_probs, matched_targets, matched_preds, select_idx = get_matches_at_iou(
                        pred_boxes=pred_boxes,
                        pred_classes=pred_classes,
                        pred_conf=pred_conf,
                        target_boxes=target_boxes,
                        target_classes=target_classes,
                        iou=0.5,
                        width=width,
                        height=height,
                        num_class=num_class)

                    # iterate over output dict
                    if k == ModelOutputKeys.Detections:
                        box_target = _select_by_index(data['bboxes'], batch_idx).to(self.device)
                        label_target = _select_by_index(data['classes'], batch_idx).to(self.device).squeeze(1)
                        labels_pred = torch.argmax(p[select_idx,5:], axis=1)

                        pred[ModelOutputKeys.Detections] = {
                            E.EvalTypes.ClassificationLogits: matched_probs,
                            E.EvalTypes.Classification: matched_preds,
                            E.EvalTypes.BBox: [{
                                "boxes": p[select_idx, 1:5],
                                "scores": p[select_idx, 0],
                                "labels": labels_pred
                            }]}
                        target[ModelOutputKeys.Detections] = {
                            E.EvalTypes.ClassificationLogits: matched_targets,
                            E.EvalTypes.Classification: matched_targets,
                            E.EvalTypes.BBox: [{
                                "boxes": box_target,
                                "labels": label_target
                        }]}
                    
                    if k == ModelOutputKeys.Keypoints:
                        select_idx = pred_conf.argmax()
                        kp_matched = p[select_idx, 5:].reshape(-1,8)
                        kp_label_pred = torch.zeros((1), device=self.device, dtype=torch.int32)
                        kp_target = _select_by_index(data['keypoints'], batch_idx).reshape(-1,8).to(self.device)
                        kp_box_target = _select_by_index(data['keypoints_bboxes'], batch_idx).to(self.device)
                        kp_label_target = _select_by_index(data['keypoints_classes'], batch_idx).to(self.device).squeeze(1)
                        
                        pred[ModelOutputKeys.Keypoints] = {
                            E.EvalTypes.BBox: [{
                                "boxes": xywh_to_xyxy(p[select_idx, 1:5]).reshape(-1,4),
                                "scores": p[select_idx, 0].reshape(-1),
                                "labels": kp_label_pred
                            }],
                            E.EvalTypes.Keypoint: kp_matched
                        }
                        target[ModelOutputKeys.Keypoints] = {
                            E.EvalTypes.BBox: [{
                                "boxes": kp_box_target,
                                "labels": kp_label_target
                            }],
                            E.EvalTypes.Keypoint: kp_target
                        }

                    if len(pred) == 0 and len(target) == 0:
                        logging.warning("could not gather targets")
                        continue
                    
                    self.eval.update(pred, target)

            if self.main_rank:
                pbar.set_description(('%s'*1) % (f'Validating:{" "*4}|',))

        if self.main_rank and config.use_tb and self.cur_epoch < config.total_epoch:
            # Compute scalar metrics
            metrics = self.eval.compute()
            for k,v in metrics.items():
                try:
                    self.writer.add_scalar("val/"+k, v, self.cur_epoch+1)
                except (RuntimeError, AssertionError):
                    if len(v.shape) == 1:
                        for idx in range(len(v)):
                            self.writer.add_scalar(f"val/{k}_{idx}", v[idx], self.cur_epoch+1)

            # save plots
            os.makedirs(os.path.join(config.save_dir, 'plots'), exist_ok=True)
            #plots = self.eval.plot()
            #for plt in plots:
            #    k,v = plt.items()
            #    self.writer.add_figure('val/'+k, v, self.cur_epoch+1)
            #    v.savefig(os.path.join(config.save_dir,'plots', f'k{self.cur_epoch+1}.png'))

            map50 = metrics['B_map_Stones']
            # Reset metrics for next use
            self.eval.reset()

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

    

def draw_gt(image, target, config, save_dir='debug', idx=0):
    from PIL import ImageDraw
    os.makedirs(save_dir, exist_ok=True)
    draw = ImageDraw.Draw(image)
    for j in range(target['boxes'].shape[0]):
        cls_id = target['labels'][j].item()
        bbox = target['boxes'][j].tolist()
        draw.rectangle(bbox, outline=tuple(config.color_map[cls_id]), width=2)
    image.save(os.path.join(save_dir, f'gt_{idx}.png'))
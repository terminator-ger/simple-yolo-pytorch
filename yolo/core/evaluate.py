import logging
from typing import Dict, List, Optional
import torch as th
from torchmetrics import Accuracy, AUROC, F1Score, ConfusionMatrix, Precision, Recall, PrecisionRecallCurve
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchmetrics
from enum import StrEnum, auto
from collections import ChainMap
import torch
from yolo.utils.utils import xywh_to_xyxy
import torchvision
import torch.nn.functional as F
from yolo.core.metrics import EucildeanDistance



class EvalTypes(StrEnum):
    BBox = 'B'
    Classification = 'C'
    ClassificationLogits = 'CL'
    Keypoint = 'K'

class LabelEvaluator(th.nn.Module):
    def __init__(self, postfix: str):
        super().__init__()
        self.postfix = postfix
        self.metrics = torchmetrics.MetricCollection({})
        self.device = 'cpu'
    

    def update(self, pred: th.Tensor | Dict, target:th.Tensor | Dict) -> None:
        self.metrics.update(pred, target)

    def compute(self) -> Dict:
        return self.metrics.compute()

    def plot(self):
        return self.metrics.plot()[0]

    def reset(self):
        self.metrics.reset()
    
    def to(self, device) -> None:
        self.metrics.to(device)
 

class Evaluator:
    def __init__(self, elevators: Dict[str, Dict[EvalTypes, LabelEvaluator]], device: Optional[torch.device]=None):
        self.evaluators = elevators
        if device is not None:
            self.to(device)

    def update(self, pred: Dict, target:Dict) -> None:
        for k,value in pred.items():
            if k not in self.evaluators:
                logging.debug(f"Metric for {k} not found")
                continue
            for ev_tvpe, eval in self.evaluators[k].items():
                eval.update(value[ev_tvpe], target[k][ev_tvpe])

    def compute(self) -> Dict:
        results = [x.compute()  for eval in self.evaluators.values() for x in eval.values()]
        return ChainMap(*results)
    
    def reset(self) -> None:
        [x.reset()  for eval in self.evaluators.values() for x in eval.values()]

    def plot(self) -> List:
        out = []
        for key in self.evaluators.keys():
            for tvpe in self.evaluators[key]:
                _names = self.evaluators[key][tvpe].metrics.keys()
                for name in _names:
                    f, a = self.evaluators[key][tvpe].metrics[name].plot()
                    out.append({name: f})
        return out

    def to(self, device) -> None:
        for _,metric in self.evaluators.items():
            for _,m in metric.items():
                m.to(device)


class BoundingBoxEvaluator(LabelEvaluator):
    def __init__(self, postfix=""):
        super().__init__(postfix)

        self.metrics = torchmetrics.MetricCollection({
                "map" : MeanAveragePrecision(box_format='xywh', iou_type='bbox', class_metrics=True)},
            prefix=EvalTypes.BBox+"_",
            postfix="_"+postfix
       )
        
    #def update(self, pred: Dict, target:Dict) -> None:
    #    self.metrics.update(pred, target)

class ClassificationEvaluatorLogits(LabelEvaluator):
    def __init__(self, postfix="", num_class=0):
        super().__init__(postfix)

        num_classes_with_bg = num_class + 1
        self.metrics = torchmetrics.MetricCollection({
                "auroc": AUROC(task='multiclass', num_classes=num_classes_with_bg), 
                "pr_curve": PrecisionRecallCurve(task='multiclass', num_classes=num_classes_with_bg)},
            prefix=EvalTypes.ClassificationLogits+"_",
            postfix="_"+postfix
       )

    def compute(self) -> Dict:
        _dict = self.metrics.compute()
        del _dict[EvalTypes.ClassificationLogits+"_pr_curve_"+self.postfix]
        return _dict


   
class ClassificationEvaluator(LabelEvaluator):
    def __init__(self, postfix="", num_class=0):
        super().__init__(postfix=postfix)

        num_classes_with_bg = num_class + 1
        self.metrics = torchmetrics.MetricCollection({
                "accuracy":  Accuracy(task='multiclass', num_classes=num_classes_with_bg, average='none'),
                "f1": F1Score(task='multiclass', num_classes=num_classes_with_bg, average='none'),
                "precision": Precision(task='multiclass', num_classes=num_classes_with_bg, average='none'),
                "recall": Recall(task='multiclass', num_classes=num_classes_with_bg, average='none'),
                "cm": ConfusionMatrix(task='multiclass', num_classes=num_classes_with_bg)},
            prefix=EvalTypes.Classification+"_",
            postfix="_"+postfix
       )

 
class KeypointEvaluator(LabelEvaluator):
    def __init__(self, postfix=""):
        super().__init__(postfix=postfix)

        self.metrics = torchmetrics.MetricCollection({
                'distance': EucildeanDistance()
            },
            prefix=EvalTypes.Keypoint+"_",
            postfix="_"+postfix
       )

def get_corners_at_iou(width: int,
                       height: int,
                       pred_boxes:   th.tensor, 
                       target_boxes: th.tensor, 
                       pred_conf:    th.tensor,
                       pred_kp:      th.tensor,
                       target_kp:    th.tensor,
                       iou: float = 0.5,
                       conf_t: float = 0.3):
    pred_boxes = xywh_to_xyxy(pred_boxes)
    pred_boxes[:, 0::2].clamp_(0, width)
    pred_boxes[:, 1::2].clamp_(0, height)
    pred = torch.cat([pred_conf.unsqueeze(1), pred_boxes, pred_kp], dim=1)
    output = pred[pred_conf > conf_t]

    # NMS per image
    kept_indices = _nms(output=output, nms_iou=iou)

    if kept_indices is not None:
        output = pred_kp[kept_indices]
        target = target_kp
    else:
        empty_tensor = torch.tensor([]).to(pred_boxes.device)
        output = empty_tensor
        target = empty_tensor

    return output, target


def get_matches_at_iou(width: int,
                       height: int,
                       num_class: int,
                       pred_boxes:   th.tensor, 
                       target_boxes: th.tensor, 
                       pred_conf:    th.tensor,
                       pred_classes: Optional[th.tensor] = None,
                       target_classes: Optional[th.tensor] = None,
                       iou: float = 0.5,
                       conf_t: float = 0.3):
    '''
    pred: tensor of shape [B, N] with N [Prob, X, Y, W, H, CLS_0, CLS_1, ...]
    '''
    pred_boxes = xywh_to_xyxy(pred_boxes)
    # remove invalid boxes
    invalid_idx0 = pred_boxes[:,0] < 0
    invalid_idx1 = pred_boxes[:,2] > width
    invalid_idx2 = pred_boxes[:,1] < 0
    invalid_idx3 = pred_boxes[:,3] > height
    invalid = invalid_idx0 | invalid_idx1 | invalid_idx2 | invalid_idx3
    #pred_boxes[:, 0::2].clamp_(0, width)
    #pred_boxes[:, 1::2].clamp_(0, height)
    pred_conf = pred_conf[~invalid]
    pred_classes = pred_classes[~invalid]
    pred_boxes = pred_boxes[~invalid]
    cls_logits, pred_cls = pred_classes.max(dim=1)
    pred_conf *= cls_logits
    pred = torch.cat([pred_conf.unsqueeze(1), pred_boxes, pred_cls.unsqueeze(1)], dim=1)
    output = pred[pred_conf > conf_t]

    # NMS per image
    kept_indices = _nms(output=output, nms_iou=iou, class_agnostic=True if num_class == 1 else False)

    if kept_indices is not None:
        output =  dict(boxes=output[kept_indices][:, 1:5], 
                        scores=output[kept_indices][:, 0], 
                        labels=output[kept_indices][:, 5].long())
        target = dict(boxes=target_boxes.to(pred_boxes.device),
                      labels=target_classes.to(pred_classes.device))
    else:
        empty_tensor = torch.tensor([]).to(pred_boxes.device)
        output = dict(boxes=empty_tensor, scores=empty_tensor, labels=empty_tensor.long())
        target = dict(boxes=empty_tensor, labels=empty_tensor.long())


    matched_probs, matched_targets, matched_preds = evaluate(output=output, target=target, raw_pred=pred_classes, num_class=num_class)
    return matched_probs, matched_targets, matched_preds, kept_indices


def _nms(output, class_agnostic=False, max_nms_num=300, nms_iou=0.6):
    if class_agnostic:
        kept_indices = torchvision.ops.boxes.nms(boxes=output[:, 1:5], scores=output[:, 0], iou_threshold=nms_iou)
    else:
        kept_indices = torchvision.ops.boxes.batched_nms(boxes=output[:, 1:5], scores=output[:, 0], idxs=output[:, 5], iou_threshold=nms_iou)

    if not kept_indices.shape[0]:
        return None

    if kept_indices.shape[0] > max_nms_num:
        kept_indices = kept_indices[:max_nms_num]

    return kept_indices


def evaluate( output, target, raw_pred, num_class):
    matched_preds, matched_targets, matched_probs = [], [], []
    # Track which predictions and ground truths have been matched
    matched_pred_indices = set()
        
    if len(output['boxes']) > 0 and len(target['boxes']) > 0:
        ious = torchvision.ops.boxes.box_iou(output['boxes'], target['boxes'])
        
        # Match predictions to ground truths (greedy matching)
        for j, true_label in enumerate(target['labels']):
            iou_row = ious[:, j]
            if iou_row.max() > 0.5:
                # True positive
                best_idx = iou_row.argmax()
                matched_preds.append(output['labels'][best_idx])
                matched_targets.append(true_label)
                # Probability vector including background class
                class_probs = F.softmax(raw_pred[best_idx], dim=0)
                # Pad with zero probability for background class
                probs = torch.cat([class_probs, torch.tensor([0.0], device=raw_pred.device)])
                matched_probs.append(probs)
                matched_pred_indices.add(best_idx)
            else:
                # False negative - ground truth without matching prediction
                # Record as background class prediction
                matched_preds.append(torch.tensor(num_class, device=raw_pred.device, dtype=torch.long))
                matched_targets.append(true_label)
                # Background class probability vector
                bg_probs = torch.zeros(num_class + 1, device=raw_pred.device, dtype=torch.float)
                bg_probs[num_class] = 1.0  # High confidence for background
                matched_probs.append(bg_probs)

    elif len(target['boxes']) > 0 and len(output['boxes']) == 0:
        # No predictions but targets exist - all are false negatives
        for true_label in target['labels']:
            matched_preds.append(torch.tensor(num_class, device=raw_pred.device, dtype=torch.long))
            matched_targets.append(true_label)
            bg_probs = torch.zeros(num_class + 1, device=raw_pred.device, dtype=torch.float)
            bg_probs[num_class] = 1.0
            matched_probs.append(bg_probs)
        
    # False positives - predictions without matching ground truth (outside if block)
    if len(output['boxes']) > 0:
        for pred_idx in range(len(output['labels'])):
            if pred_idx not in matched_pred_indices:
                # False positive prediction
                matched_preds.append(output['labels'][pred_idx])
                # Mark as background/no-class (use num_classes as background index)
                matched_targets.append(torch.tensor(num_class, device=raw_pred.device, dtype=torch.long))
                # Probability vector including background class
                class_probs = F.softmax(raw_pred[pred_idx], dim=0)
                probs = torch.cat([class_probs, torch.tensor([0.0], device=raw_pred.device)])
                matched_probs.append(probs)

    # Update metric state
    matched_preds = torch.tensor(matched_preds).to(raw_pred.device)
    matched_targets = torch.tensor(matched_targets).to(raw_pred.device)
    matched_probs = torch.stack(matched_probs).to(raw_pred.device)


    return matched_probs, matched_targets, matched_preds


# register classes
_lookup = {
    EvalTypes.BBox : BoundingBoxEvaluator,
    EvalTypes.Classification : ClassificationEvaluator,
    EvalTypes.ClassificationLogits : ClassificationEvaluatorLogits,
    EvalTypes.Keypoint : KeypointEvaluator
}

def eval_builder(eval_config: Dict) -> Dict[str, Dict[EvalTypes, LabelEvaluator]]:
    evaluators = {k:{} for k in eval_config.keys()}
    for k, _dict in eval_config.items():
        for _tvpe, args in _dict.items():
            evaluators[k].update({_tvpe: _lookup[_tvpe](**args)})
    return evaluators


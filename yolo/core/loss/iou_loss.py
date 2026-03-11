import torch
import math
from yolo.utils.utils import xywh_to_xyxy


class IoULoss:
    def __init__(self, iou_loss_type='iou', eps=1e-6):
        iou_method_hub = ('iou', 'ciou', 'diou', 'giou', 'siou')
        if iou_loss_type not in iou_method_hub:
            raise ValueError(f'Invalid IoU method. Supported methods are {", ".join(iou_method_hub)}.')
        self.iou_loss_type = iou_loss_type
        self.eps = eps

    def __call__(self, pred_boxes, target_boxes, xywh=False):
        if xywh:
            pred_boxes = xywh_to_xyxy(pred_boxes)
            target_boxes = xywh_to_xyxy(target_boxes)

        # Clamp boxes to valid ranges to prevent NaN
        pred_boxes = torch.clamp(pred_boxes, min=-1e4, max=1e4)
        target_boxes = torch.clamp(target_boxes, min=-1e4, max=1e4)

        # Calculate intersection and union
        intersection = self.intersection_area(pred_boxes, target_boxes)
        union = self.union_area(pred_boxes, target_boxes)
        union = torch.clamp(union, min=self.eps)
        iou = intersection / union
        iou = torch.clamp(iou, min=0, max=1)

        if self.iou_loss_type in ['ciou', 'diou', 'siou']:
            pred_center = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2
            target_center = (target_boxes[..., :2] + target_boxes[..., 2:]) / 2
            center_distance_sq = ((pred_center - target_center) ** 2).sum(dim=-1)
            center_distance = torch.sqrt(center_distance_sq.clamp(min=self.eps))

        if self.iou_loss_type in ['ciou', 'siou']:
            # Widths and heights
            w1 = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp(min=self.eps)
            h1 = (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp(min=self.eps)
            w2 = (target_boxes[..., 2] - target_boxes[..., 0]).clamp(min=self.eps)
            h2 = (target_boxes[..., 3] - target_boxes[..., 1]).clamp(min=self.eps)

            aspect_ratio_term = (self.arctan(w1 / h1) - self.arctan(w2 / h2)) ** 2 / (4 * math.pi ** 2)

        if self.iou_loss_type == 'ciou':
            iou -= (center_distance ** 2) / (self.diagonal_length_squared(pred_boxes, target_boxes) + self.eps)
            iou -= aspect_ratio_term

        elif self.iou_loss_type == 'diou':
            iou -= (center_distance ** 2) / (self.diagonal_length_squared(pred_boxes, target_boxes) + self.eps)

        elif self.iou_loss_type == 'giou':
            enclosing_area = self.enclosing_area(pred_boxes, target_boxes)
            iou = iou - (enclosing_area - union) / (enclosing_area + self.eps)

        elif self.iou_loss_type == 'siou':
            iou -= aspect_ratio_term

        loss = 1 - iou
        loss = torch.clamp(loss, min=0, max=2)
        return loss

    def intersection_area(self, boxes1, boxes2):
        tl = torch.max(boxes1[..., :2], boxes2[..., :2])
        br = torch.min(boxes1[..., 2:], boxes2[..., 2:])
        wh = (br - tl).clamp(min=0)
        return wh[..., 0] * wh[..., 1]

    def union_area(self, boxes1, boxes2):
        area1 = ((boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])).clamp(min=0)
        area2 = ((boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])).clamp(min=0)
        return area1 + area2 - self.intersection_area(boxes1, boxes2)

    def enclosing_area(self, boxes1, boxes2):
        tl = torch.min(boxes1[..., :2], boxes2[..., :2])
        br = torch.max(boxes1[..., 2:], boxes2[..., 2:])
        wh = (br - tl).clamp(min=self.eps)
        return wh[..., 0] * wh[..., 1]

    def diagonal_length_squared(self, boxes1, boxes2):
        tl = torch.min(boxes1[..., :2], boxes2[..., :2])
        br = torch.max(boxes1[..., 2:], boxes2[..., 2:])
        wh = (br - tl).clamp(min=self.eps)
        return ((wh) ** 2).sum(dim=-1)

    def arctan(self, x):
        return torch.atan(x)

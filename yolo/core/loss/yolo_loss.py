"""
Create by:  zh320
Date:       2024/06/15
"""

import logging

import torch, math
import torch.nn as nn
import torch.nn.functional as F

from yolo.utils import xywh_to_xyxy, box_iou
from yolo.models.yolo import ModelOutputKeys
from yolo.core.loss.focal_loss import FocalLoss
from yolo.core.loss.iou_loss import IoULoss
from typing import Dict
from .registry import LossRegistry

class LossMap(torch.nn.Module):

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self._map = torch.nn.ModuleDict()
        for model_output, loss_name in config.loss.items():
            loss_cls = LossRegistry()[loss_name]
            self._map.add_module(model_output, loss_cls(config))

    def forward(self, predictions, targets):
        k = list(predictions.keys())[0]
        device = predictions[k][0].device

        for k in targets.keys():
            targets[k] = targets[k].to(device)

        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        if "bboxes" in targets and ModelOutputKeys.Detections in predictions:
            bbox_loss, bbox_loss_dict = self._map[ModelOutputKeys.Detections](predictions=predictions[ModelOutputKeys.Detections], 
                                                            targets=targets)
            total_loss += self.config.lambda_loss_det * bbox_loss
            loss_dict.update(bbox_loss_dict)

        if "keypoints" in targets and ModelOutputKeys.Keypoints in predictions:
            kp_loss, kp_loss_dict = self._map[ModelOutputKeys.Keypoints](predictions=predictions[ModelOutputKeys.Keypoints], 
                                                           targets=targets)
            total_loss += self.config.lambda_loss_kp * kp_loss
            loss_dict.update(kp_loss_dict)
        
        return total_loss, loss_dict

def get_loss_fn(config):
    return LossMap(config)


class YOLOLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.label_assignment_method in ['single_grid', 'all_grid', 'nearby_grid']
        self.label_assignment_method = config.label_assignment_method
        assert config.assign_conf_method in ['iou', 'constant']
        self.assign_conf_method = config.assign_conf_method

        self.num_class = config.num_class
        self.num_attrib = 5 + self.num_class
        self.register_buffer('img_size', torch.tensor(config.img_size))
        self.lambda_coord = config.lambda_coord
        self.lambda_obj = config.lambda_obj
        self.lambda_noobj = config.lambda_noobj
        self.lambda_scales = config.lambda_scales
        self.lambda_kp = config.lambda_kp
        self.use_noobj_loss = config.use_noobj_loss
        self.match_iou_thres = config.match_iou_thres
        self.downsample_rate = config.downsample_rate
        self.filter_by_max_iou = config.filter_by_max_iou
        self.kp_shape = config.kp_shape

        self.num_anchor_per_grid = len(config.anchor_boxes[0])
        self.register_buffer('anchor_boxes', torch.stack([torch.tensor(anch, dtype=torch.float32).view(1, self.num_anchor_per_grid, 1, 1, 2) \
                                                            for anch in config.anchor_boxes]))

        if config.grid_sizes is None:
            grid_sizes = [(config.img_size[0]//config.downsample_rate[i], config.img_size[1]//config.downsample_rate[i]) for i in range(len(config.downsample_rate))]
        assert len(config.lambda_scales) == len(grid_sizes)
        self.grid_sizes = grid_sizes

        if config.label_assignment_method == 'all_grid':
            anchs = [anch.squeeze(0)/torch.tensor(config.img_size) for anch in self.anchor_boxes]
            for i, (anch, grid_size) in enumerate(zip(anchs, grid_sizes)):
                anchor_grid = self.build_anchor_grid(anch, grid_size)
                self.register_buffer(f'anchor_grid_{i}', anchor_grid)

        self.bce_loss_func = FocalLoss(gamma=config.focal_loss_gamma)
        self.iou_loss_func = IoULoss(config.iou_loss_type)
      

    def build_anchor_grid(self, anchors, grid_size):
        H, W = grid_size

        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
        grid_xy = torch.stack((grid_x, grid_y), dim=-1).float()  # (H, W, 2)

        # Normalize to [0,1]
        grid_xy = (grid_xy + 0.5) / torch.tensor([W, H])

        grid_xy = grid_xy[None, ...].expand(self.num_anchor_per_grid, H, W, 2)
        anchor_wh = anchors.expand(self.num_anchor_per_grid, H, W, 2)
        anchor_grid = torch.cat([grid_xy, anchor_wh], dim=-1)
        anchor_grid = xywh_to_xyxy(anchor_grid)

        return anchor_grid

 
    def label_assignment(self, bboxes, classes, batch_size, num_pred_layer):
        _, cls = classes.unbind(dim=1)
        cls = cls.long()

        assigned_labels = []
        for i in range(num_pred_layer):
            targets = torch.zeros((batch_size, self.num_anchor_per_grid, *self.grid_sizes[i], self.num_attrib), device=bboxes.device) 

            if len(bboxes):
                if self.label_assignment_method == 'single_grid':
                    assigned_idx, assigned_bboxes = self.single_grid_assignment(bboxes, cls, self.grid_sizes[i], self.anchor_boxes[i])

                elif self.label_assignment_method == 'all_grid':
                    assigned_idx, assigned_bboxes = self.all_grid_assignment(bboxes, cls, batch_size, getattr(self, f'anchor_grid_{i}'))

                elif self.label_assignment_method == 'nearby_grid':
                    assigned_idx, assigned_bboxes = self.nearby_grid_assignment(bboxes, cls, self.grid_sizes[i], self.anchor_boxes[i])

                else:
                    raise NotImplementedError(f'Unsupported label assignment method: {self.label_assignment_method}\n')

                b_idx, a_idx, h_idx, w_idx, cls_idx = assigned_idx
                # Set confidence score to 1
                targets[b_idx, a_idx, h_idx, w_idx, 0] = 1.                     # confidence
                # Set box coordinates
                targets[b_idx, a_idx, h_idx, w_idx, 1:5] = assigned_bboxes      # bbox
                # Set class (one-hot encoding)
                targets[b_idx, a_idx, h_idx, w_idx, 5+cls_idx] = 1.             # class

            assigned_labels.append(targets)

        return assigned_labels

    def single_grid_assignment(self, bboxes, cls, grid_size, anchor_boxes):
        grid_h, grid_w = grid_size

        batch_idx, x_center, y_center, width, height = bboxes.unbind(dim=1)
        batch_idx = batch_idx.long()

        width = width.unsqueeze(1).repeat(1, self.num_anchor_per_grid)
        height = height.unsqueeze(1).repeat(1, self.num_anchor_per_grid)

        w_idx = (x_center * grid_w).long()
        h_idx = (y_center * grid_h).long()

        anchor_width, anchor_height = (anchor_boxes.clone().view(self.num_anchor_per_grid, 2) / self.img_size).unbind(dim=1)

        iou = self.cal_single_grid_iou(width, height, anchor_width, anchor_height)

        # Match the anchor using area threshold
        idx = iou >= self.match_iou_thres

        b_idx = batch_idx.clone().unsqueeze(-1).expand(-1, self.num_anchor_per_grid)
        a_idx = torch.nonzero(idx, as_tuple=True)[1]
        h_idx = h_idx.unsqueeze(-1).expand(-1, self.num_anchor_per_grid)
        w_idx = w_idx.unsqueeze(-1).expand(-1, self.num_anchor_per_grid)
        b_idx = b_idx[idx].view(-1)
        h_idx = h_idx[idx].view(-1)
        w_idx = w_idx[idx].view(-1)

        cls_idx = cls.clone().unsqueeze(-1).expand(-1, self.num_anchor_per_grid)
        cls_idx = cls_idx[idx].view(-1)

        assigned_bboxes = bboxes.clone().unsqueeze(1).expand(-1, self.num_anchor_per_grid, -1)
        assigned_bboxes = assigned_bboxes[idx][:, 1:]

        return (b_idx, a_idx, h_idx, w_idx, cls_idx), assigned_bboxes

    def all_grid_assignment(self, bboxes, classes, batch_size, anchor_grid):
        device = bboxes.device
        A, H, W, _ = anchor_grid.shape

        bboxes_xyxy = xywh_to_xyxy(bboxes[:, 1:])
        bboxes_xyxy = bboxes_xyxy[:, None, None, None, :]

        iou_maps = box_iou(bboxes_xyxy, anchor_grid[None, :, :, :, :])

        valid_mask = iou_maps > self.match_iou_thres
        valid_idx = valid_mask.nonzero(as_tuple=False)
        if valid_idx.numel() == 0:
            # Return empty indices and boxes when no matches found
            empty_idx = (torch.tensor([], dtype=torch.long, device=device),  # b_idx
                        torch.tensor([], dtype=torch.long, device=device),   # a_idx
                        torch.tensor([], dtype=torch.long, device=device),   # h_idx
                        torch.tensor([], dtype=torch.long, device=device),   # w_idx
                        torch.tensor([], dtype=torch.long, device=device))  # cls_idx
            empty_boxes = torch.zeros((0, 4), device=device, dtype=bboxes.dtype)
            return empty_idx, empty_boxes

        n, a, h, w = valid_idx.unbind(1)
        b = bboxes[n, 0].long()
        cls_id = classes[n]
        boxes = bboxes[n, 1:]

        if self.filter_by_max_iou:
            ious = iou_maps[n, a, h, w]

            # Flatten spatial indices to resolve conflicts: (b, a, h, w) -> unique scalar index
            lin_idx = b * (A * H * W) + a * (H * W) + h * W + w      # (M,)

            # Sort by IoU ascending: last one per lin_idx will be the highest IoU
            sorted_ious, sorted_order = torch.sort(ious)
            lin_idx_sorted = lin_idx[sorted_order]
            index_buffer = torch.full((batch_size * A * H * W,), -1, dtype=torch.long, device=device)
            index_buffer[lin_idx_sorted] = sorted_order

            keep = index_buffer[index_buffer >= 0]
            b_idx = b[keep]
            a_idx = a[keep]
            h_idx = h[keep]
            w_idx = w[keep]
            cls_idx = cls_id[keep]
            assigned_bboxes = boxes[keep]

            return (b_idx, a_idx, h_idx, w_idx, cls_idx), assigned_bboxes
        else:
            return (b, a, h, w, cls_id), boxes

    def nearby_grid_assignment(self, bboxes, cls, grid_size, anchor_boxes):
        grid_h, grid_w = grid_size
        device = bboxes.device

        N = bboxes.shape[0]
        A = self.num_anchor_per_grid

        anchor_wh = anchor_boxes.view(A, 2).to(device) / self.img_size
 
        if bboxes.shape[1] == 5:
            batch_idx, x_center, y_center, width, height = bboxes.unbind(dim=1)
            D = 4
        else:
            ub = bboxes.unbind(dim=1)
            batch_idx, x_center, y_center, width, height = ub[0], ub[1], ub[2], ub[3], ub[4]
            D = 12
            #kp_x = bboxes[...,5::2]
            #kp_y = bboxes[...,6::2]
            #kp_offset_x = (kp_x - x_center) / (width * 0.5)
            #kp_offset_y = (kp_y - y_center) / (height * 0.5)
            #bboxes = bboxes.clone()
            #bboxes[...,5::2] = kp_offset_x
            #bboxes[...,6::2] = kp_offset_y
 
        batch_idx = batch_idx.long()
        iou = self.cal_single_grid_iou(width.unsqueeze(1), 
                                       height.unsqueeze(1), 
                                       anchor_wh[:, 0].unsqueeze(0), 
                                       anchor_wh[:, 1].unsqueeze(0))

        shifts = torch.tensor([[-1, -1], [-1, 0], [-1, 1],
                               [ 0, -1], [ 0, 0], [ 0, 1],
                               [ 1, -1], [ 1, 0], [ 1, 1]], device=device)
        num_shifts = shifts.shape[0]

        h_center = (y_center * grid_h).long()
        w_center = (x_center * grid_w).long()

        h_idx = (h_center[:, None] + shifts[:, 0]).clamp(0, grid_h - 1)
        w_idx = (w_center[:, None] + shifts[:, 1]).clamp(0, grid_w - 1)

        iou = iou[:, :, None].expand(N, A, num_shifts)
        h_idx = h_idx[:, None, :].expand(N, A, num_shifts)
        w_idx = w_idx[:, None, :].expand(N, A, num_shifts)
        b_idx = batch_idx[:, None, None].expand(N, A, num_shifts)
        a_idx = torch.arange(A, device=device)[None, :, None].expand(N, A, num_shifts)
        cls_idx = cls[:, None, None].expand(N, A, num_shifts)
        boxes = bboxes[:, 1:].unsqueeze(1).unsqueeze(2).expand(N, A, num_shifts, D)

        iou = iou.reshape(-1)
        h_idx = h_idx.reshape(-1)
        w_idx = w_idx.reshape(-1)
        b_idx = b_idx.reshape(-1)
        a_idx = a_idx.reshape(-1)
        cls_idx = cls_idx.reshape(-1)
        boxes = boxes.reshape(-1, D)


        valid_mask = iou >= self.match_iou_thres
        iou = iou[valid_mask]
        h_idx = h_idx[valid_mask]
        w_idx = w_idx[valid_mask]
        b_idx = b_idx[valid_mask]
        a_idx = a_idx[valid_mask]
        cls_idx = cls_idx[valid_mask]
        boxes = boxes[valid_mask]

        if self.filter_by_max_iou:
            # Linear index to resolve conflicts (only keep max IoU)
            lin_idx = b_idx * (A * grid_h * grid_w) + a_idx * (grid_h * grid_w) + h_idx * grid_w + w_idx

            # Sort by lin_idx and IoU (descending)
            sorted_iou, sort_idx = iou.sort(descending=True)
            lin_idx = lin_idx[sort_idx]
            b_idx = b_idx[sort_idx]
            a_idx = a_idx[sort_idx]
            h_idx = h_idx[sort_idx]
            w_idx = w_idx[sort_idx]
            cls_idx = cls_idx[sort_idx]
            boxes = boxes[sort_idx]

            # Keep only the first occurrence per unique lin_idx (highest IoU)
            # keep = torch.unique_consecutive(lin_idx, return_index=True)[1]
            lin_diff = torch.ones_like(lin_idx, dtype=torch.bool)
            lin_diff[1:] = lin_idx[1:] != lin_idx[:-1]
            keep = lin_diff.nonzero(as_tuple=False).squeeze(1)

            return (b_idx[keep], a_idx[keep], h_idx[keep], w_idx[keep], cls_idx[keep]), boxes[keep]
        else:
            return (b_idx, a_idx, h_idx, w_idx, cls_idx), boxes

    def cal_single_grid_iou(self, width, height, anchor_width, anchor_height, eps=1e-6):
        intersection_width = torch.minimum(width, anchor_width)
        intersection_height = torch.minimum(height, anchor_height)
        intersection_area = intersection_width * intersection_height

        label_area = width * height
        anchor_area = anchor_width * anchor_height
        union_area = label_area + anchor_area - intersection_area

        iou = intersection_area / (union_area + eps)
        return iou





"""
Create by:  zh320
Date:       2024/06/15
"""

from typing import Any
import torch, math
import torch.nn as nn
import torch.nn.functional as F

from src.models.yolo.utils.utils import make_anchors

from ..utils import xywh_to_xyxy, box_iou, xyxy_to_xywh, dist2bbox


def get_loss_fn(config):
    criterion = YOLOLoss(num_class=config.num_class, img_size=config.img_size, anchor_boxes=config.anchor_boxes,
                        lambda_coord=config.lambda_coord, lambda_obj=config.lambda_obj, lambda_noobj=config.lambda_noobj, 
                        lambda_scales=config.lambda_scales, use_noobj_loss=config.use_noobj_loss, iou_loss_type=config.iou_loss_type,
                        label_assignment_method=config.label_assignment_method, grid_sizes=config.grid_sizes,
                        focal_loss_gamma=config.focal_loss_gamma, match_iou_thres=config.match_iou_thres,
                        filter_by_max_iou=config.filter_by_max_iou, assign_conf_method=config.assign_conf_method)

    return criterion


class YOLOLoss(nn.Module):
    def __init__(self, num_class, img_size, anchor_boxes, lambda_coord=0.05, lambda_obj=1.0, lambda_noobj=0.5, 
                    lambda_scales=[4.0, 1.0, 0.25], use_noobj_loss=False, label_assignment_method='nearby_grid', 
                    grid_sizes=None, iou_loss_type='iou', focal_loss_gamma=0., match_iou_thres=0.0625, 
                    downsample_rate=[8, 16, 32], assign_conf_method='iou', filter_by_max_iou=True):
        super().__init__()
        assert label_assignment_method in ['single_grid', 'all_grid', 'nearby_grid']
        self.label_assignment_method = label_assignment_method
        assert assign_conf_method in ['iou', 'constant']
        self.assign_conf_method = assign_conf_method

        self.num_class = num_class
        self.num_attrib = 5 + num_class
        self.register_buffer('img_size', torch.tensor(img_size))
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_scales = lambda_scales
        self.use_noobj_loss = use_noobj_loss
        self.match_iou_thres = match_iou_thres
        self.downsample_rate = downsample_rate
        self.filter_by_max_iou = filter_by_max_iou

        self.num_anchor_per_grid = len(anchor_boxes[0])
        self.register_buffer('anchor_boxes', torch.stack([torch.tensor(anch, dtype=torch.float32).view(1, self.num_anchor_per_grid, 1, 1, 2) \
                                                            for anch in anchor_boxes]))

        if grid_sizes is None:
            grid_sizes = [(img_size[0]//downsample_rate[i], img_size[1]//downsample_rate[i]) for i in range(len(downsample_rate))]
        assert len(lambda_scales) == len(grid_sizes)
        self.grid_sizes = grid_sizes

        if label_assignment_method == 'all_grid':
            anchs = [anch.squeeze(0)/torch.tensor(img_size) for anch in self.anchor_boxes]
            for i, (anch, grid_size) in enumerate(zip(anchs, grid_sizes)):
                anchor_grid = self.build_anchor_grid(anch, grid_size)
                self.register_buffer(f'anchor_grid_{i}', anchor_grid)

        self.bce_loss_func = FocalLoss(gamma=focal_loss_gamma)
        self.iou_loss_func = IoULoss(iou_loss_type)

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

    def forward(self, predictions, bboxes, classes):
        device = predictions[0].device
        batch_size = predictions[0].shape[0]
        num_labels = len(bboxes)
        num_pred_layer = len(predictions)

        if num_labels:
            assigned_labels = self.label_assignment(bboxes, classes, batch_size, num_pred_layer)
            assert len(predictions) == len(assigned_labels)

        total_loss, conf_loss_value, iou_loss_value, class_loss_value = 0., 0., 0., 0.
        for i in range(num_pred_layer):
            # Extract predicted confidence scores
            predicted = predictions[i]
            pred_conf = predicted[..., 0]

            if num_labels:
                # Extract predicted box coordinates and class probabilities
                pred_x = predicted[..., 1:2]
                pred_y = predicted[..., 2:3]
                pred_wh = predicted[..., 3:5]
                pred_class = predicted[..., 5:]

                # Calculate x_shift and y_shift to decode predicted box coordinates
                grid_h, grid_w = predicted.size()[2:4]
                x_shift = torch.arange(grid_w).view(1, 1, 1, grid_w).repeat(batch_size, self.num_anchor_per_grid, grid_h, 1).to(device)
                y_shift = torch.arange(grid_h).view(1, 1, grid_h, 1).repeat(batch_size, self.num_anchor_per_grid, 1, grid_w).to(device)
                x_shift = x_shift.unsqueeze(-1)
                y_shift = y_shift.unsqueeze(-1)

                # Extract assigned confidence scores, box coordinates, and class probabilities
                assigned = assigned_labels[i].to(device)
                assigned_conf = assigned[..., 0]
                assigned_coord = assigned[..., 1:5]
                assigned_class = assigned[..., 5:]

                # Only consider positive examples for iou loss and class loss
                pos_mask = assigned_conf > 0

                # Compute IoU loss for normalized bounding box coordinates (need to avoid in-place operation)
                if self.label_assignment_method == 'single_grid':
                    pred_x_decode = (pred_x.sigmoid() + x_shift) * self.downsample_rate[i] / self.img_size[0]
                    pred_y_decode = (pred_y.sigmoid() + y_shift) * self.downsample_rate[i] / self.img_size[1]
                elif self.label_assignment_method == 'all_grid':
                    pred_x_decode = (pred_x.tanh() * grid_w + x_shift) * self.downsample_rate[i] / self.img_size[0]
                    pred_y_decode = (pred_y.tanh() * grid_h + y_shift) * self.downsample_rate[i] / self.img_size[1]
                elif self.label_assignment_method == 'nearby_grid':
                    pred_x_decode = (pred_x.tanh() * 1.5 + 0.5 + x_shift) * self.downsample_rate[i] / self.img_size[0]
                    pred_y_decode = (pred_y.tanh() * 1.5 + 0.5 + y_shift) * self.downsample_rate[i] / self.img_size[1]
                else:
                    raise NotImplementedError

                pred_wh_decode = pred_wh * 4 * self.anchor_boxes[i] / self.img_size
                pred_coord = torch.cat([pred_x_decode, pred_y_decode, pred_wh_decode], dim=-1)

                raw_iou_loss = self.iou_loss_func(pred_coord[pos_mask], assigned_coord[pos_mask], xywh=True)
                iou_loss = raw_iou_loss.mean()

                if self.assign_conf_method == 'iou':
                    assigned_conf[pos_mask] = (1 - raw_iou_loss).detach().clamp(min=0., max=1.)

                if self.num_class > 1:
                    # Compute class loss when there are multiple classes (binary cross-entropy)
                    class_loss = self.bce_loss_func(pred_class[pos_mask], assigned_class[pos_mask])
                else:
                    class_loss = torch.tensor(0.)
            else:
                assigned_conf = torch.zeros_like(pred_conf, device=device)

                iou_loss, class_loss = torch.tensor(0.), torch.tensor(0.)

            # Compute confidence/object loss (binary cross-entropy)
            conf_loss = self.bce_loss_func(pred_conf, assigned_conf)

            # Loss for one detection layer
            loss_per_layer = self.lambda_obj * conf_loss + self.lambda_coord * iou_loss + class_loss

            if self.use_noobj_loss:
                loss_per_layer += self.lambda_noobj * (1 - assigned_conf) * conf_loss

            # Loss for all detection layers
            total_loss += self.lambda_scales[i] * loss_per_layer

            # Record loss values
            conf_loss_value += self.lambda_scales[i] * conf_loss.item()
            iou_loss_value += self.lambda_scales[i] * iou_loss.item()
            class_loss_value += self.lambda_scales[i] * class_loss.item()

        return total_loss, (conf_loss_value, iou_loss_value, class_loss_value)

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
            return torch.zeros((batch_size, A, H, W, self.num_attrib), device=device)

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

        batch_idx, x_center, y_center, width, height = bboxes.unbind(dim=1)
        batch_idx = batch_idx.long()
        N = bboxes.shape[0]
        A = self.num_anchor_per_grid

        anchor_wh = anchor_boxes.view(A, 2).to(device) / self.img_size
        iou = self.cal_single_grid_iou(width.unsqueeze(1), height.unsqueeze(1), anchor_wh[:, 0].unsqueeze(0), 
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
        boxes = bboxes[:, 1:].unsqueeze(1).unsqueeze(2).expand(N, A, num_shifts, 4)

        iou = iou.reshape(-1)
        h_idx = h_idx.reshape(-1)
        w_idx = w_idx.reshape(-1)
        b_idx = b_idx.reshape(-1)
        a_idx = a_idx.reshape(-1)
        cls_idx = cls_idx.reshape(-1)
        boxes = boxes.reshape(-1, 4)

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


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, labels):
        bce_loss_func = F.binary_cross_entropy_with_logits(preds, labels, reduction='none')
        pt = torch.exp(-bce_loss_func)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss_func

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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

        # Calculate intersection and union
        intersection = self.intersection_area(pred_boxes, target_boxes)
        union = self.union_area(pred_boxes, target_boxes)
        iou = intersection / (union + self.eps)

        if self.iou_loss_type in ['ciou', 'diou', 'siou']:
            pred_center = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2
            target_center = (target_boxes[..., :2] + target_boxes[..., 2:]) / 2
            center_distance = torch.norm(pred_center - target_center, p=2, dim=1)

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

        return 1 - iou

    def intersection_area(self, boxes1, boxes2):
        tl = torch.max(boxes1[..., :2], boxes2[..., :2])
        br = torch.min(boxes1[..., 2:], boxes2[..., 2:])
        wh = (br - tl).clamp(min=0)
        return wh[..., 0] * wh[..., 1]

    def union_area(self, boxes1, boxes2):
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        return area1 + area2 - self.intersection_area(boxes1, boxes2)

    def enclosing_area(self, boxes1, boxes2):
        tl = torch.min(boxes1[..., :2], boxes2[..., :2])
        br = torch.max(boxes1[..., 2:], boxes2[..., 2:])
        wh = (br - tl).clamp(min=0)
        return wh[..., 0] * wh[..., 1]

    def diagonal_length_squared(self, boxes1, boxes2):
        tl = torch.min(boxes1[..., :2], boxes2[..., :2])
        br = torch.max(boxes1[..., 2:], boxes2[..., 2:])
        return ((br - tl) ** 2).sum(dim=1)

    def arctan(self, x):
        return torch.atan(x)
    
    
class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        # torch 1/num points
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor#, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        #kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * 2)#* (area + 1e-9) * 2)  # from cocoeval
        #return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()
        return (1-torch.exp(-e)).mean()
    
class PoseLoss(torch.nn.Module):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int = 10):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        self.kpt_shape = [4,3]
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [4, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(1/4).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh_to_xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl)
        
    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        # Pboxes
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # Bbox loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select target keypoints for each anchor based on batch index and target ground truth index.

        Args:
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).

        Returns:
            (torch.Tensor): Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        # Select target keypoints using helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy_to_xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss



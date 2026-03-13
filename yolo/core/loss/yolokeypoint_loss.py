import torch
from yolo.core.loss.yolo_loss import YOLOLoss
from yolo.core.loss.focal_loss import FocalLoss
from yolo.core.loss.iou_loss import IoULoss
import logging
import torch.nn.functional as F
from typing import Dict, Tuple
from yolo.core.loss.registry import LossRegistry
from yolo.models.yolo import decode_boxes, decode_kpts
loss_registry = LossRegistry()

@loss_registry.register
class YOLOKeypointLoss(YOLOLoss):
    def __init__(self, config):
        super().__init__(config)
        assert config.label_assignment_method in ['single_grid', 'all_grid', 'nearby_grid']
        self.label_assignment_method = config.label_assignment_method
        assert config.assign_conf_method in ['iou', 'constant']
        self.assign_conf_method = config.assign_conf_method

        self.num_class = config.num_class_kp
        self.num_attrib = 5 + self.num_class + (self.kp_shape[0] * self.kp_shape[1])
        self.kp_offset = 5 + self.num_class

    def forward(self, predictions: torch.Tensor, targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        '''
        '''
        gt_kps = targets['keypoints']
        classes = targets['keypoints_classes']
        gt_kp_boxes = targets['keypoints_bboxes']
        device = predictions[0].device
        batch_size = predictions[0].shape[0]
        num_kpts = predictions[0].shape[1]
        num_pred_layer = len(predictions)
        
        iou_loss   = torch.tensor(0., device=device, dtype=predictions[0].dtype)
        kp_loss    = torch.tensor(0., device=device, dtype=predictions[0].dtype)
        class_loss = torch.tensor(0., device=device, dtype=predictions[0].dtype)
 
        #assigned_labels = self.label_assignment(bboxes, classes, batch_size, num_pred_layer)
        #assert len(predictions) == len(assigned_labels)
        if num_kpts:
            assigned_labels = self.label_assignment(gt_kp_boxes, gt_kps, classes, batch_size, num_pred_layer)
            assert len(predictions) == len(assigned_labels)

        total_loss, conf_loss_value, iou_loss_value, class_loss_value, kp_loss_value = 0., 0., 0., 0., 0.
        for i in range(num_pred_layer):
            # Extract predicted confidence scores
            predicted = predictions[i]
            pred_conf = predicted[..., 0]
            assigned_conf = torch.zeros_like(pred_conf, device=device)
            
            if num_kpts:
                # Extract predicted box coordinates and class probabilities
                pred_x_box = predicted[..., 1][...,None]
                pred_y_box = predicted[..., 2][...,None]
                pred_wh = predicted[..., 3:5]
                pred_x = predicted[...,self.kp_offset::2]
                pred_y = predicted[...,self.kp_offset+1::2]
                #pred_kp = predicted[...,self.kp_offset:]
                pred_class = predicted[..., 5:self.kp_offset]

                # Calculate x_shift and y_shift to decode predicted box coordinates
                grid_h, grid_w = predicted.size()[2:4]
                x_shift = torch.arange(grid_w).view(1, 1, 1, grid_w).repeat(batch_size, num_kpts, grid_h, 1).to(device)
                y_shift = torch.arange(grid_h).view(1, 1, grid_h, 1).repeat(batch_size, num_kpts, 1, grid_w).to(device)
                x_shift = x_shift.unsqueeze(-1)
                y_shift = y_shift.unsqueeze(-1)

                assigned = assigned_labels[i].to(device)
                assigned_conf = assigned[..., 0]
                assigned_coord = assigned[..., 1:5]
                assigned_kp = assigned[..., self.kp_offset:]
                assigned_class = assigned[..., 5:self.kp_offset]
                # Only consider positive examples for iou loss and class loss
                pos_mask = assigned_conf > 0

                pred_wh_decode = (pred_wh.sigmoid() * 2) **2 * self.anchor_boxes[i] / self.img_size
                gt_w = assigned_coord[..., 2:3].clone().detach()  # fixed GT, not predicted
                gt_h = assigned_coord[..., 3:4].clone().detach()

                # Compute IoU loss for normalized bounding box coordinates (need to avoid in-place operation)
                if self.label_assignment_method == 'single_grid':
                    pred_x_box_decode = (pred_x_box.sigmoid() + x_shift) * self.downsample_rate[i] / self.img_size[0]
                    pred_y_box_decode = (pred_y_box.sigmoid() + y_shift) * self.downsample_rate[i] / self.img_size[1]
                elif self.label_assignment_method == 'all_grid':
                    pred_x_box_decode = (pred_x_box.tanh() * grid_w + x_shift) * self.downsample_rate[i] / self.img_size[0]
                    pred_y_box_decode = (pred_y_box.tanh() * grid_h + y_shift) * self.downsample_rate[i] / self.img_size[1]
                elif self.label_assignment_method == 'nearby_grid':
                    pred_x_box_decode = (pred_x_box.tanh() * 1.5 + 0.5 + x_shift) * self.downsample_rate[i] / self.img_size[0]
                    pred_y_box_decode = (pred_y_box.tanh() * 1.5 + 0.5 + y_shift) * self.downsample_rate[i] / self.img_size[1]
                    pred_kp_x_decode = pred_x_box_decode.detach() + (pred_x.tanh() * gt_w) 
                    pred_kp_y_decode = pred_y_box_decode.detach() + (pred_y.tanh() * gt_h)

                else:
                    raise NotImplementedError
                # reassemble the decoded keypoint x and y coordinates into the same shape as the original predictions for loss calculation
                kp_decode = torch.stack((pred_kp_x_decode, pred_kp_y_decode), dim=-1).flatten(-2)
                
                # Use exp to ensure positive width/height values and avoid NaN in IoU loss
                pred_coord = torch.cat([pred_x_box_decode, pred_y_box_decode, pred_wh_decode], dim=-1)
                
                raw_iou_loss = None
                if pos_mask.sum() > 0:
                    raw_iou_loss = self.iou_loss_func(pred_coord[pos_mask], assigned_coord[pos_mask], xywh=True)
                    # Ensure no NaN values in IoU loss
                    raw_iou_loss = torch.clamp(raw_iou_loss, min=0, max=2)
                    iou_loss = raw_iou_loss.mean()
                    kp_loss = F.smooth_l1_loss(kp_decode[pos_mask], assigned_kp[pos_mask], reduction='mean')
                    
                    if self.assign_conf_method == 'iou':
                        assigned_conf[pos_mask] = (1 - raw_iou_loss).detach().clamp(min=0., max=1.)
                    
                    if self.num_class > 1:
                        # Compute class loss when there are multiple classes (binary cross-entropy)
                        class_loss = self.bce_loss_func(pred_class[pos_mask], assigned_class[pos_mask])
 
                    # Compute confidence/object loss (binary cross-entropy)
                    conf_loss = self.bce_loss_func(pred_conf, assigned_conf)
        
                    loss_per_layer = (self.lambda_kp * kp_loss + 
                                      self.lambda_coord * iou_loss + 
                                      self.lambda_obj * conf_loss + 
                                       class_loss )
                    if self.use_noobj_loss:
                        loss_per_layer += self.lambda_noobj * (1 - assigned_conf) * conf_loss
                    
                    loss_per_layer = torch.clamp(loss_per_layer, min=0, max=1e4)
                    if not torch.isnan(loss_per_layer):
                        total_loss += self.lambda_scales[i] * loss_per_layer


                    kp_loss_value += self.lambda_scales[i] * kp_loss.item()
                    iou_loss_value += self.lambda_scales[i] * iou_loss.item()
                    conf_loss_value += self.lambda_scales[i] * conf_loss.item()
                    class_loss_value += self.lambda_scales[i] * class_loss.item()
             
        pos_cnt = pos_mask.sum().item()
        pos_ratio_kp = 0
        if pos_cnt > 0:
            pos_ratio_kp = num_kpts / pos_cnt
        
        loss_dict = {
            'kp_conf': conf_loss_value,
            'kp_iou': iou_loss_value,
            'kp': kp_loss_value,
            'kp_cls' : class_loss_value,
            'pos_ratio_kp': pos_ratio_kp
        }
        return total_loss, loss_dict


    def label_assignment(self, bboxes, keypoints, classes, batch_size, num_pred_layer):
        _, cls = classes.unbind(dim=1)
        cls = cls.long()

        bboxes = torch.cat((bboxes, keypoints[:,1:]), 1)

        assigned_labels = []
        for i in range(num_pred_layer):
            targets = torch.zeros((batch_size, self.num_anchor_per_grid, *self.grid_sizes[i], self.num_attrib), device=bboxes.device) 

            if len(bboxes):
                if self.label_assignment_method == 'single_grid':
                    assigned_idx, assigned = self.single_grid_assignment(bboxes, cls, self.grid_sizes[i], self.anchor_boxes[i])

                elif self.label_assignment_method == 'all_grid':
                    assigned_idx, assigned = self.all_grid_assignment(bboxes, cls, batch_size, getattr(self, f'anchor_grid_{i}'))

                elif self.label_assignment_method == 'nearby_grid':
                    assigned_idx, assigned = self.nearby_grid_assignment(bboxes, cls, self.grid_sizes[i], self.anchor_boxes[i])

                else:
                    raise NotImplementedError(f'Unsupported label assignment method: {self.label_assignment_method}\n')

                b_idx, a_idx, h_idx, w_idx, cls_idx = assigned_idx
                # Set confidence score to 1
                targets[b_idx, a_idx, h_idx, w_idx, 0] = 1.                     # confidence
                # Set box coordinatesa
                cls_offset = 5+self.num_class
                assigned_boxes = assigned[...,:4]
                assigned_kps = assigned[..., 4:]
                targets[b_idx, a_idx, h_idx, w_idx, 1:5] = assigned_boxes      # bbox
                targets[b_idx, a_idx, h_idx, w_idx, cls_offset:] = assigned_kps      # bbox
                ## Set class (one-hot encoding)
                targets[b_idx, a_idx, h_idx, w_idx, 5+cls_idx] = 1

            assigned_labels.append(targets)

        return assigned_labels


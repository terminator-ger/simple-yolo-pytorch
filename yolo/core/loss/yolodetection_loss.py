import torch
from yolo.core.loss.yolo_loss import YOLOLoss
import logging
from yolo.core.loss.registry import LossRegistry
loss_registry = LossRegistry()

@loss_registry.register
class YOLODetectionLoss(YOLOLoss):
    def __init__(self, config):
        super().__init__(config)

    def forward_bboxes(self, predictions, bboxes, classes):
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

                # Use exp to ensure positive width/height values and avoid NaN in IoU loss
                # TODO: only clamp no exp
                pred_wh_decode = torch.exp(pred_wh.clamp(min=-10, max=10)) * self.anchor_boxes[i] / self.img_size
                pred_coord = torch.cat([pred_x_decode, pred_y_decode, pred_wh_decode], dim=-1)

                class_loss = torch.tensor(0., device=device, dtype=predicted.dtype)
                iou_loss = torch.tensor(0., device=device, dtype=predicted.dtype)
                raw_iou_loss = None
                
                if pos_mask.sum() > 0:
                    raw_iou_loss = self.iou_loss_func(pred_coord[pos_mask], assigned_coord[pos_mask], xywh=True)
                    # Ensure no NaN values in IoU loss
                    raw_iou_loss = torch.clamp(raw_iou_loss, min=0, max=2)
                    iou_loss = raw_iou_loss.mean()
                    
                    if self.assign_conf_method == 'iou':
                        assigned_conf[pos_mask] = (1 - raw_iou_loss).detach().clamp(min=0., max=1.)

                if self.num_class > 1 and pos_mask.sum() > 0:
                    # Compute class loss when there are multiple classes (binary cross-entropy)
                    class_loss = self.bce_loss_func(pred_class[pos_mask], assigned_class[pos_mask])
            else:
                logging.warning("No Assignments during loss calculation")
                assigned_conf = torch.zeros_like(pred_conf, device=device)

                iou_loss = torch.tensor(0., device=device, dtype=predicted.dtype)
                class_loss = torch.tensor(0., device=device, dtype=predicted.dtype)

            # Compute confidence/object loss (binary cross-entropy)
            conf_loss = self.bce_loss_func(pred_conf, assigned_conf)
            
            # Clamp loss values to prevent NaN propagation
            conf_loss = torch.clamp(conf_loss, min=0, max=1e4)
            iou_loss = torch.clamp(iou_loss, min=0, max=1e4)
            class_loss = torch.clamp(class_loss, min=0, max=1e4)

            # Loss for one detection layer
            loss_per_layer = self.lambda_obj * conf_loss + self.lambda_coord * iou_loss + class_loss

            if self.use_noobj_loss:
                loss_per_layer += self.lambda_noobj * (1 - assigned_conf) * conf_loss

            # Loss for all detection layers
            loss_per_layer = torch.clamp(loss_per_layer, min=0, max=1e4)
            if not torch.isnan(loss_per_layer):
                total_loss += self.lambda_scales[i] * loss_per_layer

            # Record loss values
            conf_loss_value += self.lambda_scales[i] * conf_loss.item()
            iou_loss_value += self.lambda_scales[i] * iou_loss.item()
            class_loss_value += self.lambda_scales[i] * class_loss.item()
        
        pos_num = (num_labels if num_labels else 0) / pos_mask.sum().item()
        loss_dict = {
            'conf': conf_loss_value,
            'iou': iou_loss_value,
            'class': class_loss_value,
            'pos_ratio_stone': pos_num
        }
        return total_loss, loss_dict



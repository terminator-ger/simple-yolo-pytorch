"""
Create by:  zh320
Date:       2024/06/15
"""

from numpy import copy
import torch
import torch.nn as nn

from .backbone import ResNet
from .darknet import DarkNet
from .modules import SPP, PAN, ConvBNAct, conv1x1, replace_act
from .mresnet import modify_resnet


class YOLO(nn.Module):
    def __init__(self, num_class=1, backbone_type='resnet18', label_assignment_method='nearby_grid', 
                    anchor_boxes=None, act_type='relu', channel_sparsity=0.75):
        super().__init__()
        assert label_assignment_method in ['single_grid', 'all_grid', 'nearby_grid']

        if 'resnet' in backbone_type:
            resnet = ResNet(backbone_type)
            if channel_sparsity != 1.:
                assert channel_sparsity < 1, '`channel_sparsity` should be less than or equal to 1'
                resnet = modify_resnet(resnet, channel_sparsity)

            self.backbone = resnet
            last_channel = int(512*channel_sparsity) if backbone_type in ['resnet18', 'resnet34'] else int(2048*channel_sparsity)

            # Change activations of torchvision pretrained model
            if act_type != 'relu':
                replace_act(self.backbone, act_type, nn.ReLU)

        elif backbone_type == 'darknet':
            self.backbone = DarkNet(act_type=act_type, channel_sparsity=channel_sparsity)
            last_channel = int(1024 * channel_sparsity)
        else:
            raise NotImplementedError()

        self.spp = SPP(last_channel, last_channel, act_type)

        self.pan = PAN(last_channel, act_type)

        self.det_head = YOLOHead(last_channel, num_class, label_assignment_method, anchor_boxes)
        self.kp_head = YOLOPoseHEAD(num_class=num_class, ch=last_channel, anchor_boxes=anchor_boxes, label_assignment_method=label_assignment_method)

    def forward(self, x, is_training=True):
        _, x2, x3, x4 = self.backbone(x)

        x4 = self.spp(x4)

        res1, res2, res3 = self.pan(x4, x3, x2)

        x1 = self.det_head([res1, res2, res3], is_training=is_training)
        x2 = self.kp_head([res1, res2, res3], is_training=is_training)

        return {'det': x1, 'pose':x2}


class YOLOHead(nn.Module):
    def __init__(self, in_channel, num_class, label_assignment_method, anchor_boxes):
        super().__init__()
        self.label_assignment_method = label_assignment_method

        assert anchor_boxes is not None, 'Anchor boxes must be given.\n'
        self.anchor_boxes = torch.tensor(anchor_boxes)
        assert self.anchor_boxes.shape[0] * self.anchor_boxes.shape[1] != 0
        assert self.anchor_boxes.shape[2] == 2  # width, height

        self.num_anchor = self.anchor_boxes.shape[1]
        self.num_attrib = num_class + 5     # 5: conf, dx, dy, dw, dh
        out_channel = self.num_anchor * self.num_attrib
        self.downsample_rate = torch.tensor([8, 16, 32])

        self.heads = nn.ModuleList([conv1x1(in_channel//2**(2-i), out_channel) for i in range(3)])

    def forward(self, feats, is_training=True):
        device = feats[0].device

        out = []
        for i, head in enumerate(self.heads):
            feat = head(feats[i])
            batch_size, _, grid_h, grid_w = feat.size()
            feat = feat.view(batch_size, self.num_anchor, self.num_attrib, grid_h, grid_w)
            feat = feat.permute(0, 1, 3, 4, 2).contiguous()

            if is_training:
                out.append(feat)

            else: 
                # When not training, we need to decode the output using position shifts and predefined anchor boxes.
                # The output in the last dimension will be [conf, dx, dy, width, height, class1, class2, ...]
                x_shift = torch.arange(grid_w).view(1, 1, 1, grid_w).repeat(batch_size, self.num_anchor, grid_h, 1).to(device)
                y_shift = torch.arange(grid_h).view(1, 1, grid_h, 1).repeat(batch_size, self.num_anchor, 1, grid_w).to(device)

                base_anchors = (self.anchor_boxes[i].unsqueeze(0).unsqueeze(2).unsqueeze(3)) \
                                .repeat(1, 1, grid_h, grid_w, 1).to(device, dtype=torch.float32)

                # Decode box confidence using sigmoid
                feat[..., 0] = feat[..., 0].sigmoid()

                # Decode box center coords by adding position shifts
                if self.label_assignment_method == 'single_grid':
                    feat[..., 1] = (feat[..., 1].sigmoid() + x_shift) * self.downsample_rate[i]
                    feat[..., 2] = (feat[..., 2].sigmoid() + y_shift) * self.downsample_rate[i]
                elif self.label_assignment_method == 'all_grid':
                    feat[..., 1] = (feat[..., 1].tanh() * grid_w + x_shift) * self.downsample_rate[i]
                    feat[..., 2] = (feat[..., 2].tanh() * grid_h + y_shift) * self.downsample_rate[i]
                elif self.label_assignment_method == 'nearby_grid':
                    feat[..., 1] = (feat[..., 1].tanh() * 1.5 + 0.5 + x_shift) * self.downsample_rate[i]
                    feat[..., 2] = (feat[..., 2].tanh() * 1.5 + 0.5 + y_shift) * self.downsample_rate[i]
                else:
                    raise NotImplementedError

                # Decode box width and height by multiplying predefined anchor boxes
                feat[..., 3:5] = feat[..., 3:5] * 4 * base_anchors

                # Decode class logits using sigmoid
                feat[..., 5:] = feat[..., 5:].sigmoid()

                # Reshape the output in order to perform NMS
                out.append(feat.view(batch_size, -1, self.num_attrib))

        out = out if is_training else torch.cat(out, dim=1)

        return out
    
    
class YOLOPoseHEAD(nn.Module):
    """YOLO Pose head for keypoints models.

    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, num_class: int = 80, kpt_shape: tuple = (4, 3), reg_max=16, ch=32, anchor_boxes=None, label_assignment_method=None):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__()
        self.label_assignment_method = label_assignment_method
        assert anchor_boxes is not None, 'Anchor boxes must be given.\n'
        self.anchor_boxes = torch.tensor(anchor_boxes)
        assert self.anchor_boxes.shape[0] * self.anchor_boxes.shape[1] != 0
        assert self.anchor_boxes.shape[2] == 2  # width, height

        self.num_anchor = self.anchor_boxes.shape[1]
 
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.num_class = num_class
        self.nl = 3
        self.max_det = 12
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = num_class + self.reg_max * 4  
        c4 = max(ch // 4, self.nk)
        self.heads = nn.ModuleList(nn.Sequential(ConvBNAct(ch//2**(2-i), c4, 3), ConvBNAct(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for i in range(3))
        self.downsample_rate = torch.tensor([8, 16, 32])

    def forward(
        self, 
        x: list[torch.Tensor],
        is_training: bool=True
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        bs = x[0].shape[0]  # batch size
        out = []
        for i, head in enumerate(self.heads):
            bs, _, grid_h, grid_w = x[i].shape
            feat = head(x[i])
            feat = feat.view(bs, self.kpt_shape[0], self.kpt_shape[1], grid_h, grid_w)
            feat = feat.permute(0,1,3,4,2)
            
            if is_training:
                out.append(feat)
            else:
                x_shift = torch.arange(grid_w).view(1, 1, 1, grid_w).repeat(bs, self.kpt_shape[0], grid_h, 1).to('cuda')
                y_shift = torch.arange(grid_h).view(1, 1, grid_h, 1).repeat(bs, self.kpt_shape[0], 1, grid_w).to('cuda')

                #base_anchors = (self.anchor_boxes[i].unsqueeze(0).unsqueeze(2).unsqueeze(3)) \
                #                        .repeat(1, 1, grid_h, grid_w, 1).to(dtype=torch.float32)
                feat[...,0] = feat[...,0].sigmoid()
                # Decode box center coords by adding position shifts
                if self.label_assignment_method == 'single_grid':
                    feat[..., 1] = (feat[..., 1].sigmoid() + x_shift) * self.downsample_rate[i]
                    feat[..., 2] = (feat[..., 2].sigmoid() + y_shift) * self.downsample_rate[i]
                elif self.label_assignment_method == 'all_grid':
                    feat[..., 1] = (feat[..., 1].tanh() * grid_w + x_shift) * self.downsample_rate[i]
                    feat[..., 2] = (feat[..., 2].tanh() * grid_h + y_shift) * self.downsample_rate[i]
                elif self.label_assignment_method == 'nearby_grid':
                    feat[..., 1] = (feat[..., 1].tanh() * 1.5 + 0.5 + x_shift) * self.downsample_rate[i]
                    feat[..., 2] = (feat[..., 2].tanh() * 1.5 + 0.5 + y_shift) * self.downsample_rate[i]
                else:
                    raise NotImplementedError
                out.append(feat.reshape(bs, -1, self.kpt_shape[1]))
        
        out = out if is_training else torch.cat(out, dim=1)

        return out
 
        
                

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nk) with last dimension
                format [x, y, w, h, class_probs, keypoints].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + self.nk) and
                last dimension format [x, y, w, h, max_class_prob, class_index, keypoints].
        """
        boxes, scores, kpts = preds.split([4, self.num_class, self.nk], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))
        return torch.cat([boxes, scores, conf, kpts], dim=-1)

    
    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get top-k indices from scores.

        Args:
            scores (torch.Tensor): Scores tensor with shape (batch_size, num_anchors, num_classes).
            max_det (int): Maximum detections per image.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Top scores, class indices, and filtered indices.
        """
        batch_size, anchors, nc = scores.shape  # i.e. shape(16,8400,84)
        # Use max_det directly during export for TensorRT compatibility (requires k to be constant),
        # otherwise use min(max_det, anchors) for safety with small inputs during Python inference
        k = min(max_det, anchors)
        if self.agnostic_nms:
            scores, labels = scores.max(dim=-1, keepdim=True)
            scores, indices = scores.topk(k, dim=1)
            labels = labels.gather(1, indices)
            return scores, labels, indices
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # original index
        return scores[..., None], (index % nc)[..., None].float(), idx
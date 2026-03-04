"""
Create by:  zh320
Date:       2024/06/15
"""

import torch
import torch.nn as nn

from .backbone import ResNet
from .darknet import DarkNet
from .modules import SPP, PAN, ConvBNAct, conv1x1, replace_act
from .mresnet import modify_resnet
from enum import StrEnum, auto

class ModelOutputKeys(StrEnum):
    Detections = auto()
    Keypoints = auto()

class YOLO(nn.Module):
    def __init__(self, num_class=1, backbone_type='resnet18', label_assignment_method='nearby_grid', 
                    anchor_boxes=None, act_type='relu', channel_sparsity=0.75, p2=False, downsample_rate=None):
        super().__init__()
        assert label_assignment_method in ['single_grid', 'all_grid', 'nearby_grid']

        self.p2 = p2
        self.downsample_rate = downsample_rate
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

        self.pan = PAN(last_channel, act_type, p2)

        self.det_head = YOLOHead(last_channel, num_class, label_assignment_method, anchor_boxes, p2, downsample_rate=downsample_rate)
        self.kp_head = YOLOPoseHEAD(kpt_shape=(4,2), num_class=num_class, ch=last_channel, anchor_boxes=anchor_boxes, label_assignment_method=label_assignment_method)

    def forward(self, x, is_training=True):
        x1, x2, x3, x4 = self.backbone(x)

        x4 = self.spp(x4)

        res1, res2, res3, res4 = self.pan(x4, x3, x2, x1 if self.p2 else None)

        det = self.det_head([res1, res2, res3, res4], is_training=is_training)
        kp = self.kp_head([res1, res2, res3, res4], is_training=is_training)


        return {ModelOutputKeys.Detections: det, ModelOutputKeys.Keypoints: kp}



class YOLOHead(nn.Module):
    def __init__(self, in_channel, num_class, label_assignment_method, anchor_boxes, p2=False, downsample_rate=None):
        super().__init__()
        self.label_assignment_method = label_assignment_method
        self.p2 = p2

        assert anchor_boxes is not None, 'Anchor boxes must be given.\n'
        self.anchor_boxes = torch.tensor(anchor_boxes)
        assert self.anchor_boxes.shape[0] * self.anchor_boxes.shape[1] != 0
        assert self.anchor_boxes.shape[2] == 2  # width, height

        self.num_anchor = self.anchor_boxes.shape[1]
        self.num_attrib = num_class + 5     # 5: conf, dx, dy, dw, dh
        out_channel = self.num_anchor * self.num_attrib
        self.downsample_rate = downsample_rate

        N = 4 if p2 else 3
        self.heads = nn.ModuleList([conv1x1(in_channel//2**((N-1)-i), out_channel) for i in range(N)])

    def forward(self, feats, is_training=True):
        device = feats[1].device
        if feats[0] is None:
            feats = feats[1:]
 
        out = []
        for i, head in enumerate(self.heads):
            feat = head(feats[i])
            batch_size, _, grid_h, grid_w = feat.size()
            feat = feat.view(batch_size, self.num_anchor, self.num_attrib, grid_h, grid_w)
            feat = feat.permute(0, 1, 3, 4, 2).contiguous()  #bs, num_anchor, grid_h, grid_w, num_attrib
            # num_attrib: [conf, dx, dy, dw, dh, class1, class2, ...]

            if is_training:
                out.append(feat)

            else: 
                feat = decode_boxes(feat, self.anchor_boxes[i], self.label_assignment_method, self.downsample_rate[i])
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

    def __init__(self, num_class: int = 80, kpt_shape: tuple = (4, 4), reg_max=16, ch=32, anchor_boxes=None, label_assignment_method=None):
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
        self.num_attrib = self.nk + 5 
        c4 = max(ch // 4, self.num_attrib)
        out_channel = self.num_attrib * self.num_anchor
        self.heads = nn.ModuleList(nn.Sequential(ConvBNAct(ch//2**(2-i), c4, 3), ConvBNAct(c4, c4, 3), nn.Conv2d(c4, out_channel, 1)) for i in range(3))
        self.downsample_rate = torch.tensor([8, 16, 32])

    def forward(
        self, 
        feats: list[torch.Tensor],
        is_training: bool=True
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        if feats[0] is None:
            feats = feats[1:]
 
        device = feats[0].device
        bs = feats[0].shape[0]  # batch size
        self.downsample_rate.to(device)
        out = []
        for i, head in enumerate(self.heads):
            batch_size, _, grid_h, grid_w = feats[i].shape
            feat = head(feats[i])
            feat = feat.view(batch_size, self.num_anchor, self.num_attrib, grid_h, grid_w)
            feat = feat.permute(0,1,3,4,2)  #bs, num_anchor, grid_h, grid_w, num_attrib 
            # kpt_dim: [x, y]
            
            if is_training:
                out.append(feat)
            else:
                feat = decode_boxes(feat, self.anchor_boxes[i], self.label_assignment_method, self.downsample_rate[i])
                feat = decode_kpts(feat, self.anchor_boxes[i], self.label_assignment_method, self.downsample_rate[i])
                out.append(feat.reshape(batch_size, -1, self.num_attrib))
        
        out = out if is_training else torch.cat(out, dim=1)

        return out
    

def decode_boxes(feat, anchor_boxes, label_assignment_method, downsample_rate, ):

    batch_size, _, grid_h, grid_w,_ = feat.size()
    device = feat.device
    num_anchor = anchor_boxes.shape[0]
    # When not training, we need to decode the output using position shifts and predefined anchor boxes.
    # The output in the last dimension will be [conf, dx, dy, width, height, class1, class2, ...]
    x_shift = torch.arange(grid_w).view(1, 1, 1, grid_w).repeat(batch_size, num_anchor, grid_h, 1).to(device)
    y_shift = torch.arange(grid_h).view(1, 1, grid_h, 1).repeat(batch_size, num_anchor, 1, grid_w).to(device)

    base_anchors = (anchor_boxes.unsqueeze(0).unsqueeze(2).unsqueeze(3)) \
                    .repeat(1, 1, grid_h, grid_w, 1).to(device, dtype=torch.float32)

    # Decode box confidence using sigmoid
    feat[..., 0] = feat[..., 0].sigmoid()

    # Decode box center coords by adding position shifts
    if label_assignment_method == 'single_grid':
        feat[..., 1] = (feat[..., 1].sigmoid() + x_shift) * downsample_rate
        feat[..., 2] = (feat[..., 2].sigmoid() + y_shift) * downsample_rate
        #feat[..., 5::2] = (feat[..., 5::2].sigmoid() + x_shift) * downsample_rate
        #feat[..., 6::2] = (feat[..., 6::2].sigmoid() + y_shift) * downsample_rate
    elif label_assignment_method == 'all_grid':
        feat[..., 1] = (feat[..., 1].tanh() * grid_w + x_shift) * downsample_rate
        feat[..., 2] = (feat[..., 2].tanh() * grid_h + y_shift) * downsample_rate
        #feat[..., 5::2] = (feat[..., 5::2].tanh() * grid_w + x_shift) * downsample_rate
        #feat[..., 6::2] = (feat[..., 6::2].tanh() * grid_h + y_shift) * downsample_rate
    elif label_assignment_method == 'nearby_grid':
        feat[..., 1] = (feat[..., 1].tanh() * 1.5 + 0.5 + x_shift) * downsample_rate
        feat[..., 2] = (feat[..., 2].tanh() * 1.5 + 0.5 + y_shift) * downsample_rate
        #feat[..., 5::2] = (feat[..., 5::2].tanh() * 1.5 + 0.5 + x_shift) * downsample_rate
        #feat[..., 6::2] = (feat[..., 6::2].tanh() * 1.5 + 0.5 + y_shift) * downsample_rate
    else:
        raise NotImplementedError

    feat[...,3:5] = feat[...,3:5].clip(min=0)
    # Decode box width and height by multiplying predefined anchor boxes
    feat[..., 3:5] = feat[..., 3:5] * 4 * base_anchors

    return feat

def decode_kpts(feat, anchor_boxes, label_assignment_method, downsample_rate):
    num_kpts = (feat.shape[-1] - 5) // 2
    batch_size, _, grid_h, grid_w,_ = feat.size()
    device = feat.device
    num_anchor = anchor_boxes.shape[0]
    # When not training, we need to decode the output using position shifts and predefined anchor boxes.
    # The output in the last dimension will be [conf, dx, dy, width, height, class1, class2, ...]
    x_shift = torch.arange(grid_w).view(1, 1, 1, grid_w, 1).repeat(batch_size, num_anchor, grid_h, 1, num_kpts).to(device)
    y_shift = torch.arange(grid_h).view(1, 1, grid_h, 1, 1).repeat(batch_size, num_anchor, 1, grid_w, num_kpts).to(device)

    # Decode box center coords by adding position shifts
    if label_assignment_method == 'single_grid':
        feat[..., 5::2] = (feat[..., 5::2].sigmoid() + x_shift) * downsample_rate
        feat[..., 6::2] = (feat[..., 6::2].sigmoid() + y_shift) * downsample_rate
    elif label_assignment_method == 'all_grid':
        feat[..., 5::2] = (feat[..., 5::2].tanh() * grid_w + x_shift) * downsample_rate
        feat[..., 6::2] = (feat[..., 6::2].tanh() * grid_h + y_shift) * downsample_rate
    elif label_assignment_method == 'nearby_grid':
        feat[..., 5::2] = (feat[..., 5::2].tanh() * 1.5 + 0.5 + x_shift) * downsample_rate
        feat[..., 6::2] = (feat[..., 6::2].tanh() * 1.5 + 0.5 + y_shift) * downsample_rate
    else:
        raise NotImplementedError

    return feat
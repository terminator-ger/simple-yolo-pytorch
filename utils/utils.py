import os, random, torch, json
import numpy as np


def xyxy_to_xywh(xyxy):
    if not isinstance(xyxy, (torch.Tensor, np.ndarray)):
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    if xyxy.shape[-1] != 4:
        raise ValueError(f"Last dimension must be 4, got {xyxy.shape[-1]}")

    x1, y1, x2, y2 = xyxy[..., 0], xyxy[..., 1], xyxy[..., 2], xyxy[..., 3]
    width = x2 - x1
    height = y2 - y1
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    if isinstance(xyxy, torch.Tensor):
        return torch.stack([x_center, y_center, width, height], dim=-1)
    else:
        return np.stack([x_center, y_center, width, height], axis=-1)


def xywh_to_xyxy(xywh):
    if not isinstance(xywh, (torch.Tensor, np.ndarray)):
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    if xywh.shape[-1] != 4:
        raise ValueError(f"Last dimension must be 4, got {xywh.shape[-1]}")

    x_center, y_center, width, height = xywh[..., 0], xywh[..., 1], xywh[..., 2], xywh[..., 3]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    if isinstance(xywh, torch.Tensor):
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        return np.stack([x1, y1, x2, y2], axis=-1)


def box_iou(boxes1, boxes2, eps=1e-6):
    if isinstance(boxes1, torch.Tensor) and isinstance(boxes2, torch.Tensor):
        return box_iou_torch(boxes1, boxes2, eps)
    elif isinstance(boxes1, np.ndarray) and isinstance(boxes2, np.ndarray):
        return box_iou_numpy(boxes1, boxes2, eps)
    else:
        raise TypeError("Both inputs must be of the same type: either torch.Tensor or np.ndarray")


def box_iou_torch(boxes1, boxes2, eps=1e-6):
    inter_min = torch.max(boxes1[..., :2], boxes2[..., :2])
    inter_max = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    inter_wh = (inter_max - inter_min).clamp(min=0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - inter_area

    return inter_area / (union + eps)


def box_iou_numpy(boxes1, boxes2, eps=1e-6):
    inter_min = np.maximum(boxes1[..., :2], boxes2[..., :2])
    inter_max = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_wh = np.clip(inter_max - inter_min, a_min=0, a_max=None)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - inter_area

    return inter_area / (union + eps)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_writer(config, main_rank):
    if config.use_tb and main_rank:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.tb_log_dir)
    else:
        writer = None
    return writer


def get_logger(config, main_rank):
    if main_rank:
        import sys
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, format="[{time:YYYY-MM-DD HH:mm}] {message}", level="INFO")

        log_path = f'{config.save_dir}/{config.logger_name}.log'
        logger.add(log_path, format="[{time:YYYY-MM-DD HH:mm}] {message}", level="INFO")
    else:
        logger = None
    return logger


def save_config(config):
    config_dict = vars(config)
    with open(f'{config.save_dir}/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)


def log_config(config, logger):
    keys = ['task', 'dataset', 'num_class', 'model', 'backbone_type',
            'optimizer_type', 'lr_policy', 'total_epoch', 'train_bs', 'val_bs',  
            'train_num', 'val_num', 'gpu_num', 'num_workers', 'amp_training', 
            'DDP', 'use_ema', 'label_assignment_method']

    config_dict = vars(config)
    infos = f"\n\n\n{'#'*25} Config Informations {'#'*25}\n" 
    infos += '\n'.join('%s: %s' % (k, config_dict[k]) for k in keys)
    infos += f"\n{'#'*71}\n\n"
    logger.info(infos)
    
    
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int | None = None) -> torch.Tensor:
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)  # dist (lt, rb)
    return dist


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):  # use len(feats) to avoid TracerWarning from iterating over strides tensor
        stride = strides[i]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)



from yolo.core.loss.yolodetection_loss import YOLODetectionLoss as DetectionLoss
from yolo.core.loss.yolokeypoint_loss import YOLOKeypointLoss as KeypointLoss

from enum import Enum

class Loss(Enum):
    YOLODetectionLoss = DetectionLoss
    YOLOKeypointLoss = KeypointLoss
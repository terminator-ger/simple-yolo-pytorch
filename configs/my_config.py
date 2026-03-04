from .base_config import BaseConfig
import torch as th
import yolo.core.evaluate as E
from yolo.models.yolo import ModelOutputKeys
class MyConfig(BaseConfig):
    def __init__(self,):
        super().__init__()
        # Task
        self.task = 'predict' # train, val, predict, debug

        # VOC
        self.dataset = 'gosyn'
        #self.data_root = "/media/michael/Data/dev/pygo_synthetic/renders_new/"
        self.data_root = "/media/michael/SSD/data/renders/"
        self.num_class = 2

        # Model
        self.model = 'yolo'
        self.backbone_type = 'resnet18'
        self.channel_sparsity = 0.75

        # Training
        self.total_epoch = 70
        self.train_bs = 4
        self.optimizer_type = 'adam'
        self.base_lr = 1e-2
        self.freeze_backbone = True

        # Validating
        self.val_bs = 16

        # Debugging
        self.num_debug_batch = 1

        # Loss
        self.lambda_coord = 0.1
        self.lambda_obj = 1.0
        self.lambda_noobj = 0.5
        self.lambda_scales = [1.0, 1.0, 1.0]
        self.use_noobj_loss = False
        self.iou_loss_type = 'ciou'
        self.focal_loss_gamma = 3.5
        self.label_assignment_method = 'nearby_grid'
        self.match_iou_thres = 0.0625
        self.filter_by_max_iou = True

        # Testing (VOC)
        self.is_testing = False
        self.test_bs = 1
        self.test_data_folder = 'pred_images'
        #self.test_data_folder = 'data/images/val'
        self.test_conf_thrs = 0.4
        class_map = {'black':0, 'white':1,}
        self.class_map = {v:k for k,v in class_map.items()}

        # Training setting
        self.use_ema = True

        # Scheduler
        self.lr_policy = 'cos_warmup'
        self.warmup_epochs = 3

        self.anchor_boxes = [#[[19, 11], [20, 16], [28, 14]],
                             #[[29, 17], [29, 20], [28, 23]],
                             #[[41, 19], [29, 28], [41, 23]],
                             #[[43, 27], [42, 31], [42, 39]]
                            [[4,  6], [6, 10], [8, 14]],
                            [[12,16], [16,22], [20,28]],
                            [[24,32], [28,40], [32,48]]
                            ]
    
        self.load_ckpt_path = '/home/michael/data/dev/simple-yolo-pytorch/save/run_0102/last.pth'

        self.downsample_rate = [8, 16, 32]
        self.p2 = False


        self.img_size = [640,640]      # W, H
        self.evaluators = {
            ModelOutputKeys.Detections: {
                    E.EvalTypes.BBox: {"postfix": "Stones",},
                    E.EvalTypes.Classification: {"postfix": "Stones", "num_class": self.num_class,},
                    E.EvalTypes.ClassificationLogits: {"postfix": "Stones", "num_class": self.num_class},
                },
            ModelOutputKeys.Keypoints: {
                E.EvalTypes.BBox: {"postfix": "Board",},
                E.EvalTypes.Keypoint: {"postfix": 'Board',}
            }
        }

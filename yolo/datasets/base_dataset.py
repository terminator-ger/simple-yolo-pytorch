"""
Create by:  zh320
Date:       2024/06/15
"""

import os, random, torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as AT
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET

from yolo.utils.utils import xyxy_to_xywh


class BaseDataset(Dataset):
    def __init__(self, config, mode='train'):
        assert mode in ['train', 'val'], f'Unsupported dataset mode: {mode}.\n'

        self.data_folder = os.path.expanduser(config.data_root)
        #if not os.path.isdir(self.data_folder):
        #    raise RuntimeError(f'Path to {config.dataset} dataset: {self.data_folder} does not exist.\n')

        self.config = config
        self.mosaic_p = config.mosaic_p
        self.load_lbl_once = config.load_lbl_once
        self.mode = mode
        self.class_map = None
        self.transform = None
        self.images, self.labels = [], []

        self.normalize = AT.Compose([
                                    AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                    ToTensorV2()
                                    ])

    def _check_dataset(self):
        if self.class_map is None:
            if self.config.class_map is None:
                raise RuntimeError('You need to provide class map for dataset reading.\n')
            else:
                self.class_map = self.config.class_map
        assert len(self.class_map) == self.config.num_class

        assert len(self.images) > 0, 'No image found.\n'
        assert len(self.images) == len(self.labels), 'Number of images =/= number of labels.\n'
        self.image_indices = list(range(len(self.images)))

        if self.transform is None and self.mode == 'train':
            raise RuntimeError('You need to provide augmentations in train mode.\n')

        if self.load_lbl_once:
            self._cache_lbl()

    def _cache_lbl(self):
        print(f'Loading {self.mode} xml annotations into ram once...')
        self.box_ann, self.cls_ann = [], []
        rm_items = 0
        for lbl_path in self.labels:
            box, cls, rm_item = self.read_xml_ann(lbl_path, self.class_map, self.config.min_label_area)
            self.box_ann.append(box)
            self.cls_ann.append(cls)
            rm_items += rm_item
        print(f'Finished loading annotations. Remove {rm_items} unqualified (too small) objects in total.\n')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        mosaic = random.random() < self.mosaic_p and self.mode == 'train'

        # Load one image and its corresponding labels
        data = self.load_data(index, mosaic)
        width, height,_ = data['pixel_values'].shape
        image=data['pixel_values']
        bboxes=data['bboxes']
        class_labels=data['classes']
        keypoints=data['keypoints']
        bboxes_kp=data['bboxes_keypoints']
        classes_kp=data['classes_keypoints']
        # Perform augmentation
        if self.mode == 'train':
            augmented = self.transform(image=image,
                                       bboxes=bboxes,
                                       class_labels=class_labels,
                                       keypoints=keypoints,
                                       bboxes_kp=bboxes_kp,
                                       classes_kp=classes_kp)
            image = augmented['image']

            bboxes       = np.zeros((0, 4), dtype=np.float32)
            class_labels = np.zeros((0, 1), dtype=np.float32)
            keypoints    = np.zeros((0, 8), dtype=np.float32) 
            bboxes_kp    = np.zeros((0, 4), dtype=np.float32) 
            classes_kp   = np.zeros((0, 1), dtype=np.float32)

            # Update bbox and class after augmentation
            if len(augmented['bboxes'])>0:
                bboxes = np.asarray(augmented['bboxes'])
                class_labels = np.asarray(augmented['class_labels'])[:,None]
                bboxes = xyxy_to_xywh(bboxes)

            if len(augmented['keypoints']>0):
                keypoints = np.asarray(augmented['keypoints']) 
                bboxes_kp = np.asarray(augmented['bboxes_kp'])
                classes_kp = np.asarray(augmented['classes_kp'])

                # Transform bbox from xyxy format to xywh format because loss is calculated in xywh space
                keypoints = (keypoints / np.array([width, height])).reshape(1,-1)
                bboxes_kp = xyxy_to_xywh(bboxes_kp)
        
        elif self.mode == 'val':
            # We do NOT need to transform the bbox to xywh in val mode because torchvision.nms require xyxy format
            img_height, img_width, _ = image.shape
            bboxes[:, 0::2] *= img_width
            bboxes[:, 1::2] *= img_height
            bboxes_kp[:, 0::2] *= img_width
            bboxes_kp[:, 1::2] *= img_height
            keypoints = np.asanyarray(keypoints).reshape(-1,8)


        # Perform normalization
        image = self.normalize(image=image)['image']
        bboxes       = np.concatenate((np.zeros((bboxes.shape[0], 1)),       bboxes),      axis=1)
        class_labels = np.concatenate((np.zeros((class_labels.shape[0], 1)), class_labels),axis=1)
        keypoints    = np.concatenate((np.zeros((keypoints.shape[0], 1)),    keypoints),   axis=1)
        bboxes_kp    = np.concatenate((np.zeros((bboxes_kp.shape[0], 1)),    bboxes_kp),   axis=1)
        classes_kp   = np.concatenate((np.zeros((classes_kp.shape[0], 1)),   classes_kp),  axis=1)

        return {
            'pixel_values': image,
            'bboxes': torch.from_numpy(bboxes).to(torch.float32),
            'classes': torch.from_numpy(class_labels).to(torch.long),
            'keypoints': torch.from_numpy(keypoints).to(torch.float32),
            'bboxes_keypoints': torch.from_numpy(bboxes_kp).to(torch.float32),
            'classes_keypoints': torch.from_numpy(classes_kp).to(torch.long)
        }

    def _clip(self, augmented):
        for k in ['bboxes', 'bboxes_kp']:
            augmented[k] = augmented[k].clip(0,1)
        return augmented


    def load_data(self, index, mosaic):
        img_width_full, img_height_full = self.config.img_size

        if mosaic:
            # currently not working
            # Mosaic Training
            image_indices = self.image_indices.copy()
            image_indices.remove(index)
            random.shuffle(image_indices)
            indices = [index] + image_indices[:3]

            full_img = np.zeros([2*img_height_full, 2*img_width_full, 3])
            for k, idx in enumerate(indices):
                data = self.load_one_img_lbl(idx)
                image_k = data['pixel']
                bboxes_k = data['bboxes']
                classes_k = data['classes']
                kp_k = data['keypoints'] 
                kp_box = data['keypoints_bbox']

                img_height_k, img_width_k, _ = image_k.shape

                x_k, y_k = (-1) ** (k % 2 + 1), (-1) ** (k // 2 + 1)
                x_start = min(img_width_full, img_width_full + x_k * img_width_k)
                y_start = min(img_height_full, img_height_full + y_k * img_height_k)
                x_end = max(img_width_full, img_width_full + x_k * img_width_k)
                y_end = max(img_height_full, img_height_full + y_k * img_height_k)

                full_img[y_start:y_end, x_start:x_end] = image_k

                bboxes_k[:, 0::2] += x_start
                bboxes_k[:, 1::2] += y_start

                if k == 0:
                    bboxes = bboxes_k.copy()
                    classes = classes_k.copy()
                else:
                    bboxes = np.concatenate([bboxes, bboxes_k], axis=0)
                    classes = np.concatenate([classes, classes_k], axis=0)

            bboxes[:, 0::2] /= (2*img_width_full)
            bboxes[:, 1::2] /= (2*img_height_full)
            return {
                'pixel_values' : full_img.astype(np.uint8),
                'bboxes': bboxes.astype(np.float32),
                'classes': classes.astype(np.float32),
                'keypoints': kp_k if kp_k is not None else None,
                'bboxes_keypoints': kp_box if kp_k is not None else np.array([0,0,0,0]),
            }
        
        else:
        #    # Pad image for fixed size training
                return  self.load_one_img_lbl(index)
                

    def load_one_img_lbl(self, index):
        image = Image.open(self.images[index]).convert('RGB')

        img_width, img_height = image.size
        img_width_full, img_height_full = self.config.img_size

        if img_width >= img_height:
            resize_ratio = img_width_full / img_width
            new_img_width = img_width_full
            new_img_height = round(img_height * resize_ratio)
        else:
            resize_ratio = img_height_full / img_height
            new_img_height = img_height_full
            new_img_width = round(img_width * resize_ratio)

        image = np.array(image.resize((new_img_width, new_img_height)))

        if self.load_lbl_once:
            bboxes, classes = self.box_ann[index].copy(), self.cls_ann[index].copy()
        else:
            bboxes, classes, _ = self.read_xml_ann(self.labels[index], self.class_map)
        bboxes *= resize_ratio

        return image, bboxes, classes

    @staticmethod
    def collate_func(batch):
        '''Function to handle varying number of object within one image'''
        images = [x['pixel_values'] for x in batch]
        bboxes = [x['bboxes'] for x in batch]
        classes = [x['classes'] for x in batch]
        classes_kp = [x['classes_keypoints'] for x in batch]
        keypoints = [x['keypoints'] for x in batch]
        bboxes_keypoints = [x['bboxes_keypoints'] for x in batch]

        for idx, bbox in enumerate(bboxes):
            bbox[:, 0] = idx

        for idx, cls in enumerate(classes):
            cls[:, 0] = idx

        for idx, kp in enumerate(keypoints):
            kp[:,0] = idx
        
        for idx, kp_box in enumerate(bboxes_keypoints):
            kp_box[:,0] = idx

        for idx, cls_kp in enumerate(classes_kp):
            cls_kp[:,0] = idx


        return {
            'pixel_values': torch.stack(images, 0),
            'bboxes': torch.cat(bboxes, 0),
            'classes': torch.cat(classes, 0),
            'keypoints': torch.cat(keypoints, 0),
            'keypoints_bboxes': torch.cat(bboxes_keypoints, 0),
            'keypoints_classes': torch.cat(classes_kp, 0)
        }

    @classmethod
    def read_xml_ann(cls, file_path, class_map, min_label_area=0, normalize=False):
        tree = ET.ElementTree(file=file_path)
        root = tree.getroot()
        size = root.find('size')
        img_width = float(size.find('width').text)
        img_height = float(size.find('height').text)
        objs = root.findall('object')

        box, cid = [], []
        rm_item = 0
        for obj in objs:
            obj_name = obj.find('name').text
            class_id = float(class_map[obj_name])
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            if (xmax - xmin) * (ymax - ymin) <= min_label_area:
                rm_item += 1
                continue

            if normalize:
                xmin /= img_width
                ymin /= img_height
                xmax /= img_width
                ymax /= img_height

            box.append([xmin, ymin, xmax, ymax])
            cid.append([class_id])

        return np.array(box), np.array(cid), rm_item
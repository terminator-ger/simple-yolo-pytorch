
import os, sys
#SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
import torch as th
from torchvision.io import decode_image
import torchvision
import cv2
import numpy as np
import cmath
from math import atan2
import torch.nn.functional as F
from typing import Dict

#from .utils.utils import xyxy_to_xywh, xywh_to_xyxy
from yolo.datasets.base_dataset import BaseDataset
from yolo.datasets.dataset_registry import register_dataset
from random import random

import albumentations as AT

def erode(img, kernel_size=3, iterations=1):
    # Ensure 4D tensor
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.dim() == 3:
        img = img.unsqueeze(0)

    padding = kernel_size // 2
    kernel = th.ones((1, 1, kernel_size, kernel_size), dtype=img.dtype, device=img.device)   
    img = th.logical_not(img).float()  # Invert image for erosion
    for _ in range(iterations):
        img = F.max_pool2d(img, 
                           kernel_size=kernel_size,
                           stride=1,
                           padding=padding)
        #img = F.conv2d(img, kernel, padding=1)
    img = th.logical_not(img).float()  # Invert image for erosion

    return img.squeeze()

def convexHull(pts):    #Graham's scan.
    xleftmost, yleftmost = min(pts)
    by_theta = [(atan2(x-xleftmost, y-yleftmost), x, y) for x, y in pts]
    by_theta.sort()
    as_complex = [complex(x, y) for _, x, y in by_theta]
    chull = as_complex[:2]
    for pt in as_complex[2:]:
        #Perp product.
        while ((pt - chull[-1]).conjugate() * (chull[-1] - chull[-2])).imag < 0:
            chull.pop()
        chull.append(pt)
    return [(pt.real, pt.imag) for pt in chull]


def dft(xs):
    return [sum(x * cmath.exp(2j*np.pi*i*k/len(xs)) 
                for i, x in enumerate(xs))
            for k in range(len(xs))]

def interpolateSmoothly(xs, N):
    """For each point, add N points."""
    fs = dft(xs)
    half = (len(xs) + 1) // 2
    fs2 = fs[:half] + [0]*(len(fs)*N) + fs[half:]
    return [x.real / len(xs) for x in dft(fs2)[::-1]]


class GOSYNImageDataset(th.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, train=True, return_stone_bbox=True, return_board_corners=True):
        self.img_labels = pd.read_parquet(os.path.join(img_dir, annotations_file))
        if train:
            # reserve 80% for training
            self.img_labels = self.img_labels.iloc[:int(0.8*len(self.img_labels))]
        else:
            # reserve 20% for testing
            self.img_labels = self.img_labels.iloc[int(0.8*len(self.img_labels)):]
        self.train = train
            
        self.img_dir = img_dir
        #self.transform = self.image_transformer
        self.occlude_factor = 0.4
        self.return_stone_bbox = return_stone_bbox
        self.return_board_corners = return_board_corners
        self.random_background = True
        self.image_width = 640
        self.image_height = 640
        self.export =  False


    def pad_backgound(self, data:Dict) -> Dict:
        image = data['pixel_values']
        h,w = image.shape[0], image.shape[1]
        background_path = "/home/michael/data/unlabeled2017"
        background_images = os.listdir(background_path)
        bg_file = os.path.join(background_path, background_images[np.random.randint(len(background_images))])
        bg = cv2.resize(cv2.imread(bg_file), (w,h))
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        mask = image[...,3] != 0  # Alpha channel mask
        mask = mask.astype(np.uint8) * 255
        mask = cv2.erode(mask, kernel=np.ones((3,3), np.uint8), iterations=1).astype(bool)  
        mask = np.stack([mask,mask,mask],-1)
        image = np.where(mask, image[:,:,:3], bg)
        data['pixel_values'] = image
        return data


    def image_transformer(self, data: Dict):
        # scale down image size
        image = data['pixel_values']
        
        c,h,w = image.shape
        bbox = data['bbox']
        bbox = bbox / th.tensor([w,h,w,h])
        corners = data['corners']
        corners = corners / th.tensor([w,h,1])
        

        image = torchvision.transforms.Resize((self.image_width, self.image_height))(image)
        
        h,w = image.shape[1], image.shape[2]
        is_occluded = th.tensor([0], dtype=th.int32)

        background_path = "/home/michael/data/unlabeled2017"
        background_images = os.listdir(background_path)
        if self.random_background:
            bg_file = os.path.join(background_path, background_images[np.random.randint(len(background_images))])
        else:
            bg_file = os.path.join(background_path, background_images[0])
            
        
        image_fg = image[3].numpy()
        ret, thresh = cv2.threshold(image_fg, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask = image[3,:,:] != 0  # Alpha channel mask
        mask = erode(mask.to(th.float32)).bool()  # Further erosion using PyTorch
        mask = th.stack([mask,mask,mask],0)
        bg = cv2.resize(cv2.imread(bg_file), (h,w))
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        bg = th.from_numpy(bg).permute(2,0,1)
        image = th.where(mask, image[:3], bg)
        
        if False: #th.rand(1).item() < self.occlude_factor:
            is_occluded = th.tensor([1], dtype=th.int32)
            label = th.zeros_like(label, dtype=th.int32)
            # Apply occlusion
            # sample a random point within the image bounds
            pts = [(random() / 2 + 0.5) * cmath.exp(2j*np.pi*i/7) for i in range(7)]
            pts = convexHull([(pt.real, pt.imag ) for pt in pts])
            xs, ys = [interpolateSmoothly(zs, 30) for zs in zip(*pts)]
            scale_x = np.random.randint(w//16, w//4)
            center_x = np.random.randint(0, w)
            center_y = np.random.randint(0, h)
            
            while cv2.pointPolygonTest(contours[0], (center_x, center_y), False) < 0:
                center_x = np.random.randint(0, w)
                center_y = np.random.randint(0, h)
            
            xs = [int((x * scale_x) + center_x) for x in xs]
            ys = [int((y * scale_x) + center_y) for y in ys]
            
            cnt = np.stack((xs,ys))[:,:,None].transpose(1,2,0)
            color = np.random.randint(0,255,size=(3,)).tolist()
            img_np = image.permute(1,2,0).detach().numpy().copy()
            image = cv2.drawContours(img_np, (cnt,), 0, color=color, thickness=cv2.FILLED)
            image = th.tensor(image).permute(2,0,1)
            label = th.ones_like(label, dtype=th.int32) * 3
        
        image = image / 255.0  # Normalize to [0, 1]
        data['pixel_values'] = image
        data['bbox'] = bbox
        data['corners'] = corners
        return data

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]['filename'])
        #image = decode_image(img_path)
        positions = th.from_numpy(np.stack(self.img_labels.iloc[idx]['labels'])).to(th.int32)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
 
        bbox = np.zeros((0,4), dtype=np.float32)
        cls = np.zeros((1), dtype=np.float32)
        corners = None
       
        #resize
        old_w, old_h = image.shape[1], image.shape[0]
        image = cv2.resize(image, (self.image_width, self.image_height))        
        
        if self.return_stone_bbox:
            if self.img_labels.iloc[idx]['bbox'] is None or len(self.img_labels.iloc[idx]['bbox']) ==0:
                return self.__getitem__(idx+1)  # Skip this sample if bbox is missing or empty
            bbox = np.stack(self.img_labels.iloc[idx]['bbox']).astype(np.float32)
            bbox[:, 0::2] = bbox[:, 0::2] / old_w #* self.image_width
            bbox[:, 1::2] = bbox[:, 1::2] / old_h #* self.image_height
            # invert bbox y axis direction
            bbox[:,1], bbox[:,3] = bbox[:,3], bbox[:,1].copy()
            #bbox[:,1] = 1 - bbox[:,1]
            #bbox[:,3] = 1 - bbox[:,3]
            if any(bbox[:,3] <= bbox[:,1]):
                print(f"Warning: Invalid bbox with y_max > y_min in image {img_path}")
                bbox = bbox[bbox[:,3] > bbox[:,1]]

            idx_lt_0 = (bbox[:,3] - bbox[:,1] < 0)
            if any(idx_lt_0):
                print(f"Warning: Invalid bbox with y_max < y_min in image {img_path}")
                bbox = bbox[~idx_lt_0]

            
            cls = np.stack(self.img_labels.iloc[idx]['cls']).astype(np.float32)[:,None]
            cls -= 1

            # delete bbox and cls for bboxes outside the image after resizing
            #cond_a = bbox[:,0] < 0
            #cond_b = bbox[:,0] > self.image_width
            #cond_c = bbox[:,1] < 0
            #cond_d = bbox[:,1] > self.image_height
            #cond = np.logical_or(np.logical_or(np.logical_or(cond_a, cond_b), cond_c), cond_d)
            #bbox = bbox[~cond]
            #cls = cls[~cond]

        if self.return_board_corners:
            corners =np.stack(self.img_labels.iloc[idx]['corners']).astype(np.float32)
            corners[:, 0] = corners[:, 0] / old_w
            corners[:, 1] = corners[:, 1] / old_h 

            cond_a = corners[:,0] < 0 
            cond_b = corners[:,0] > self.image_width
            cond_c = corners[:,1] < 0 
            cond_d = corners[:,1] > self.image_height
            
            #_idx = np.argwhere(np.logical_or(np.logical_or(np.logical_or(cond_a, cond_b), cond_c), cond_d))
            #corners[...,2] = 1.0
            #corners[_idx,2] = 0.0


        data =  {"pixel_values": image, 
                "labels": positions, 
                "bbox": bbox, 
                "cls": cls, 
                "corners": corners
        }

        data = self.pad_backgound(data)
        
        if False:
            for rect in data['bbox']:
                x_min_norm,y_min_norm,x_max_norm,y_max_norm = rect
                x_min_norm *= 512
                y_min_norm *= 512
                x_max_norm *= 512
                y_max_norm *= 512
                print(f"{int(x_min_norm)}, {int(y_min_norm)}, {int(x_max_norm)}, {int(y_max_norm)}")

        if self.export:
            img_transformed = data['pixel_values']
            folder = "train" if self.train else "val"
            #for rect in data['bbox']:
            #    x_min_norm,y_min_norm,x_max_norm,y_max_norm = rect
            #    x_min_norm *= self.image_width
            #    y_min_norm *= self.image_height
            #    x_max_norm *= self.image_width
            #    y_max_norm *= self.image_height
            #    img_transformed = cv2.rectangle(img_transformed, (int(x_min_norm), int(y_min_norm)), (int(x_max_norm), int(y_max_norm)), (255,0,0), 1)
            
            cv2.imwrite(f'data_pose/images/{folder}/{idx}.png', cv2.cvtColor(img_transformed, cv2.COLOR_RGB2BGR))
            with open(f'data_pose/labels/{folder}/{idx}.txt', 'w') as f:
                for cls, bbox in zip(data['cls'],data['bbox']):
                    cls = int(cls.item())
                    x_center, y_center, width, height = bbox.tolist()
                    f.write(f"{cls} {x_center} {y_center} {width} {height}")
                    for _ in range(4):
                        f.write(f" {x_center} {y_center}")
                    f.write("\n")

            #with open(f'data_pose/labels/{folder}/{idx}.txt', 'w') as f:
                kp = data['corners']
                cls = 2
                x = kp[:,0].mean()
                y = kp[:,1].mean()
                w = kp[:,0].max() - kp[:,0].min()
                h = kp[:,1].max() - kp[:,1].min()
                #x,y,vis = kp.tolist()
                f.write(f"{cls} {x} {y} {w} {h}")
                for kp_x,kp_y,_ in kp.tolist():
                    f.write(f" {kp_x} {kp_y}")
                f.write("\n")
           
        return data
       
if __name__ == "__main__":
    dataset = GOSYNImageDataset(annotations_file="labels.parquet.gz", img_dir="/home/michael/data/dev/pygo_syn_dataset/renders/")
    wh = []
    for d in dataset:
        if d['bbox'].shape[0] == 0:
            continue
        w = (d['bbox'][:,2] - d['bbox'][:,0]).mean().item()
        h = (d['bbox'][:,3] - d['bbox'][:,1]).mean().item()
        wh.append([w,h])
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=12, random_state=0).fit(wh)
    a = np.asarray(kmeans.cluster_centers_)
    wh = np.array([640, 640])
    a = a * wh
    whsq = a[:,0]* a[:,1]
    idx = np.argsort(whsq)

    wh = a[idx].astype(int)
    print(wh)

    #dataset = GOSYNImageDataset(annotations_file="labels.parquet.gz", img_dir="/home/michael/data/dev/pygo_syn_dataset/renders/", train=False)
    #[_ for _ in dataset]



@register_dataset
class GOSYN(BaseDataset):
    """Wrapper to adapt GOSYNImageDataset to the BaseDataset interface.
    This class delegates data loading to `GOSYNImageDataset` but exposes
    the minimal methods `__len__`, `__getitem__` and `collate_func` so it
    can be used interchangeably with other datasets in the repo.
    """
    def __init__(self, config, mode='train'):
        super().__init__(config, mode)

        annotations_file = getattr(config, 'annotations_file', 'labels.parquet.gz')
        img_dir = getattr(config, 'data_root', '.')
        train = (mode == 'train')

        # instantiate the existing GOSYNImageDataset which handles transforms
        self._internal = GOSYNImageDataset(annotations_file=annotations_file,
                                          img_dir=img_dir,
                                          train=train,
                                          return_stone_bbox=True,
                                          return_board_corners=True)

        # keep config reference
        self.config = config
        self.image_indices = list(range(len(self._internal)))
        if mode == 'train':
            self.transform = AT.Compose([
                AT.RandomScale(scale_limit=config.randscale),
                AT.Perspective(scale=config.perspective_range, p=config.perspective_p),
                AT.Rotate(limit=config.rotate_limit, p=config.rotate_p),
                AT.PadIfNeeded(min_height=config.img_size[1], min_width=config.img_size[0], border_mode=0, value=(0,0,0)),
                AT.RandomCrop(height=config.img_size[1], width=config.img_size[0]),
                AT.ColorJitter(brightness=config.brightness, contrast=config.contrast, saturation=config.saturation, hue=config.hue),
                #AT.HorizontalFlip(p=config.h_flip),
                ], bbox_params=AT.BboxParams(format='albumentations', 
                                             label_fields=['class_labels'], 
                                             filter_invalid_bboxes=True,
                                             min_visibility=0.5,
                                             clip=True,
                                             min_area=0.01
                )
            )

    def __len__(self):
        return len(self._internal)

    def load_one_img_lbl(self, index):
        data = self._internal.__getitem__(index)
        image = data['pixel_values']
        bbox = data['bbox']
        cls = data['cls']
        return image, bbox, cls

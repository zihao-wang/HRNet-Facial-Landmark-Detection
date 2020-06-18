import os
import random
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image, ImageFile

from ..utils.transforms import (crop, fliplr_joints, generate_target,
                                transform_pixel)


class Dental(data.Dataset):
    """Dental dataset
    """
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.color = cfg.MODEL.COLOR  # TODO: [cfg] select the annotated color
        self.num_points = 0
        self.sigma = cfg.MODEL.SIGMA
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.input_size = cfg.MODEL.IMAGE_SIZE

        self.image_files = []
        self.annotation_files = []

        if self.is_train:
            data_dir = "data/train"
            annotation_dir = "data/golden_train_annotation"
            for c in os.listdir(annotation_dir):
                ann_file = os.path.join(annotation_dir, c, 'annotation.txt')
                img_file = os.path.join(data_dir, c, '1.tiff')
                if os.path.exists(img_file):
                    self.annotation_files.append(ann_file)
                    self.image_files.append(img_file)
        else:
            data_dir = "data/test/adult"
            annotation_dir = "data/test/sorted_adult_annotation"
            for c in os.listdir(annotation_dir):
                ann_file = os.path.join(annotation_dir, c, 'annotation.txt')
                img_file = os.path.join(data_dir, c, '1.tiff')
                if os.path.exists(img_file):
                    self.annotation_files.append(ann_file)
                    self.image_files.append(img_file)
            
            data_dir = "data/test/non-adult"
            annotation_dir = "data/test/sorted_non-adult_annotation"
            for c in os.listdir(annotation_dir):
                ann_file = os.path.join(annotation_dir, c, 'annotation.txt')
                img_file = os.path.join(data_dir, c, '1.tiff')
                if os.path.exists(img_file):
                    self.annotation_files.append(ann_file)
                    self.image_files.append(img_file)

    def __len__(self):
        return len(self.annotation_files)
    
    def load_img(self, idx):
        img_file = self.image_files[idx]
        # print(self.input_size, img_file)
        img = Image.open(img_file).convert('RGB').resize(self.input_size)
        mat = np.asarray(img)
        # print(mat.shape)
        mat = mat.reshape(3, mat.shape[0], mat.shape[1])
        # mat = np.repeat(mat, 3, axis=0)  # TODO: make sure why 3 replications
        return mat

    def load_pts(self, idx):
        anno_file = self.annotation_files[idx]
        point_clouds = defaultdict(list)
        with open(anno_file, 'rt') as f:
            key = None
            for l in f.readlines():
                s = l.strip()
                if not ',' in s and key != s:
                    key = s
                    continue
                if key:
                    point_clouds[key].append([float(x) for x in s.split(', ')])
        if self.color.lower() == 'all':
            pts = []
            for color in ['yellow', 'red', 'green', 'blue']:
                pts += point_clouds[color]
        else:
            pts = point_clouds[self.color.lower()]
        return np.asarray(pts)

    def get_num_points(self):
        if self.num_points > 0:
            return self.num_points
        else:
            self.num_points = len(self.load_pts(0))
            return self.num_points

    def __getitem__(self, idx):
        img = self.load_img(idx)
        pts = self.load_pts(idx)
        _, W, H = img.shape
        # TODO: data argumentation
        
        target = np.zeros((self.get_num_points(), W, H), dtype=np.float32)
        for i in range(self.get_num_points()):
            target[i] = generate_target(target[i], pts[i], self.sigma, label_type=self.label_type)
        
        img = img.astype(np.float32)
        img = img/255
        img = torch.tensor(img)
        target = torch.tensor(target)
        meta = {
            'index': idx,
            'center': torch.tensor([W//2, H//2]),
            'scale': 1,
            'pts': torch.tensor(pts),
            'tpts': torch.tensor(pts)
        }
        return img, target, meta
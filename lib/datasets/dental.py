import os
import random
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image, ImageFile

from ..utils.transforms import (crop, fliplr_joints, generate_target,
                                transform_pixel)

class Dental(data.Dataset):
    """Dental dataset
    """

    W_length = 173
    H_length = 234

    W_org_px = 1360
    H_org_px = 1840

    def __init__(self, cfg, split="train", seed=1, train_ratio=0.8, size=10000):
        self.cfg = cfg
        self.num_points = 0
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA / self.W_length * self.output_size[0]
        self.color = cfg.MODEL.COLOR
        self.size = size
        self.image_files = []
        self.annotation_files = []
        self.factor = [self.output_size[0] / self.W_org_px,
                       self.output_size[1] / self.H_org_px]
        print('scaling factor', self.factor)

        if split.lower() == "train" and split.lower() == "valid":
            np.random.seed(seed)
            data_dir = "data/final_train/训练集"
            annotation_dir = "data/final_train/训练集"
            selection = np.random.rand(len(os.listdir(annotation_dir)))
            for i, c in enumerate(os.listdir(annotation_dir)):
                ann_file = os.path.join(annotation_dir, c, 'annotation.txt')
                img_file = os.path.join(data_dir, c, '1.tiff')
                if ((split.lower() == "train" and selection[i] < train_ratio)
                     or
                   (split.lower() == "valid" and selection[i] > train_ratio)):
                    if os.path.exists(ann_file) and os.path.exists(img_file):
                        self.annotation_files.append(ann_file)
                        self.image_files.append(img_file)
        else:
            data_dir = "data/final_test/测试集/未成年"
            annotation_dir = "data/final_test/测试集/未成年"
            for c in os.listdir(annotation_dir):
                ann_file = os.path.join(annotation_dir, c, 'annotation.txt')
                img_file = os.path.join(data_dir, c, '1.tiff')
                if os.path.exists(ann_file) and os.path.exists(img_file):
                    self.annotation_files.append(ann_file)
                    self.image_files.append(img_file)
            print('Testing data:', len(self.image_files))

    def __len__(self):
        return min(len(self.annotation_files), self.size)

    def load_img(self, idx, rotation=0):
        img_file = self.image_files[idx]
        # print(self.input_size, img_file)
        img = Image.open(img_file).convert('RGB').rotate(rotation).resize(self.input_size)
        mat = np.asarray(img)
        # mat = mat.reshape(3, mat.shape[0], mat.shape[1])
        mat = np.transpose(mat, (2, 1, 0))  # input channel, width, height
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
                    h, w = [float(x) for x in s.split(', ')]
                    point_clouds[key].append([w * self.factor[0], h * self.factor[1]])
        # for color in ['yellow', 'red', 'green', 'blue']:
        #         point_clouds[color] = point_clouds[color][::-1]
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
        rotation = np.random.choice(self.cfg.DATASET.ROT_FACTOR)

        img = self.load_img(idx, rotation)
        _, W, H = img.shape

        pts = self.load_pts(idx)
        # print("get item", idx)
        rot_mat = np.zeros((2, 2))
        rot_rad = rotation * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        pts = pts.dot(rot_mat)

        target = np.zeros((self.get_num_points(), self.output_size[0], self.output_size[1]), dtype=np.float32)
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


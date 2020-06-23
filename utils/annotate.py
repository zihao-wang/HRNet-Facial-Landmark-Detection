# -*- coding: utf-8 -*-
import os

from PIL import Image
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm


feat_map = {
    'yellow': np.asarray([255, 255, 0]),
    'red': np.asarray([255, 0, 0]),
    'green': np.asarray([0, 153, 68]),
    'blue': np.asarray([0, 0, 255])
}


img_dir = "data/test/non-adult"
ann_dir = "data/test/non-adult_annotation"


def pixel_labeler(pix_array, tol=1):
    for k, v in feat_map.items():
        if np.linalg.norm(v-pix_array, ord=1) < tol:
            return k
    else:
        return None

    
def flood_flow(mat, bar, label_func=pixel_labeler):
    W, L, _ = mat.shape
    labels = defaultdict(list)
    visited = np.zeros((W, L))
    color_counter = Counter()

    def inspect(_i, _j, _label):
        visited[_i, _j] = 1
        dxs = [-1, 1, 0, 0]
        dys = [0, 0, -1, 1]
        collected_x = [_i]
        collected_y = [_j]
        for dx, dy in zip(dxs, dys):
            ni, nj = _i + dx, _j + dy
            if visited[ni, nj] == 1:
                continue
            cursor_label = label_func(mat[ni, nj, :])
            visited[ni, nj] = 1
            if cursor_label == _label:
                cx, cy = inspect(ni, nj, _label)
                collected_x.extend(cx)
                collected_y.extend(cy)
        return collected_x, collected_y
    with tqdm(range(W), desc=bar) as t:
        for i in t:
            for j in range(L):
                if visited[i, j] == 1: continue
                label = label_func(mat[i, j, :])
                if label:
                    cx, cy = inspect(i, j, label)
                    labels[label].append([int(np.mean(cx)), int(np.mean(cy))])
    #                 print(i, j, np.mean(cx), len(cx), np.mean(cy), len(cy))
                    color_counter[label] += 1
            t.set_postfix(color_counter)
    return labels

# counts = 0
for fold in os.listdir(img_dir):
    img_file = os.path.join(img_dir, fold, "2.tif")
    txt_file = os.path.join(ann_dir, fold, "annotation.txt")
    img = Image.open(img_file)
    mat = np.asarray(img)
    print(mat.shape)
    labels = flood_flow(mat, str(img_file))
    os.makedirs(os.path.join(ann_dir, fold), exist_ok=True)
    # with open(txt_file, 'wt') as f:
    #     for color in feat_map:
    #         f.write(color + "\n")
    #         for x, y in labels[color]:
    #             f.write("{}, {}\n".format(x, y))
    # counts += 1
    # if counts >= 20:
        # break
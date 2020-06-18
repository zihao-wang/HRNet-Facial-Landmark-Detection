import os

from collections import defaultdict

from PIL import Image
import numpy as np


class DataReader:
    def __init__(self, folder='train'):
        self.folder = folder
        self.afolder = folder + "_annotation"
        self.case_folder_list = os.listdir(self.folder)
        
    def get_img(self, f):
        img_file = os.path.join(self.folder, f, '1.tiff')
        img = Image.open(img_file).convert('L')
        mat = np.asarray(img)
        return mat
    
    def get_annotation(self, f):
        annotation_file = os.path.join(self.afolder, f, 'annotation.txt')
        point_cloud = self.parse_annotation(annotation_file)
        return point_cloud

    @staticmethod
    def parse_annotation(f):
        point_clouds = defaultdict(list)
        with open(f, 'rt') as f:
            key = None
            for l in f.readlines():
                s = l.strip()
                if not ',' in s and key != s:
                    key = s
                    continue
                if key:
                    point_clouds[key].append([float(x) for x in s.split(', ')])
        return point_clouds


def test_channels(mat):
    clist = [mat[:,:,i] for i in range(mat.shape[-1])]
    c0 = clist[0]
    for c in clist:
        print(np.sum(np.abs(c)), np.sum(np.abs(c0 - c)))


if __name__ == "__main__":
    dr = DataReader()
    pc = dr.get_annotation('训练 (1)')
    for k, v in pc.items():
        print(k, v)
        print(k, np.asarray(v))
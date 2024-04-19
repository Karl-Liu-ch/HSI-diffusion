import sys
sys.path.append('./')
import os
import numpy as np
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex

from torch.utils.data import Dataset
import numpy as np
import random
import cv2
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import torch
import torch.nn as nn
# from options import opt
import scipy.io
import re
from dataset.data_augmentation import *
# from utils import *
from omegaconf import OmegaConf
config_path = "configs/hsi_vqgan.yaml"
cfg = OmegaConf.load(config_path)
trainconfig = cfg.data.params.train.params
validconfig = cfg.data.params.validation.params

def Normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def clone_normalize(x):
    x_ = np.copy(x)
    x_ = (x_ - x_.min()) / (x_.max() - x_.min())
    return x_

class GetDataset(Dataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=True, datanames = ['ARAD/']):
        self.datanames = datanames
        self.crop_size = crop_size
        self.arg = arg
        self.rgb = []
        self.hyper = []

    def augmentation(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img
    
    def __getitem__(self, idx):
        ex = {}
        bgr = self.rgb[idx]
        rgb = (np.copy(bgr) / 255.0).astype(np.float32)
        # rgb = Normalize(rgb)
        rgb = np.transpose(rgb, [2, 0, 1])
        # ycrcb = Normalize(cv2.cvtColor(bgr, cv2.COLOR_RGB2YCrCb))
        ycrcb = (cv2.cvtColor(bgr, cv2.COLOR_RGB2YCrCb) / 240.0).astype(np.float32)
        # bgr = Normalize(bgr)
        bgr = (bgr / 255.0).astype(np.float32)
        bgr = np.concatenate([bgr, ycrcb], axis=2)
        hyper = self.hyper[idx]
        # hyper = Normalize(hyper)
        bgr = np.transpose(bgr, [2, 0, 1])
        hyper = np.transpose(hyper, [2, 0, 1])
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.augmentation(bgr, rotTimes, vFlip, hFlip)
            rgb = self.augmentation(rgb, rotTimes, vFlip, hFlip)
            hyper = self.augmentation(hyper, rotTimes, vFlip, hFlip)
        ex["label"] = np.ascontiguousarray(hyper)
        ex["cond"] = np.ascontiguousarray(rgb)
        ex["ycrcb"] = np.ascontiguousarray(bgr)
        return ex

    def __len__(self):
        return self.length

class TrainDataset(GetDataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=True, datanames = ['BGU/','ARAD/'], random_split = True):
        super().__init__(data_root, crop_size, valid_ratio, test_ratio, arg=arg, datanames = datanames)
        self.trainset = []
        for name in self.datanames:
            if random_split:
                self.trainset.append(random_split_train(data_root+ 'clean/' + name,valid_ratio=valid_ratio, test_ratio=test_ratio, imsize=self.crop_size))
            else:
                self.trainset.append(split_train(data_root+ 'clean/' + name,valid_ratio=valid_ratio, test_ratio=test_ratio, imsize=self.crop_size))
        for trainset in self.trainset:
            self.rgb.extend(trainset[1])
            self.hyper.extend(trainset[0])
        self.length = len(self.hyper)

class TestDataset(GetDataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=False, datanames = ['BGU/','ARAD/'], cave = False, random_split = True):
        super().__init__(data_root, crop_size, valid_ratio, test_ratio, arg=arg, datanames = datanames)
        self.testset = []
        for name in self.datanames:
            if random_split:
                self.testset.append(random_split_test(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, imsize=self.crop_size))
            else:
                self.testset.append(split_test(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, imsize=self.crop_size))
        if cave:
            if random_split:
                self.testset.append(random_split_test(data_root+ 'clean/' + name, valid_ratio= 0, test_ratio=1, imsize=self.crop_size))
            else:
                self.testset.append(split_test(data_root+ 'clean/' + 'CAVE/', valid_ratio= 0, test_ratio=1, imsize=self.crop_size))
        for testset in self.testset:
            self.rgb.extend(testset[1])
            self.hyper.extend(testset[0])
        self.length = len(self.hyper)

class TestFullDataset(GetDataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=False, datanames = ['BGU/','ARAD/'], cave = True, random_split = True):
        super().__init__(data_root, crop_size, valid_ratio, test_ratio, arg=arg, datanames = datanames)
        self.testset = []
        for name in self.datanames:
            if random_split:
                self.testset.append(random_split_full_test(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio))
            else:
                self.testset.append(split_full_test(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio))
        if cave:
            if random_split:
                self.testset.append(random_split_full_test(data_root+ 'clean/' + 'CAVE/', valid_ratio= 0, test_ratio=1))
            else:
                self.testset.append(split_full_test(data_root+ 'clean/' + 'CAVE/', valid_ratio= 0, test_ratio=1))
        for testset in self.testset:
            self.rgb.extend(testset[1])
            self.hyper.extend(testset[0])
        self.length = len(self.hyper)

class ValidDataset(GetDataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=True, datanames = ['BGU/','ARAD/'], random_split = True):
        super().__init__(data_root, crop_size, valid_ratio, test_ratio, arg=arg, datanames = datanames)
        self.validset = []
        for name in self.datanames:
            if random_split:
                self.validset.append(random_split_valid(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, imsize=self.crop_size))
            else:
                self.validset.append(split_valid(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, imsize=self.crop_size))
        for validset in self.validset:
            self.rgb.extend(validset[1])
            self.hyper.extend(validset[0])
        self.length = len(self.hyper)

class TestDatasetclean(GetDataset):
    def __init__(self, data_root, arg=False, datanames = ['ARAD/']):
        self.datanames = datanames
        self.arg = arg
        self.rgb = []
        self.hyper = []
        self.testset = []
        tail = re.compile('.dat')
        f = os.listdir(data_root)
        a = [(tail.search(file) != None) for file in f]
        a.remove(False)
        self.length = len(a)
        for i in range(self.length):
            name = str(i).zfill(3) + '.mat'
            mat = scipy.io.loadmat(data_root+name)
            self.rgb.extend(mat['rgb'])
            self.hyper.extend(mat['cube'])

class ValidFullDataset(GetDataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=False, datanames = ['BGU/','ARAD/'], random_split = True):
        super().__init__(data_root, crop_size, valid_ratio, test_ratio, arg=arg, datanames = datanames)
        self.testset = []
        for name in self.datanames:
            if random_split:
                self.testset.append(random_split_full_valid(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio))
            else:
                self.testset.append(split_full_valid(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio))
        for testset in self.testset:
            self.rgb.extend(testset[1])
            self.hyper.extend(testset[0])
        self.length = len(self.hyper)

if __name__ == '__main__':
    testset = TestFullDataset(**trainconfig)
    print(testset.__len__())
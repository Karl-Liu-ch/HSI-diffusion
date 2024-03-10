import sys
sys.path.append('./')
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
from dataset.data_augmentation import split_train, split_valid, split_test, split_full_test
from utils import *

root = '/work3/s212645/Spectral_Reconstruction/clean/'
datanames = ['CAVE/']
datanames = ['BGU/','ARAD/']
# datanames = ['ARAD/', 'BGU/']
# datanames = ['CAVE/', 'ARAD/', 'BGU/']

class GetDataset(Dataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=True, datanames = ['BGU/','ARAD/']):
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
        bgr = self.rgb[idx]
        rgb = np.copy(bgr)
        # rgb = Normalize(rgb)
        rgb = np.transpose(rgb, [2, 0, 1])
        # bgr = Image256(bgr)
        ycrcb = Normalize(cv2.cvtColor(bgr, cv2.COLOR_RGB2YCrCb))
        # ycrcb = bgr
        # bgr = Normalize(ycrcb)
        bgr = np.concatenate([bgr, ycrcb], axis=2)
        hyper = self.hyper[idx]
        bgr = np.transpose(bgr, [2, 0, 1])
        hyper = np.transpose(hyper, [2, 0, 1])
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.augmentation(bgr, rotTimes, vFlip, hFlip)
            hyper = self.augmentation(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper), np.ascontiguousarray(rgb)

    def __len__(self):
        return self.length

class TrainDataset(GetDataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=True, datanames = ['BGU/','ARAD/']):
        super().__init__(data_root, crop_size, valid_ratio, test_ratio, arg=arg, datanames = datanames)
        self.trainset = []
        for name in self.datanames:
            self.trainset.append(split_train(data_root+ 'clean/' + name,valid_ratio=valid_ratio, test_ratio=test_ratio, imsize=self.crop_size))
        for trainset in self.trainset:
            self.rgb.extend(trainset[1])
            self.hyper.extend(trainset[0])
        self.length = len(self.hyper)

class TestDataset(GetDataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=False, datanames = ['BGU/','ARAD/']):
        super().__init__(data_root, crop_size, valid_ratio, test_ratio, arg=arg, datanames = datanames)
        self.testset = []
        for name in self.datanames:
            self.testset.append(split_test(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, imsize=self.crop_size))
        for testset in self.testset:
            self.rgb.extend(testset[1])
            self.hyper.extend(testset[0])
        self.length = len(self.hyper)

class TestFullDataset(GetDataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=False, datanames = ['BGU/','ARAD/']):
        super().__init__(data_root, crop_size, valid_ratio, test_ratio, arg=arg, datanames = datanames)
        self.testset = []
        for name in self.datanames:
            self.testset.append(split_full_test(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio))
        for testset in self.testset:
            self.rgb.extend(testset[1])
            self.hyper.extend(testset[0])
        self.length = len(self.hyper)

class ValidDataset(GetDataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=True, datanames = ['BGU/','ARAD/']):
        super().__init__(data_root, crop_size, valid_ratio, test_ratio, arg=arg, datanames = datanames)
        self.validset = []
        for name in self.datanames:
            self.validset.append(split_valid(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, imsize=self.crop_size))
        for validset in self.validset:
            self.rgb.extend(validset[1])
            self.hyper.extend(validset[0])
        self.length = len(self.hyper)

class TestDatasetclean(GetDataset):
    def __init__(self, data_root, arg=False, datanames = ['BGU/','ARAD/']):
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

if __name__ == '__main__':
    root = '/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/'
    testset = TestDatasetclean(root)
    print(testset.__len__())
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

class Train_Dataset(Dataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, datanames = ['ARAD/'], arg=True, stride=8):
        sample = scipy.io.loadmat(f'{data_root}clean/{datanames[0]}001.mat')['rgb']
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.ycrcbs = []
        self.arg = arg
        h,w,_ = sample.shape  # img shape
        if crop_size >= h or crop_size >= w:
            self.full = True
        else:
            self.full = False
        if not self.full:
            self.h,self.w = h,w  # img shape
            self.h_last = (h - crop_size) % stride > 0
            self.w_last = (w - crop_size) % stride > 0
            self.stride = stride
            self.patch_per_line = (w-crop_size)//stride+1
            self.patch_per_colum = (h-crop_size)//stride+1
            if self.h_last:
                self.patch_per_colum += 1
            if self.w_last:
                self.patch_per_line += 1
            self.patch_per_img = self.patch_per_line*self.patch_per_colum
        else:
            self.h,self.w = h,w  # img shape
            self.stride = 1
            self.patch_per_line = 1
            self.patch_per_colum = 1
            self.patch_per_img = self.patch_per_line*self.patch_per_colum
        self.datanames = datanames
        trainsets = []
        self.split_dataset(trainsets, data_root, valid_ratio, test_ratio)
        for trainset in trainsets:
            self.hypers.extend(trainset[0])
            self.bgrs.extend(trainset[1])
            self.ycrcbs.extend(trainset[2])
        del trainsets
        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def split_dataset(self, trainsets, data_root, valid_ratio, test_ratio):
        for name in self.datanames:
            trainsets.append(random_split_full(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, mode='train'))

    def arguement(self, img, rotTimes, vFlip, hFlip):
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
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        ycrcb = self.ycrcbs[img_idx]
        hyper = hyper.astype(np.float32)
        ycrcb = (ycrcb / 240.0).astype(np.float32)
        bgr = (bgr / 255.0).astype(np.float32)
        bgr = np.transpose(bgr, [2, 0, 1])
        hyper = np.transpose(hyper, [2, 0, 1])
        ycrcb = np.transpose(ycrcb, [2, 0, 1])
        ycrcb = np.concatenate([bgr, ycrcb], axis=0).astype(np.float32)

        if not self.full:
            if h_idx*stride+crop_size > self.h:
                start_h = self.h - crop_size
                end_h = self.h
            else:
                start_h = h_idx*stride
                end_h = h_idx*stride+crop_size
            if w_idx*stride+crop_size > self.w:
                start_w = self.w - crop_size
                end_w = self.w
            else:
                start_w = w_idx*stride
                end_w = w_idx*stride+crop_size
            bgr = bgr[:,start_h:end_h, start_w:end_w]
            hyper = hyper[:,start_h:end_h, start_w:end_w]
            ycrcb = ycrcb[:,start_h:end_h, start_w:end_w]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
            ycrcb = self.arguement(ycrcb, rotTimes, vFlip, hFlip)
        ex["label"] = np.ascontiguousarray(hyper)
        ex["cond"] = np.ascontiguousarray(bgr)
        ex["ycrcb"] = np.ascontiguousarray(ycrcb)
        return ex

    def __len__(self):
        return self.patch_per_img*self.img_num
    
class Valid_Dataset(Train_Dataset):
    def split_dataset(self, trainsets, data_root, valid_ratio, test_ratio):
        for name in self.datanames:
            trainsets.append(random_split_full(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, mode='val'))

class Valid_Full_Dataset(Train_Dataset):
    def split_dataset(self, trainsets, data_root, valid_ratio, test_ratio):
        for name in self.datanames:
            trainsets.append(random_split_full(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, mode='val'))

class Test_Dataset(Train_Dataset):
    def split_dataset(self, trainsets, data_root, valid_ratio, test_ratio):
        for name in self.datanames:
            trainsets.append(random_split_full(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, mode='test'))

class GetDataset(Dataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=True, datanames = ['ARAD/']):
        self.datanames = datanames
        self.crop_size = crop_size
        self.arg = arg
        self.hypers = []
        self.bgrs = []
        self.ycrcbs = []

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
        bgr = self.bgrs[idx]
        hyper = self.hypers[idx]
        ycrcb = self.ycrcbs[idx]
        hyper = hyper.astype(np.float32)
        ycrcb = ycrcb.astype(np.float32)
        bgr = bgr.astype(np.float32)
        bgr = np.transpose(bgr, [2, 0, 1])
        hyper = np.transpose(hyper, [2, 0, 1])
        ycrcb = np.transpose(ycrcb, [2, 0, 1])
        ycrcb = np.concatenate([bgr, ycrcb], axis=0).astype(np.float32)

        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.augmentation(bgr, rotTimes, vFlip, hFlip)
            ycrcb = self.augmentation(ycrcb, rotTimes, vFlip, hFlip)
            hyper = self.augmentation(hyper, rotTimes, vFlip, hFlip)
        ex["label"] = np.ascontiguousarray(hyper)
        ex["cond"] = np.ascontiguousarray(bgr)
        ex["ycrcb"] = np.ascontiguousarray(ycrcb)
        return ex

    def __len__(self):
        return self.length

class TrainDataset(GetDataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=True, datanames = ['BGU/','ARAD/'], random_split = True, stride=128):
        super().__init__(data_root, crop_size, valid_ratio, test_ratio, arg=arg, datanames = datanames)
        trainsets = []
        for name in self.datanames:
            if random_split:
                trainsets.append(random_split_dataset(data_root+ 'clean/' + name,valid_ratio=valid_ratio, test_ratio=test_ratio, imsize=self.crop_size, mode='train', stride=stride))
            else:
                trainsets.append(split_dataset(data_root+ 'clean/' + name,valid_ratio=valid_ratio, test_ratio=test_ratio, imsize=self.crop_size, mode='train', stride=stride))
        for trainset in trainsets:
            self.hypers.extend(trainset[0])
            self.bgrs.extend(trainset[1])
            self.ycrcbs.extend(trainset[2])
        self.length = len(self.hypers)

class TestDataset(GetDataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=False, datanames = ['BGU/','ARAD/'], cave = False, random_split = True, stride=128):
        super().__init__(data_root, crop_size, valid_ratio, test_ratio, arg=arg, datanames = datanames)
        self.testset = []
        for name in self.datanames:
            if random_split:
                self.testset.append(random_split_dataset(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, imsize=self.crop_size, mode='test', stride=stride))
            else:
                self.testset.append(split_dataset(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, imsize=self.crop_size, mode='test', stride=stride))
        if cave:
            if random_split:
                self.testset.append(random_split_dataset(data_root+ 'clean/' + name, valid_ratio= 0, test_ratio=1, imsize=self.crop_size, mode='test', stride=stride))
            else:
                self.testset.append(split_dataset(data_root+ 'clean/' + 'CAVE/', valid_ratio= 0, test_ratio=1, imsize=self.crop_size, mode='test', stride=stride))
        for testset in self.testset:
            self.hypers.extend(testset[0])
            self.bgrs.extend(testset[1])
            self.ycrcbs.extend(testset[2])
        self.length = len(self.hyper)

class TestFullDataset(GetDataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=False, datanames = ['BGU/','ARAD/'], cave = True, random_split = True):
        super().__init__(data_root, crop_size, valid_ratio, test_ratio, arg=arg, datanames = datanames)
        self.testset = []
        for name in self.datanames:
            if random_split:
                self.testset.append(random_split_full(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, mode='test'))
            else:
                self.testset.append(split_full(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, mode='test'))
        if cave:
            if random_split:
                self.testset.append(random_split_full(data_root+ 'clean/' + 'CAVE/', valid_ratio= 0, test_ratio=1, mode='test'))
            else:
                self.testset.append(split_full(data_root+ 'clean/' + 'CAVE/', valid_ratio= 0, test_ratio=1, mode='test'))
        for testset in self.testset:
            self.hypers.extend(testset[0])
            self.bgrs.extend(testset[1])
            self.ycrcbs.extend(testset[2])
        self.length = len(self.hyper)

class ValidFullDataset(GetDataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=False, datanames = ['BGU/','ARAD/'], random_split = True):
        super().__init__(data_root, crop_size, valid_ratio, test_ratio, arg=arg, datanames = datanames)
        self.testset = []
        for name in self.datanames:
            if random_split:
                self.testset.append(random_split_dataset(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, imsize=crop_size, mode='val'))
            else:
                self.testset.append(split_dataset(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, imsize=crop_size, mode='val'))
        for testset in self.testset:
            self.hypers.extend(testset[0])
            self.bgrs.extend(testset[1])
            self.ycrcbs.extend(testset[2])
        self.length = len(self.hyper)

class ValidDataset(GetDataset):
    def __init__(self, data_root, crop_size, valid_ratio, test_ratio, arg=True, datanames = ['BGU/','ARAD/'], random_split = True, stride=128):
        super().__init__(data_root, crop_size, valid_ratio, test_ratio, arg=arg, datanames = datanames)
        self.validset = []
        for name in self.datanames:
            if random_split:
                self.validset.append(random_split_dataset(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, imsize=self.crop_size, mode='val', stride=stride))
            else:
                self.validset.append(split_dataset(data_root+ 'clean/' + name, valid_ratio= valid_ratio, test_ratio=test_ratio, imsize=self.crop_size, mode='val', stride=stride))
        for validset in self.validset:
            self.hypers.extend(validset[0])
            self.bgrs.extend(validset[1])
            self.ycrcbs.extend(validset[2])
        self.length = len(self.hyper)

if __name__ == '__main__':
    testset = TrainDataset(root, 256, 0.001, 0.998, arg=False, datanames=['ARAD/'], stride=128)
    print(testset.__len__())
    for test in testset:
        print(test['label'].max())
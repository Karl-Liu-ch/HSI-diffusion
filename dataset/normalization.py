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
import scipy.io
import re
import matplotlib.pyplot as plt
import os
import h5py

# root = '/work3/s212645/Spectral_Reconstruction/'
# name = 'ICVL/'
# path = root + name
# path2 = '/work3/s212645/Spectral_Reconstruction/ICVL_normalization/'

# if not os.path.exists(path2):
#     os.makedirs(path2)

# filelist = os.listdir(path)
# filelist.sort()
# reg = re.compile(r'.*.mat')
# for file in filelist:
#     if re.findall(reg, file):
#         mat = scipy.io.loadmat(path+file)
#         hyper = np.float32(np.array(mat['cube']))
#         rgb = np.float32(np.array(mat['rgb']))
#         hyper = hyper / 4095.0
#         mat['cube'] = hyper
#         scipy.io.savemat(path2+file, mat)
#         print(file, ' saved')

root = '/work3/s212645/Spectral_Reconstruction/BGU_HS/'
RGBpath = '/work3/s212645/Spectral_Reconstruction/BGU_HS/NTIRE2018_Train1_RealWorld/'
SPpath = '/work3/s212645/Spectral_Reconstruction/BGU_HS/NTIRE2018_Train1_Spectral/'
path2 = '/work3/s212645/Spectral_Reconstruction/BGU_HS/BGU_HS/'

filelist = os.listdir(SPpath)
filelist.sort()
reg = re.compile(r'.*.mat')
i = 0
for file in filelist:
    if re.findall(reg, file):
        i += 1
        matnew = {}
        with h5py.File(SPpath+file, 'r') as mat:
            hyper = np.float32(np.array(mat['rad']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = (hyper-hyper.min())/(hyper.max()-hyper.min())
        matnew['cube'] = hyper
        rgbname = file.split('.')[0]+'_camera.jpg'
        rgbpath = RGBpath + rgbname
        rgb = cv2.imread(rgbpath)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = np.float32(rgb)
        rgb = (rgb-rgb.min())/(rgb.max()-rgb.min())
        matnew['rgb'] = rgb
        newname = str(i).zfill(3)
        scipy.io.savemat(path2+newname+'.mat', matnew)
        print(file, ' saved')
        break
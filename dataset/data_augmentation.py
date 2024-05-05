import sys
sys.path.append('./')
from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import torch
import torch.nn as nn
# from options import opt
import scipy.io
import re
import matplotlib.pyplot as plt
from torch import randperm
from tqdm import tqdm

root = '/work3/s212645/Spectral_Reconstruction/'
datanames = ['ICVL/', 'ARAD/', 'CAVE/']

def Rename(path):
    filelist = os.listdir(path)
    reg = re.compile(r'.*.mat')
    for file in filelist:
        if re.findall(reg, file):
            oldname = path + file
            number = file.split('.mat')[0]
            newnumber = number.zfill(3)
            newname = path + newnumber + '.mat'
            os.rename(oldname, newname)

def gen_test_list(matlist, startidx, size):
    testlist = []
    for i in range(size):
        testlist.append(matlist[i + startidx])
    return testlist

def cross_validation_lists(path, test_size):
    crossvaltrainlists = []
    crossvaltestlists = []
    filelist = os.listdir(path)
    filelist.sort()
    assert len(filelist) > test_size
    reg = re.compile(r'.*.mat')
    matlist = []
    for file in filelist:
        if re.match(reg, file):
            matlist.append(file)
    matlist.sort()
    all_length = len(matlist)
    assert all_length % test_size == 0
    for i in range(all_length // test_size):
        trainlist = []
        testlist = []
        testlist = gen_test_list(matlist, i * test_size, test_size)
        trainlist = matlist.copy()
        for mat in testlist:
            trainlist.remove(mat)
        crossvaltestlists.append(testlist)
        crossvaltrainlists.append(trainlist)
    return crossvaltrainlists, crossvaltestlists

def cross_validation_dataset(path, test_size, imsize):
    train_sets = []
    test_sets = []
    crossvaltrainlists, crossvaltestlists = cross_validation_lists(path, test_size)
    num_crossval = len(crossvaltrainlists)
    for i in range(num_crossval):
        train_sets.append(get_dataset(crossvaltrainlists[i], imsize))
        test_sets.append(get_dataset(crossvaltestlists[i], imsize))
    return train_sets, test_sets

def get_dataset(path, list, imsize, stride):
    specs, rgbs, ycrcbs = get_all_mats(path, list, imsize, stride)
    return specs, rgbs, ycrcbs

def get_full_dataset(path, list):
    specs = []
    rgbs = []
    ycrcbs = []
    pbar = tqdm(list)
    for file in pbar:
        mat = scipy.io.loadmat(path+file)
        spec = np.float32(np.array(mat['cube']))
        rgb = np.array(mat['rgb'])
        ycrcb = np.array(mat['ycrcb'])
        specs.append(spec)
        rgbs.append(rgb)
        ycrcbs.append(ycrcb)
        pbar.set_postfix({'File':file})
    return specs, rgbs, ycrcbs

def split_dataset(path, valid_ratio, test_ratio, imsize, mode = 'train', stride = 128):
    filelist = os.listdir(path)
    filelist.sort()
    reg = re.compile(r'.*.mat')
    matlist = []
    for file in filelist:
        if re.match(reg, file):
            matlist.append(file)
    all_length = len(matlist)
    test_size = int(all_length * test_ratio)
    valid_size = int(all_length * valid_ratio)
    train_size = all_length - test_size - valid_size
    matlist.sort()
    trainlist = []
    validlist = []
    testlist = []
    for i in range(train_size):
        trainlist.append(matlist[i])
    for j in range(valid_size):
        validlist.append(matlist[j + train_size])
    for k in range(test_size):
        testlist.append(matlist[k + train_size + valid_size])
    if mode == 'train':
        train_sets = get_dataset(path, trainlist, imsize, stride)
        return train_sets
    elif mode == 'val':
        valid_sets = get_dataset(path, validlist, imsize, stride)
        return valid_sets
    elif mode == 'test':
        test_sets = get_dataset(path, testlist, imsize, stride)
        return test_sets

def split_full(path, valid_ratio, test_ratio, mode = 'train'):
    filelist = os.listdir(path)
    filelist.sort()
    reg = re.compile(r'.*.mat')
    matlist = []
    for file in filelist:
        if re.match(reg, file):
            matlist.append(file)
    all_length = len(matlist)
    test_size = int(all_length * test_ratio)
    valid_size = int(all_length * valid_ratio)
    train_size = all_length - test_size - valid_size
    matlist.sort()
    trainlist = []
    validlist = []
    testlist = []
    for i in range(train_size):
        trainlist.append(matlist[i])
    for j in range(valid_size):
        validlist.append(matlist[j + train_size])
    for k in range(test_size):
        testlist.append(matlist[k + train_size + valid_size])
    match mode:
        case 'train':
            datasets = get_full_dataset(path, trainlist)
        case 'val':
            datasets = get_full_dataset(path, validlist)
        case 'test':
            datasets = get_full_dataset(path, testlist)
    return datasets
    # if mode == 'train':
    #     train_sets = get_full_dataset(path, trainlist)
    #     return train_sets
    # elif mode == 'val':
    #     valid_sets = get_full_dataset(path, validlist)
    #     return valid_sets
    # elif mode == 'test':
    #     test_sets = get_full_dataset(path, testlist)
    #     return test_sets

def random_split_full(path, valid_ratio, test_ratio, mode = 'train'):
    filelist = os.listdir(path)
    filelist.sort()
    reg = re.compile(r'.*.mat')
    matlist = []
    for file in filelist:
        if re.match(reg, file):
            matlist.append(file)
    all_length = len(matlist)
    test_size = int(all_length * test_ratio)
    valid_size = int(all_length * valid_ratio)
    train_size = all_length - test_size - valid_size
    matlist.sort()
    trainlist = []
    validlist = []
    testlist = []
    generator = torch.Generator().manual_seed(42)
    indices = randperm(31, generator=generator).tolist()
    indices = randperm(all_length, generator=generator).tolist()
    for i in range(train_size):
        trainlist.append(matlist[indices[i]])
    for j in range(valid_size):
        validlist.append(matlist[indices[j + train_size]])
    for k in range(test_size):
        testlist.append(matlist[indices[k + train_size + valid_size]])
    if mode == 'train':
        train_sets = get_full_dataset(path, trainlist)
        return train_sets
    elif mode == 'val':
        valid_sets = get_full_dataset(path, validlist)
        return valid_sets
    elif mode == 'test':
        test_sets = get_full_dataset(path, testlist)
        return test_sets

def random_split_dataset(path, valid_ratio, test_ratio, imsize, mode = 'train', stride = 128):
    filelist = os.listdir(path)
    filelist.sort()
    reg = re.compile(r'.*.mat')
    matlist = []
    for file in filelist:
        if re.match(reg, file):
            matlist.append(file)
    all_length = len(matlist)
    test_size = int(all_length * test_ratio)
    valid_size = int(all_length * valid_ratio)
    train_size = all_length - test_size - valid_size
    matlist.sort()
    trainlist = []
    validlist = []
    testlist = []
    generator = torch.Generator().manual_seed(42)
    indices = randperm(31, generator=generator).tolist()
    indices = randperm(all_length, generator=generator).tolist()
    for i in range(train_size):
        trainlist.append(matlist[indices[i]])
    for j in range(valid_size):
        validlist.append(matlist[indices[j + train_size]])
    for k in range(test_size):
        testlist.append(matlist[indices[k + train_size + valid_size]])
    if mode == 'train':
        train_sets = get_dataset(path, trainlist, imsize, stride)
        return train_sets
    elif mode == 'val':
        valid_sets = get_dataset(path, validlist, imsize, stride)
        return valid_sets
    elif mode == 'test':
        test_sets = get_dataset(path, testlist, imsize, stride)
        return test_sets

def get_all_mats(path, filelist, imsize, stride, rescale = False):
    specs = []
    rgbs = []
    ycrcbs = []
    pbar = tqdm(filelist)
    for file in pbar:
        mat = scipy.io.loadmat(path+file)
        if rescale:
            spec, rgb, ycrcb = get_all_patches_with_rescale(mat, imsize, stride)
        else:
            spec, rgb, ycrcb = get_all_patches(mat, imsize, stride)
        specs += spec
        rgbs += rgb
        ycrcbs += ycrcb
        pbar.set_postfix({'File':file})
    return specs, rgbs, ycrcbs
    
def Resize(hyper, rgb, ycrcb, h, w):
    hyper_s = cv2.resize(hyper, [h, w], interpolation = cv2.INTER_LINEAR)
    rgb_s = cv2.resize(rgb, [h, w], interpolation = cv2.INTER_LINEAR)
    ycrcb_s = cv2.resize(ycrcb, [h, w], interpolation = cv2.INTER_LINEAR)
    return hyper_s, rgb_s, ycrcb_s

def data_resize(mat, imsize, one = True):
    spectral_images = []
    rgb_images = []
    ycrcb_images = []
    hyper = np.float32(np.array(mat['cube']))
    if one:
        rgb = normalization(np.array(mat['rgb']), np.array(mat['rgb']).max(), np.array(mat['rgb']).min())
        ycrcb = normalization(np.array(mat['ycrcb']), np.array(mat['ycrcb']).max(), np.array(mat['ycrcb']).min())
    else:
        rgb = normalization(np.array(mat['rgb']), 255.0, 0.0)
        ycrcb = normalization(np.array(mat['ycrcb']), 240.0, 0.0)
    h = rgb.shape[0]
    w = rgb.shape[1]
    while min(h // imsize, w // imsize) >= 1:
        hyper_s, rgb_s, ycrcb_s = Resize(hyper, rgb, ycrcb, h, w)
        spectral_images.append(hyper_s)
        rgb_images.append(rgb_s)
        ycrcb_images.append(ycrcb_s)
        h = h // 2
        w = w // 2
    if h > imsize / 2 or w > imsize / 2 :
        hyper_s, rgb_s, ycrcb_s = Resize(hyper, rgb, ycrcb, imsize, imsize)
        spectral_images.append(hyper_s)
        rgb_images.append(rgb_s)
        ycrcb_images.append(ycrcb_s)
    return spectral_images, rgb_images, ycrcb_images

def patch_gen(array, imsize, h, w):
    return array[h-imsize:h, w-imsize:w, :]

def Im2Patch(img, imsize, stride):
    patches = []
    h, w, c = img.shape
    if imsize >= h and imsize >= w:
        patches.append(img)
    else:
        h_num = (h - imsize) // stride + 1
        w_num = (w - imsize) // stride + 1
        h_last = (h - imsize) % stride
        w_last = (w - imsize) % stride
        for i in range(h_num):
            start_h = i * stride
            end_h = i * stride + imsize
            for j in range(w_num):
                patch = img[start_h:end_h,j * stride:j * stride + imsize, :]
                patches.append(patch)
                if j == w_num - 1 and w_last > 0:
                    patch = img[start_h:end_h, -imsize:, :]
                    patches.append(patch)
            if i == h_num - 1 and h_last > 0:
                start_h = -imsize
                for j in range(w_num):
                    patch = img[start_h:,j * stride:j * stride + imsize, :]
                    patches.append(patch)
                    if j == w_num - 1 and w_last > 0:
                        patch = img[start_h:, -imsize:, :]
                        patches.append(patch)
    return patches

def patch_image(array, imsize, stride=8):
    h = array.shape[0]
    w = array.shape[1]
    assert h >= imsize
    assert w >= imsize
    iter_h = int(np.ceil(h / imsize) + 1)
    iter_w = int(np.ceil(w / imsize) + 1)
    patches = []
    for i in range(1, iter_h):
        for j in range(1, iter_w):
            patches.append(patch_gen(array, imsize, min(h, i * imsize), min(w, j * imsize)))
    return patches
    
def get_all_patches_with_rescale(mat, imsize, stride):
    spectrals = []
    rgbs = []
    ycrcbs = []
    resize_spectrals, resize_rgbs, resize_ycrcbs = data_resize(mat, imsize)
    for i in range(len(resize_spectrals)):
        hyperpatches = Im2Patch(resize_spectrals[i], imsize, stride)
        rgbpatches = Im2Patch(resize_rgbs[i], imsize, stride)
        ycrcbpatches = Im2Patch(resize_ycrcbs[i], imsize, stride)
        spectrals += hyperpatches
        rgbs += rgbpatches
        ycrcbs += ycrcbpatches
    return spectrals, rgbs, ycrcbs

def get_all_patches(mat, imsize, stride = 128, one = True):
    spectrals = []
    rgbs = []
    ycrcbs = []
    # resize_spectrals, resize_rgbs = data_resize(mat, imsize)
    # for i in range(len(resize_spectrals)):
    hyper = np.float32(np.array(mat['cube']))
    if one:
        rgb = normalization(np.array(mat['rgb']), np.array(mat['rgb']).max(), np.array(mat['rgb']).min())
        ycrcb = normalization(np.array(mat['ycrcb']), np.array(mat['ycrcb']).max(), np.array(mat['ycrcb']).min())
    else:
        rgb = normalization(np.array(mat['rgb']), 255.0, 0.0)
        ycrcb = normalization(np.array(mat['ycrcb']), 240.0, 0.0)
    # hyperpatches = patch_image(hyper, imsize, stride)
    # rgbpatches = patch_image(rgb, imsize, stride)
    hyperpatches = Im2Patch(hyper, imsize, stride)
    rgbpatches = Im2Patch(rgb, imsize, stride)
    ycrcbpatches = Im2Patch(ycrcb, imsize, stride)
    spectrals += hyperpatches
    rgbs += rgbpatches
    ycrcbs += ycrcbpatches
    return spectrals, rgbs, ycrcbs

def normalization(a, a_max, a_min):
    return (a - a_min) / (a_max - a_min)

if __name__ == '__main__':
    generator = torch.Generator().manual_seed(42)
    indices = randperm(950, generator=generator).tolist()
    print(indices[-1])
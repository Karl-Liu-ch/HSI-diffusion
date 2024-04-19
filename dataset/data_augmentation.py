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
generator = torch.Generator().manual_seed(42)
indices = randperm(31, generator=generator).tolist()

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

def get_dataset(path, list, imsize):
    specs, rgbs = get_all_mats(path, list, imsize)
    return specs, rgbs

def get_full_dataset(path, list):
    specs = []
    rgbs = []
    pbar = tqdm(list)
    for file in pbar:
        mat = scipy.io.loadmat(path+file)
        spec = np.float32(np.array(mat['cube']))
        rgb = np.float32(np.array(mat['rgb']))
        specs.append(spec)
        rgbs.append(rgb)
        pbar.set_postfix({'File':file})
    return specs, rgbs

def split_train(path, valid_ratio, test_ratio, imsize):
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
    train_sets = get_dataset(path, trainlist, imsize)
    return train_sets

def split_test(path, valid_ratio, test_ratio, imsize):
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
    test_sets = get_dataset(path, testlist, imsize)
    return test_sets

def split_full_test(path, valid_ratio, test_ratio):
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
    test_sets = get_full_dataset(path, testlist)
    return test_sets

def split_full_valid(path, valid_ratio, test_ratio):
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
    valid_sets = get_full_dataset(path, validlist)
    return valid_sets

def split_valid(path, valid_ratio, test_ratio, imsize):
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
    valid_sets = get_dataset(path, validlist, imsize)
    return valid_sets

def random_split_train(path, valid_ratio, test_ratio, imsize):
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

    indices = randperm(all_length, generator=generator).tolist()
    for i in range(train_size):
        trainlist.append(matlist[indices[i]])
    for j in range(valid_size):
        validlist.append(matlist[indices[j + train_size]])
    for k in range(test_size):
        testlist.append(matlist[indices[k + train_size + valid_size]])
    train_sets = get_dataset(path, trainlist, imsize)
    return train_sets

def random_split_test(path, valid_ratio, test_ratio, imsize):
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
    indices = randperm(all_length, generator=generator).tolist()
    for i in range(train_size):
        trainlist.append(matlist[indices[i]])
    for j in range(valid_size):
        validlist.append(matlist[indices[j + train_size]])
    for k in range(test_size):
        testlist.append(matlist[indices[k + train_size + valid_size]])
    test_sets = get_dataset(path, testlist, imsize)
    return test_sets

def random_split_full_test(path, valid_ratio, test_ratio):
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
    indices = randperm(all_length, generator=generator).tolist()
    for i in range(train_size):
        trainlist.append(matlist[indices[i]])
    for j in range(valid_size):
        validlist.append(matlist[indices[j + train_size]])
    for k in range(test_size):
        testlist.append(matlist[indices[k + train_size + valid_size]])
    test_sets = get_full_dataset(path, testlist)
    print(indices[-1], all_length)
    return test_sets

def random_split_full_valid(path, valid_ratio, test_ratio):
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
    indices = randperm(all_length, generator=generator).tolist()
    for i in range(train_size):
        trainlist.append(matlist[indices[i]])
    for j in range(valid_size):
        validlist.append(matlist[indices[j + train_size]])
    for k in range(test_size):
        testlist.append(matlist[indices[k + train_size + valid_size]])
    valid_sets = get_full_dataset(path, validlist)
    print(indices[-1], all_length)
    return valid_sets

def random_split_valid(path, valid_ratio, test_ratio, imsize):
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
    indices = randperm(all_length, generator=generator).tolist()
    for i in range(train_size):
        trainlist.append(matlist[indices[i]])
    for j in range(valid_size):
        validlist.append(matlist[indices[j + train_size]])
    for k in range(test_size):
        testlist.append(matlist[indices[k + train_size + valid_size]])
    valid_sets = get_dataset(path, validlist, imsize)
    return valid_sets

def get_all_mats(path, filelist, imsize, rescale = True):
    specs = []
    rgbs = []
    pbar = tqdm(filelist)
    for file in pbar:
        mat = scipy.io.loadmat(path+file)
        if rescale:
            spec, rgb = get_all_patches_with_rescale(mat, imsize)
        else:
            spec, rgb = get_all_patches(mat, imsize)
        specs += spec
        rgbs += rgb
        pbar.set_postfix({'File':file})
    return specs, rgbs
    
def Resize(hyper, rgb, h, w):
    hyper_s = cv2.resize(hyper, [h, w], interpolation = cv2.INTER_LINEAR)
    rgb_s = cv2.resize(rgb, [h, w], interpolation = cv2.INTER_LINEAR)
    return hyper_s, rgb_s

def data_resize(mat, imsize):
    spectral_images = []
    rgb_images = []
    hyper = np.float32(np.array(mat['cube']))
    rgb = np.float32(np.array(mat['rgb']))
    h = rgb.shape[0]
    w = rgb.shape[1]
    while min(h // imsize, w // imsize) >= 1:
        hyper_s, rgb_s = Resize(hyper, rgb, h, w)
        spectral_images.append(hyper_s)
        rgb_images.append(rgb_s)
        h = h // 2
        w = w // 2
    if h > imsize / 2 or w > imsize / 2 :
        hyper_s, rgb_s = Resize(hyper, rgb, imsize, imsize)
        spectral_images.append(hyper_s)
        rgb_images.append(rgb_s)
    return spectral_images, rgb_images

def patch_gen(array, imsize, h, w):
    return array[h-imsize:h, w-imsize:w, :]

def Im2Patch(img, imsize, stride=1):
    patches = []
    h, w, c = img.shape
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
    
def get_all_patches_with_rescale(mat, imsize):
    spectrals = []
    rgbs = []
    resize_spectrals, resize_rgbs = data_resize(mat, imsize)
    for i in range(len(resize_spectrals)):
        hyperpatches = Im2Patch(resize_spectrals[i], imsize, stride=imsize)
        rgbpatches = Im2Patch(resize_rgbs[i], imsize, stride=imsize)
        spectrals += hyperpatches
        rgbs += rgbpatches
    return spectrals, rgbs

def get_all_patches(mat, imsize, stride = 8):
    spectrals = []
    rgbs = []
    # resize_spectrals, resize_rgbs = data_resize(mat, imsize)
    # for i in range(len(resize_spectrals)):
    hyper = np.float32(np.array(mat['cube']))
    rgb = np.float32(np.array(mat['rgb']))
    # hyperpatches = patch_image(hyper, imsize, stride)
    # rgbpatches = patch_image(rgb, imsize, stride)
    hyperpatches = Im2Patch(hyper, imsize, stride)
    rgbpatches = Im2Patch(rgb, imsize, stride)
    spectrals += hyperpatches
    rgbs += rgbpatches
    return spectrals, rgbs

if __name__ == '__main__':
    generator = torch.Generator().manual_seed(42)
    indices = randperm(950, generator=generator).tolist()
    print(indices[-1])
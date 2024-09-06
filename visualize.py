from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os
from torch import nn, optim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import SpectralAngleMapper, PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from spectral import *
import cv2
from NTIRE2022Util import *
import scipy.io
from utils import *
import matplotlib.pyplot as plt
from PIL import Image

def visualize_tensor(x, t, name, width = 256, gap_height = 64):
    hsi = t
    hyper = x[0].cpu().numpy().transpose(1,2,0)
    filter, _ = load_rgb_filter('resources/CIE_xyz_1964_10deg.csv')
    try:
        os.mkdir(f'results/{name}/')
    except:
        pass
    images = []
    bands = [400, 500, 550, 700]
    b = 0

    for band in bands:
        image = np.zeros([128, 128, 3])
        # Assuming 'filter' and 'hyper' are defined
        for i in range(128):
            for j in range(128):
                image[i, j, :] = filter[band - 360, :] * hyper[i, j, b]
        image = (image - image.min()) / (image.max() - image.min())
        images.append(image)

    # Calculate the total height of the result image including gaps
    result_height = 128 * len(images) + gap_height * (len(images) - 1)

    # Create a new blank image with white background
    result_image = Image.new('RGB', (width, result_height), color='white')

    # Paste each individual image into the result with 10 pixels gap
    for idx, image in enumerate(images):
        pil_image = Image.fromarray((image * 255).astype('uint8'))
        result_image.paste(pil_image, ((width-128) // 2, (128 + gap_height) * idx))

    # Add a title below the combined image
    title = t
    plt.figure(figsize=(8, 8))
    plt.imshow(result_image)
    plt.title(title, fontsize=10, pad=10)
    plt.axis('off')
    plt.savefig(f'results/{name}/{hsi}.png', format='PNG', bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)

    # Open and save the result image again using PIL to remove compression
    img = Image.open(f'results/{name}/{hsi}.png')
    img.save(f'results/{name}/{hsi}.png', format='PNG', compress_level=0)

def visualize_result(name, root, hsi, width = 256, gap_height = 64):
    sample = scipy.io.loadmat(root + str(hsi).zfill(3) + '.mat')
    hyper = sample['cube']
    rgb = sample['rgb']
    filter, _ = load_rgb_filter('resources/CIE_xyz_1964_10deg.csv')
    try:
        os.mkdir(f'results/{name}/')
    except:
        pass
    images = []
    bands = [400, 500, 550, 700]
    b = 0

    images.append(rgb)
    for band in bands:
        image = np.zeros([128, 128, 3])
        # Assuming 'filter' and 'hyper' are defined
        for i in range(128):
            for j in range(128):
                image[i, j, :] = filter[band - 360, :] * hyper[i, j, b]
        image = (image - image.min()) / (image.max() - image.min())
        images.append(image)

    # Calculate the total height of the result image including gaps
    result_height = 128 * len(images) + gap_height * (len(images) - 1)

    # Create a new blank image with white background
    result_image = Image.new('RGB', (width, result_height), color='white')

    # Paste each individual image into the result with 10 pixels gap
    for idx, image in enumerate(images):
        pil_image = Image.fromarray((image * 255).astype('uint8'))
        result_image.paste(pil_image, ((width-128) // 2, (128 + gap_height) * idx))

    # Add a title below the combined image
    title = name
    plt.figure(figsize=(8, 8))
    plt.imshow(result_image)
    plt.title(title, fontsize=10, pad=10)
    plt.axis('off')
    plt.savefig(f'results/{name}/{hsi}.png', format='PNG', bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)

    # Open and save the result image again using PIL to remove compression
    img = Image.open(f'results/{name}/{hsi}.png')
    img.save(f'results/{name}/{hsi}.png', format='PNG', compress_level=0)

def plot_spectral_density(root, hsi):
    sample = scipy.io.loadmat(root + str(hsi).zfill(3) + '.mat')
    hyper = sample['cube']
    bands = np.linspace(400,700,31)
    b = 0
    spectral_density = []
    for band in bands:
        band = int(band)
        spectral_density.append(hyper[:,:,b].mean())
        b += 1
    # plt.plot(bands, spectral_density)
    return np.asarray(spectral_density)

def concat_results(hsi, modelnames, gap_height=0):
    img1 = Image.open(f'results/GT/{hsi}.png')
    idx = len(modelnames) + 1
    result_width = img1.width * idx + gap_height * (idx - 1)
    result_image = Image.new('RGB', (result_width, img1.height), color='white')
    i = 0
    for k, v in modelnames.items():
        img = Image.open(f'results/{v}/{hsi}.png')
        result_image.paste(img, ((img1.width + gap_height) * i, 0))
        i += 1
    img = Image.open(f'results/GT/{hsi}.png')
    result_image.paste(img, ((img1.width + gap_height) * i, 0))
    result_image.save(f'results/{hsi}concat.png', format='PNG', compress_level=0)

if __name__ == '__main__':
    modelnames = {'CVAE': 'CVAE', 
                  'CVAESP': 'CVAE-HS', 
                  'pix2pix': 'pix2pix', 
                  'SNCWGANres': 'SNCWGAN+ResNet', 
                  'SNCWGANunet': 'SNCWGAN+Unet', 
                  'SNCWGANdense': 'SNCWGAN+DenseNet', 
                  'HSCNN_Plus': 'HSCNN_Plus', 
                  'MSTPlusPlus': 'MST++', 
                  'D2GANNZ': 'D2GAN', 
                  'SNCWGANNoNoise': 'SNCWGAN+DT'}
    root = '/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/'
    hsis = [5642,5453,5621]
    for k, v in modelnames.items():
        path = root + k + '/'
        Dict = {}
        Dict[v] = {}
        for hsi in hsis:
            visualize_result(v, path, hsi, width=224, gap_height=64)
            Dict[v][hsi] = plot_spectral_density(path, hsi)
        modelnames[k] = Dict

    GT_dict = {}
    GT_dict['GT'] = {}
    path = '/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/'
    for hsi in hsis:
        GT_dict['GT'][hsi] = plot_spectral_density(path, hsi)
        visualize_result('GT', path, hsi, width=224, gap_height=64)
    # for hsi in hsis:
    #     concat_results(hsi, modelnames, gap_height=0)
    # bands = np.linspace(400,700,31)
    # plt.plot(bands, GT_dict['GT'][5642])
    # plt.show()
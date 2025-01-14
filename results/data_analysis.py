import sys
sys.path.append('./')
import pandas as pd
import re
from dataset.datasets import TrainDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.io
import os

root_path = '/work3/s212645/Spectral_Reconstruction/'

dataset_name = 'BGU'

# for i in range(31):
#     df = {}
#     n = i * 10 + 620
#     band = str(n)
#     index = (n - 400) // 10

#     arad_path = root_path+'clean/ARAD/'
#     dir = os.listdir(arad_path)
#     n = len(dir)
#     arads = []
#     arad_ycrcb = []
#     for itm in tqdm(range(n)):
#         name = str(itm+1)
#         name = name.zfill(3) + '.mat'
#         mat = scipy.io.loadmat(arad_path + name)
#         # hyper = mat['cube'].reshape(-1,31)[:,index].mean()
#         hyper = mat['cube'].reshape(-1,31)[:,index]
#         arads.append(hyper)
#     # arads = np.array(arads)
#     arads = np.concatenate(arads, axis=0)

#     caves = []
#     cave_ycrcb = []
#     cave_path = root_path+f'clean/{dataset_name}/'
#     dir = os.listdir(cave_path)
#     n = len(dir)
#     for itm in tqdm(range(n)):
#         name = str(itm+1)
#         name = name.zfill(3) + '.mat'
#         mat = scipy.io.loadmat(cave_path + name)
#         # hyper = mat['cube'].reshape(-1,31)[:,index].mean()
#         hyper = mat['cube'].reshape(-1,31)[:,index]
#         caves.append(hyper)
#     # caves = np.array(caves)
#     caves = np.concatenate(caves, axis=0)

#     df['ARAD'] = arads
#     df[f'{dataset_name}'] = caves
#     fig = sns.boxplot(df)
#     plt.title(f'{band}-hs')
#     plt.savefig(f'results/{dataset_name}/{band}-hs.png')
#     plt.close()

# df = {}

# caves = []
# cave_ycrcb = []
# cave_path = root_path+f'clean/{dataset_name}/'
# dir = os.listdir(cave_path)
# n = len(dir)
# for itm in tqdm(range(n)):
#     name = str(itm+1)
#     name = name.zfill(3) + '.mat'
#     mat = scipy.io.loadmat(cave_path + name)
#     # rgb = mat['rgb'].reshape(-1,3)
#     ycrcb = mat['ycrcb'].reshape(-1,3)[:,0]
#     cave_ycrcb.append(ycrcb)
#     # np.concatenate([np.array(cave_ycrcb), ycrcb], axis=0)
# # cave_ycrcb = np.array(cave_ycrcb).reshape(-1)
# cave_ycrcb = np.concatenate(cave_ycrcb, axis=0)

# arad_path = root_path+'clean/ARAD/'
# dir = os.listdir(arad_path)
# n = len(dir)
# arads = []
# arad_ycrcb = []
# for itm in tqdm(range(n)):
#     name = str(itm+1)
#     name = name.zfill(3) + '.mat'
#     mat = scipy.io.loadmat(arad_path + name)
#     # rgb = mat['rgb'].reshape(-1,3)
#     ycrcb = mat['ycrcb'].reshape(-1,3)[:,0]
#     arad_ycrcb.append(ycrcb)
# # arad_ycrcb = np.array(arad_ycrcb).reshape(-1)
# arad_ycrcb = np.concatenate(arad_ycrcb, axis=0)

# df['ARAD'] = arad_ycrcb
# df[f'{dataset_name}'] = cave_ycrcb
# fig = sns.boxplot(df)
# plt.title('Luminance')
# plt.savefig(f'results/{dataset_name}/Luminance.png')
# plt.close()


caves = []
cave_ycrcb = []
cave_path = root_path+f'clean/{dataset_name}/'
dir = os.listdir(cave_path)
n = len(dir)

for itm in tqdm(range(n)):
    name = str(itm+1)
    name = name.zfill(3) + '.mat'
    mat = scipy.io.loadmat(cave_path + name)
    hyper = mat['cube'].reshape(-1,31).mean(axis=0)
    caves.append(hyper)
# caves = np.concatenate(caves, axis = 0)
caves = np.array(caves)
print(caves.shape)
plt.bar(np.linspace(400, 700, 31), caves.mean(axis=0), align='center', width=10, alpha=0.7, ecolor='black', capsize=10, color='blue', edgecolor='black')
plt.errorbar(np.linspace(400, 700, 31), caves.mean(axis=0), yerr=caves.std(axis=0), fmt='none', ecolor='red', capsize=5, label='Std Dev')
plt.title(f'{dataset_name} HSI Histogram with Mean and Std Dev')
plt.xlabel('Band (nm)')
plt.ylabel('Mean Value')
plt.legend()
plt.savefig(f'results/{dataset_name}-hsi-histogram.png')
plt.close()

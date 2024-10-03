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

root_path = '/work3/s212645/Spectral_Reconstruction/'
# arad = TrainDataset(root_path, 1e8, 0, 0, False, ['ARAD/'], True)
# bgu = TrainDataset(root_path, 1e8, 0, 0, False, ['BGU/'], True)
# cave = TrainDataset(root_path, 1e8, 0, 0, False, ['CAVE/'], True)

arad_path = root_path+'clean/ARAD/'
arads = []
arad_ycrcb = []
for itm in tqdm(range(950)):
    name = str(itm+1)
    name = name.zfill(3) + '.mat'
    mat = scipy.io.loadmat(arad_path + name)
    hyper = mat['cube'].reshape(-1,31)
    # rgb = mat['rgb'].reshape(-1,3)
    ycrcb = mat['ycrcb'].reshape(-1,3)
    arads.append(hyper)
    arad_ycrcb.append(ycrcb)
arads = np.array(arads).reshape(-1,31)
arad_ycrcb = np.array(arad_ycrcb).reshape(-1,3)

caves = []
cave_ycrcb = []
cave_path = root_path+'clean/CAVE/'
for itm in tqdm(range(31)):
    name = str(itm+1)
    name = name.zfill(3) + '.mat'
    mat = scipy.io.loadmat(arad_path + name)
    hyper = mat['cube'].reshape(-1,31)
    # rgb = mat['rgb'].reshape(-1,3)
    ycrcb = mat['ycrcb'].reshape(-1,3)
    caves.append(hyper)
    cave_ycrcb.append(ycrcb)
caves = np.array(caves).reshape(-1,31)
cave_ycrcb = np.array(cave_ycrcb).reshape(-1,3)

# plt.bar(np.linspace(400, 700, 31), arads.mean(axis=0), align='center', width=10, alpha=0.7, ecolor='black', capsize=10, color='blue', edgecolor='black')
# plt.errorbar(np.linspace(400, 700, 31), arads.mean(axis=0), yerr=arads.std(axis=0), fmt='none', ecolor='red', capsize=5, label='Std Dev')
# plt.title('ARAD HSI Histogram with Mean and Std Dev')
# plt.xlabel('Band (nm)')
# plt.ylabel('Mean Value')
# plt.legend()
# plt.savefig('results/arad-hsi-histogram.png')
# plt.close()

# plt.bar(np.linspace(400, 700, 31), caves.mean(axis=0), align='center', width=10, alpha=0.7, ecolor='black', capsize=10, color='blue', edgecolor='black')
# plt.errorbar(np.linspace(400, 700, 31), caves.mean(axis=0), yerr=caves.std(axis=0), fmt='none', ecolor='red', capsize=5, label='Std Dev')
# plt.title('CAVE HSI Histogram with Mean and Std Dev')
# plt.xlabel('Band (nm)')
# plt.ylabel('Mean Value')
# plt.legend()
# plt.savefig(f'results/cave-hsi-histogram.png')
# plt.close()

# for i in tqdm(range(2)):
i = 30
df = {}
band = i * 10 + 400
band = str(band)
df['ARAD'] = arads[:,i]
df['CAVE'] = caves[:,i]
fig = sns.boxplot(df)
plt.title(f'{band}-hs')
plt.savefig(f'results/{band}-hs.png')
plt.close()

# for i in tqdm(range(31)):
# i = 2
# df = {}
# df['ARAD'] = arad_ycrcb[:,i]
# df['CAVE'] = cave_ycrcb[:,i]
# fig = sns.boxplot(df)
# plt.title(f'ycrcb-{i}')
# plt.savefig(f'results/{i}-ycrcb.png')
# plt.close()

# df = {}
# df['ARAD'] = arads.reshape(-1)
# df['CAVE'] = caves.reshape(-1)
# fig = sns.boxplot(df)
# plt.title(f'Spectrum-All')
# plt.savefig(f'results/All-Band-HSI.png')
# plt.close()
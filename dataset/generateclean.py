import sys
sys.path.append('./')
from utils import reconRGBfromNumpy
import numpy as np
import scipy.io
import os
import h5py
from utils import *

root = '/work3/s212645/Spectral_Reconstruction/'
ARAD = 'ARAD/'
BGU1 = '/work3/s212645/Spectral_Reconstruction/BGU_HS/NTIRE2018_Train1_Spectral/'
BGU2 = '/work3/s212645/Spectral_Reconstruction/BGU_HS/NTIRE2018_Train2_Spectral/'
BGU3 = '/work3/s212645/Spectral_Reconstruction/BGU_HS/NTIRE2018_Validate_Spectral/'

cleanpath = '/work3/s212645/Spectral_Reconstruction/clean/'
# os.mkdir(cleanpath)
# os.mkdir(cleanpath + ARAD)
# os.mkdir(cleanpath + 'BGU/')
# os.mkdir(cleanpath + 'CAVE/')

# for i in range(950):
#     mat_ = {}
#     matpath = str(i + 1).zfill(3) + '.mat'
#     matpath = root + ARAD + matpath
#     mat = scipy.io.loadmat(matpath)['mat']
#     hyper = mat[0][0][1]
#     rgbrecon = reconRGBfromNumpy(hyper)
#     mat_['rgb'] = rgbrecon
#     mat_['rgb'] = Image256(mat_['rgb'])
#     mat_['cube'] = hyper
#     savepath = cleanpath + ARAD + str(i + 1).zfill(3) + '.mat'
#     scipy.io.savemat(savepath, mat_)
#     print(matpath + ' finish')

# for i in range(203):
#     matpath = str(i + 1).zfill(3) + '.mat'
#     matpath = BGU1 + 'BGU_HS_00' + matpath
#     Mat = {}
#     with h5py.File(matpath, 'r') as mat:
#         hyper = np.transpose(mat.get('rad'), [2,1,0])
#         hyper = (hyper-hyper.min())/(hyper.max()-hyper.min())
#         Mat['cube'] = hyper
#         rgbrecon = reconRGBfromNumpy(hyper)
#         Mat['rgb'] = rgbrecon
#         savepath = cleanpath + 'BGU/' + str(i + 1).zfill(3) + '.mat'
#         scipy.io.savemat(savepath, Mat)
    
# for i in range(53):
#     matpath = str(i + 204).zfill(3) + '.mat'
#     matpath = BGU2 + 'BGU_HS_00' + matpath
#     Mat = {}
#     with h5py.File(matpath, 'r') as mat:
#         hyper = np.transpose(mat.get('rad'), [2,1,0])
#         hyper = (hyper-hyper.min())/(hyper.max()-hyper.min())
#         Mat['cube'] = hyper
#         rgbrecon = reconRGBfromNumpy(hyper)
#         Mat['rgb'] = rgbrecon
#         savepath = cleanpath + 'BGU/' + str(i + 204).zfill(3) + '.mat'
#         scipy.io.savemat(savepath, Mat)

# for i in range(5):
#     matpath = str(i * 2 + 257).zfill(3) + '.mat'
#     matpath = BGU3 + 'BGU_HS_00' + matpath
#     Mat = {}
#     with h5py.File(matpath, 'r') as mat:
#         hyper = np.transpose(mat.get('rad'), [2,1,0])
#         hyper = (hyper-hyper.min())/(hyper.max()-hyper.min())
#         Mat['cube'] = hyper
#         rgbrecon = reconRGBfromNumpy(hyper)
#         Mat['rgb'] = rgbrecon
#         savepath = cleanpath + 'BGU/' + str(i + 257).zfill(3) + '.mat'
#         scipy.io.savemat(savepath, Mat)

for i in range(950):
    matpath = str(i + 1).zfill(3) + '.mat'
    matpath = cleanpath + ARAD + matpath
    mat = scipy.io.loadmat(matpath)
    hyper = mat['cube'].astype(np.float32)
    rgbrecon = reconRGBfromNumpy(hyper)
    mat['rgb'] = rgbrecon.astype(np.float32)
    # mat['rgb'] = Normalize(mat['rgb'])
    # mat['rgb'] = Image256(mat['rgb'])
    savepath = cleanpath + ARAD + str(i + 1).zfill(3) + '.mat'
    scipy.io.savemat(savepath, mat)
    print(matpath + ' finish')

for i in range(261):
    matpath = str(i + 1).zfill(3) + '.mat'
    matpath = cleanpath + 'BGU/' + matpath
    mat = scipy.io.loadmat(matpath)
    # mat['rgb'] = Image256(mat['rgb'])
    hyper = mat['cube'].astype(np.float32)
    rgbrecon = reconRGBfromNumpy(hyper)
    mat['rgb'] = rgbrecon.astype(np.float32)
    savepath = cleanpath + 'BGU/' + str(i + 1).zfill(3) + '.mat'
    scipy.io.savemat(savepath, mat)
    print(matpath + ' finish')

for i in range(31):
    matpath = str(i + 1).zfill(3) + '.mat'
    matpath = cleanpath + 'CAVE/' + matpath
    mat = scipy.io.loadmat(matpath)
    hyper = mat['cube'].astype(np.float32)
    rgbrecon = reconRGBfromNumpy(hyper)
    # rgbrecon = Image256(rgbrecon)
    mat['rgb'] = rgbrecon.astype(np.float32)
    savepath = cleanpath + 'CAVE/' + str(i + 1).zfill(3) + '.mat'
    scipy.io.savemat(savepath, mat)

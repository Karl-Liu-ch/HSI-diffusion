import sys
sys.path.append('./')
import torch
import numpy as np
import scipy.io
import re
from utils import *
from dataset.datasets import *
from options import opt
from tqdm import tqdm
import os
import colour
from NTIRE2022Util import *
from skimage.metrics import structural_similarity as ssim
import seaborn as sns
from dataset.hsi_dataset import ValidDataset as ValidDataset_Orig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def computeDeltaE(recovered, groundTruth):
    image1_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(groundTruth))
    image2_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(recovered))
    deltae = colour.delta_E(image1_lab, image2_lab, method="CIE 2000")
    return deltae.mean()

def back_projection(img, camera_filter):
    return np.matmul(img, camera_filter)

def test(modelname = 'DTN', dataname = 'ARAD/', valid_ratio=0.1, test_ratio=0.1):
    root = '/work3/s212645/Spectral_Reconstruction/'
    fake_root = f'/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/{modelname}-{dataname}'
    arad_test = TestDataset(data_root= root, crop_size=1e8, valid_ratio=valid_ratio, test_ratio=test_ratio, arg=False, datanames=[dataname])

    filelist = os.listdir(fake_root)
    filelist.sort()
    reg = re.compile(r'.*.mat')
    matlist = []
    for file in filelist:
        if re.match(reg, file):
            matlist.append(file)
    matlist.sort()

    mrae_errors = AverageMeter()
    rmse_errors = AverageMeter()
    psnr_errors = AverageMeter()
    sam_errors = AverageMeter()
    ssim_errors = AverageMeter()
    psnrrgb_errors = AverageMeter()
    mraergb_errors = AverageMeter()
    deltaE_errors = AverageMeter()

    pbar = tqdm(range(len(matlist)))
    for i in pbar:
        realmat = arad_test[i]
        fakemat = scipy.io.loadmat(os.path.join(fake_root,matlist[i]))
        realhyper = realmat['label'].transpose(1,2,0)
        fakehyper = fakemat['cube'].astype(np.float32)
        if realhyper.all() == False:
            realhyper += 1e-5
            fakehyper += 1e-5
        realrgb = back_projection(realhyper, CAM_FILTER) / 10.0
        fakergb = back_projection(fakehyper, CAM_FILTER) / 10.0

        mraeerror = computeMRAE(realhyper, fakehyper)
        mrae_errors.update(mraeerror)
        rmseerror = compute_rmse(realhyper, fakehyper)
        rmse_errors.update(rmseerror)
        psnrerror = compute_psnr(realhyper, fakehyper, 1.0)
        psnr_errors.update(psnrerror)
        samerror = compute_sam(realhyper, fakehyper)
        sam_errors.update(samerror)

        mraergberror = computeMRAE(realrgb, fakergb)
        mraergb_errors.update(mraergberror)
        deltaE_error = computeDeltaE(fakergb, realrgb)
        deltaE_errors.update(deltaE_error)
        prsnrgb = compute_psnr(fakergb, realrgb, max(realrgb.max(), fakergb.max()))
        psnrrgb_errors.update(prsnrgb)
        ssimerror = ssim(fakergb, realrgb, channel_axis=2, data_range=max(realrgb.max() - realrgb.min(), fakergb.max() - fakergb.min()))
        ssim_errors.update(ssimerror)

    file = '/zhome/02/b/164706/Master_Courses/thesis/HSI-diffusion/result.txt'
    f = open(file, 'a')
    f.write(f'{modelname}-{dataname}:\n')
    f.write(f'MRAE:{mrae_errors.avg}, RMSE: {rmse_errors.avg}, PSNR:{psnr_errors.avg}, SAM: {sam_errors.avg}, MRAERGB: {mraergb_errors.avg}, SSIM: {ssim_errors.avg}, PSNRRGB: {psnrrgb_errors.avg}, Delta E: {deltaE_errors.avg}')
    f.write('\n')
    f.close()
    
def SAMHeatMap(preds, target):
    dot_product = np.sum(preds * target, axis=2)
    preds_norm = np.linalg.norm(preds, axis=2)
    target_norm = np.linalg.norm(target, axis=2)
    sam_score = np.arccos(dot_product / (preds_norm * target_norm))
    return sam_score

def computeMRAE_(groundTruth, recovered):
    assert groundTruth.shape == recovered.shape
    difference = np.abs(groundTruth - recovered) / (groundTruth)
    return difference.mean(2)

def DeltaEHeatmap(groundTruth, recovered):
    image1_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(groundTruth))
    image2_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(recovered))
    deltae = colour.delta_E(image1_lab, image2_lab, method="CIE 2000")
    return deltae

def gen_heatmap(real_hsi, fake_hsi, filename):
    mrae = computeMRAE_(real_hsi, fake_hsi)
    sns.heatmap(mrae, cmap='jet', vmin=0, vmax=2)
    plt.axis('off')
    plt.savefig(f'results/{filename}-mrae.png')
    plt.close()
    sam = SAMHeatMap(fake_hsi, real_hsi)
    sns.heatmap(sam, cmap='jet', vmin=0, vmax=2)
    plt.axis('off')
    plt.savefig(f'results/{filename}-sam.png')
    plt.close()

def gen_heatmap_deltaE(real_rgb, fake_rgb, filename):
    deltaE = DeltaEHeatmap(real_rgb, fake_rgb)
    sns.heatmap(deltaE, cmap='jet', vmin=0, vmax=2.0)
    plt.axis('off')
    plt.savefig(f'results/{filename}-deltaE.png')
    plt.close()

def gen_color_map(hsi, filename):
    plt.figure(figsize=[6, 20])
    plt.subplot(4,1,1)
    title = filename.split('/')[0]
    if title == 'MSTPlusPlus':
        title = 'MST++'
    elif title == 'HSCNN_Plus':
        title = 'HSCNND'
    elif title == 'pix2pix':
        title = 'Pix2Pix'
    elif title == 'DTN-SNTransformerDiscriminator':
        title = 'Ours'
    plt.title(title, fontsize=40)
    img1 = np.matmul(hsi[:,:,:8], CAM_FILTER[:8,:])
    img1 = img1 / img1.max()
    plt.imshow(img1)
    plt.axis('off')
    img1 = np.matmul(hsi[:,:,8:16], CAM_FILTER[8:16,:])
    img1 = img1 / img1.max()
    plt.subplot(4,1,2)
    plt.imshow(img1)
    plt.axis('off')
    img1 = np.matmul(hsi[:,:,16:24], CAM_FILTER[16:24,:])
    img1 = img1 / img1.max()
    plt.subplot(4,1,3)
    plt.imshow(img1)
    plt.axis('off')
    img1 = np.matmul(hsi[:,:,24:], CAM_FILTER[24:,:])
    img1 = img1 / img1.max()
    plt.subplot(4,1,4)
    plt.imshow(img1)
    plt.axis('off')
    plt.savefig(f'results/{filename}-img1.png')
    plt.close()

def gen_density(hsi, filename):
    band = np.linspace(400, 700, 31)
    density = hsi.reshape(-1, 31).mean(0)
    title = filename.split('/')[0]
    if title == 'MSTPlusPlus':
        title = 'MST++'
    elif title == 'HSCNN_Plus':
        title = 'HSCNND'
    elif title == 'pix2pix':
        title = 'Pix2Pix'
    elif title == 'DTN-SNTransformerDiscriminator':
        title = 'Ours'
    plt.plot(band, density, '+-', label=title)
    # plt.savefig(f'results/density/{filename}.png')
    # plt.close()

def gen_all_figures(modelnames):
    arads = TestDataset('/work3/s212645/Spectral_Reconstruction/', 1e8, 0.1, 0.1, False, ['ARAD/'])
    # bgus = TestDataset('/work3/s212645/Spectral_Reconstruction/', 1e8, 0.1, 0.1, False, ['BGU/'])
    # caves = TestDataset('/work3/s212645/Spectral_Reconstruction/', 1e8, 0, 1, False, ['CAVE/'])
    # testsets = {'ARAD/': arads, 'BGU/': bgus, 'CAVE/': caves}
    testsets = {'ARAD/': arads}
    for k in testsets.keys():
        test_set = testsets[k]
        dataname = k
        for i, testset in zip(range(len(test_set)), tqdm(test_set)):
            real_hsi = testset['cond'].transpose(1,2,0)
            data = dataname.split('/')[0]
            real_hsi = testset['label'].transpose(1,2,0)
            try:
                os.mkdir(f'results/GT/')
            except:
                pass
            for modelname in modelnames:
                try:
                    os.mkdir(f'results/{modelname}/')
                except:
                    pass
                name = str(i).zfill(3) + '.mat'
                data = dataname.split('/')[0]

                real_hsi = testset['label'].transpose(1,2,0)
                fake_hsi = scipy.io.loadmat(f'/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/{modelname}-{dataname}/{name}')['cube']
                gen_color_map(fake_hsi, f'{modelname}/{data}-{i}')
                gen_heatmap(real_hsi, fake_hsi, f'{modelname}/{data}-{i}')
                real_hsi = testset['cond'].transpose(1,2,0)
                fake_hsi = scipy.io.loadmat(f'/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/{modelname}-{dataname}/{name}')['rgb']
                gen_heatmap_deltaE(real_hsi, fake_hsi, f'{modelname}/{data}-{i}')

def gen_all_density(modelnames):
    # modelnames = ['MSTPlusPlus', 'AWAN', 'HSCNN_Plus', 'Restormer', 'pix2pix', 'SSTransformer']
    arads = TestDataset('/work3/s212645/Spectral_Reconstruction/', 1e8, 0.1, 0.1, False, ['ARAD/'])
    # bgus = TestDataset('/work3/s212645/Spectral_Reconstruction/', 1e8, 0.1, 0.1, False, ['BGU/'])
    # caves = TestDataset('/work3/s212645/Spectral_Reconstruction/', 1e8, 0, 1, False, ['CAVE/'])
    # testsets = {'ARAD/': arads, 'BGU/': bgus, 'CAVE/': caves}
    testsets = {'ARAD/': arads}
    for k in testsets.keys():
        test_set = testsets[k]
        dataname = k
        for i, testset in zip(range(len(test_set)), tqdm(test_set)):
            real_hsi = testset['cond'].transpose(1,2,0)
            data = dataname.split('/')[0]
            real_hsi = testset['label'].transpose(1,2,0)
            try:
                os.mkdir(f'results/GT/')
            except:
                pass
            gen_density(real_hsi, f'GT/{data}-{i}')
            for modelname in modelnames:
                try:
                    os.mkdir(f'results/{modelname}/')
                except:
                    pass
                name = str(i).zfill(3) + '.mat'
                data = dataname.split('/')[0]

                real_hsi = testset['label'].transpose(1,2,0)
                fake_hsi = scipy.io.loadmat(f'/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/{modelname}-{dataname}/{name}')['cube']
                gen_density(fake_hsi, f'{modelname}/{data}-{i}')
            plt.legend()
            plt.savefig(f'results/density/{i}.png')
            plt.close()

def gen_resutls(modelnames):
    datanames = ['ARAD/', 'BGU/', 'CAVE/']
    for modelname in modelnames:
        for dataname in datanames:
            if dataname == 'CAVE/':
                test(modelname=modelname, dataname=dataname, valid_ratio=0, test_ratio=1)
            else:
                test(modelname=modelname, dataname=dataname)


if __name__ == '__main__':
    # modelnames = ['MSTPlusPlus', 'AWAN', 'HSCNN_Plus', 'Restormer', 'pix2pix', 'DTN-SNTransformerDiscriminator', 'SSTransformer']
    modelnames = ['SSTransformer_no_spatial', 'SSTransformer_no_spectral', 'SSTransformer_no_rpe', 'SSTransformer_ycrcb', 'SSTransformer']
    # modelnames = ['SSTransformer_ycrcb']
    gen_all_density(modelnames)
    # gen_all_figures(modelnames)
    # gen_resutls(modelnames)
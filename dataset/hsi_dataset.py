from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
from tqdm import tqdm

class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8, one = True):
        print(f'stride: {stride}, crop size: {crop_size}')
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.ycrcbs = []
        self.arg = arg
        h,w = 482,512  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        hyper_data_path = f'{data_root}/Train_spectral/'
        bgr_data_path = f'{data_root}/Train_RGB/'

        # with open(f'{data_root}/split_txt/train_list.txt', 'r') as fin:
        with open('dataset/split_txt/train_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n','.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr) of ntire2022 dataset:{len(bgr_list)}')
        pbar = tqdm(range(len(hyper_list)))
        for i in pbar:
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                print(hyper_path)
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper =np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            assert hyper_list[i].split('.')[0] ==bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_RGB2YCrCb)
            ycrcb = np.float32(ycrcb)
            if one:
                ycrcb = (ycrcb-ycrcb.min())/(ycrcb.max()-ycrcb.min())
            else:
                ycrcb = ycrcb / 240.0
            ycrcb = np.transpose(ycrcb, [2, 0, 1])  # [3,482,512]

            bgr = np.float32(bgr)
            if one:
                bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            else:
                bgr = bgr / 255.0
            bgr = np.transpose(bgr, [2, 0, 1])  # [3,482,512]

            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            self.ycrcbs.append(np.concatenate([bgr, ycrcb], axis=0))
            mat.close()
            pbar.set_postfix_str(f'Ntire2022 scene {i} is loaded.')
        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

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
        if self.crop_size > 482:
            hyper = self.hypers[idx]
            bgr = self.bgrs[idx]
            ycrcb = self.ycrcbs[idx]
            batch = {'label':np.ascontiguousarray(hyper), 'cond': np.ascontiguousarray(bgr), 'ycrcb': np.ascontiguousarray(ycrcb)}
            return batch
        else:
            stride = self.stride
            crop_size = self.crop_size
            img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
            h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
            bgr = self.bgrs[img_idx]
            ycrcb = self.ycrcbs[img_idx]
            hyper = self.hypers[img_idx]
            bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
            ycrcb = ycrcb[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
            hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            if self.arg:
                bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
                hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
                ycrcb = self.arguement(ycrcb, rotTimes, vFlip, hFlip)
            batch = {'label':np.ascontiguousarray(hyper), 'cond': np.ascontiguousarray(bgr), 'ycrcb': np.ascontiguousarray(ycrcb)}
            return batch

    def __len__(self):
        if self.crop_size > 482:
            return len(self.hypers)
        else:
            return self.patch_per_img*self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True, one = True):
        self.hypers = []
        self.bgrs = []
        self.ycrcbs = []
        hyper_data_path = f'{data_root}/Train_spectral/'
        bgr_data_path = f'{data_root}/Train_RGB/'
        with open('dataset/split_txt/valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper_valid) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of ntire2022 dataset:{len(bgr_list)}')
        pbar = tqdm(range(len(hyper_list)))
        for i in pbar:
        # for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                print(hyper_path)
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_RGB2YCrCb)
            ycrcb = np.float32(ycrcb)
            if one:
                ycrcb = (ycrcb-ycrcb.min())/(ycrcb.max()-ycrcb.min())
            else:
                ycrcb = ycrcb / 240.0
            ycrcb = np.transpose(ycrcb, [2, 0, 1])  # [3,482,512]

            bgr = np.float32(bgr)
            if one:
                bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            else:
                bgr = bgr / 255.0
            bgr = np.transpose(bgr, [2, 0, 1])

            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            self.ycrcbs.append(np.concatenate([bgr, ycrcb], axis=0))
            mat.close()
            pbar.set_postfix_str(f'Ntire2022 scene {i} is loaded.')
            # print(f'Ntire2022 scene {i} is loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        ycrcb = self.ycrcbs[idx]
        batch = {'label':np.ascontiguousarray(hyper), 'cond': np.ascontiguousarray(bgr), 'ycrcb': np.ascontiguousarray(ycrcb)}
        return batch
        # return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)
    
if __name__ == '__main__':
    dataroot = '/work3/s212645/Spectral_Reconstruction/dataset/ARAD/'
    dataset = TrainDataset(dataroot, crop_size=512, arg=True, stride=256)
    print(len(dataset))
    print(dataset[0]['label'].shape)
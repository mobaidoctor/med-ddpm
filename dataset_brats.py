#-*- coding:utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
from glob import glob
from utils.dtypes_brats import LabelEnum
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import torch
import re
import os


class NiftiImageGenerator(Dataset):
    def __init__(self, imagefolder, input_size, depth_size, transform=None):
        self.imagefolder = imagefolder
        self.input_size = input_size
        self.depth_size = depth_size
        self.inputfiles = glob(os.path.join(imagefolder, '*.nii.gz'))
        self.scaler = MinMaxScaler()
        self.transform = transform

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # 0 -> 1 scale
        return img

    def plot_samples(self, n_slice=15, n_row=4):
        samples = [self[index] for index in np.random.randint(0, len(self), n_row*n_row)]
        for i in range(n_row):
            for j in range(n_row):
                sample = samples[n_row*i+j]
                sample = sample[0]
                plt.subplot(n_row, n_row, n_row*i+j+1)
                plt.imshow(sample[:, :, n_slice])
        plt.show()

    def __len__(self):
        return len(self.inputfiles)

    def __getitem__(self, index):
        inputfile = self.inputfiles[index]
        img = self.read_image(inputfile)
        h, w, d= img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(inputfile)
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]

        if self.transform is not None:
            img = self.transform(img)
        return img

class NiftiPairImageGenerator(Dataset):
    def __init__(self,
            input_folder: str,
            target_folder1: str,
            target_folder2: str,
            target_folder3: str,
            target_folder4: str,
            input_size: int,
            depth_size: int,
            input_channel: int = 8,
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False
        ):
        self.input_folder = input_folder
        self.target_folder1 = target_folder1
        self.target_folder2 = target_folder2
        self.target_folder3 = target_folder3
        self.target_folder4 = target_folder4
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output

    def pair_file(self):
        input_files = sorted(glob(os.path.join(self.input_folder, '*')))
        target_files1 = sorted(glob(os.path.join(self.target_folder1, '*')))
        target_files2 = sorted(glob(os.path.join(self.target_folder2, '*')))
        target_files3 = sorted(glob(os.path.join(self.target_folder3, '*')))
        target_files4 = sorted(glob(os.path.join(self.target_folder4, '*')))
        pairs = []
        for (input_file, target_file1, target_file2, target_file3, target_file4) in zip(input_files, target_files1, target_files2, target_files3, target_files4):
#             assert int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file1))) and int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file2))) and int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file3))) and int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file4)))
            pairs.append((input_file, target_file1, target_file2, target_file3, target_file4))
        return pairs

    def label2value(self, masked_img):
        result_img = masked_img.copy()
        result_img[result_img==LabelEnum.BACKGROUND.value] = 0.0
        result_img[result_img==LabelEnum.TUMORAREA1.value] = 0.25
        result_img[result_img==LabelEnum.TUMORAREA2.value] = 0.5
        result_img[result_img==LabelEnum.TUMORAREA3.value] = 0.75
        result_img[result_img==LabelEnum.BRAINAREA.value] = 1.0
        #result_img = self.scaler.fit_transform(result_img.reshape(-1, result_img.shape[-1])).reshape(result_img.shape)
        return result_img

    def label2masks(self, masked_img):
        result_img = np.zeros(masked_img.shape + (4,))   # ( (H, W, D) + (2,)  =  (H, W, D, 2)  -> (B, 2, H, W, D))
        result_img[masked_img==LabelEnum.TUMORAREA1.value, 0] = 1
        result_img[masked_img==LabelEnum.TUMORAREA2.value, 1] = 1
        result_img[masked_img==LabelEnum.TUMORAREA3.value, 2] = 1
        result_img[masked_img==LabelEnum.BRAINAREA.value, 3] = 1
        return result_img

    def combine_mask_channels(self, masks): # masks: (2, H, W, D), 2->mask channels
        result_img = np.zeros(masks.shape[1:])
        result_img += LabelEnum.TUMORAREA1.value * masks[0]
        result_img += LabelEnum.TUMORAREA2.value * masks[1]
        result_img += LabelEnum.TUMORAREA3.value * masks[2]
        result_img += LabelEnum.BRAINAREA.value * masks[3]
        return result_img

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        return img

    def plot(self, index, n_slice=30):
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[:, :, n_slice])
        plt.subplot(1, 2, 2)
        plt.imshow(target_img[:, :, n_slice])
        plt.show()

    def resize_img(self, img):
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]
        return img

    def resize_img_4d(self, input_img):
        h, w, d, c = input_img.shape
        result_img = np.zeros((self.input_size, self.input_size, self.depth_size, 2))
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            for ch in range(c):
                buff = input_img.copy()[..., ch]
                img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
                cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
                img = np.asarray(cop(img))[0]
                result_img[..., ch] += img
            return result_img
        else:
            return input_img

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file)
            input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
            input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)
            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)
                input_tensors.append(input_img)
        return torch.cat(input_tensors, 0).cuda()

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_file, target_file1, target_file2, target_file3, target_file4 = self.pair_files[index]
        input_img = self.read_image(input_file)
        input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
        input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)

        target_img1 = self.read_image(target_file1)
        target_img1 = self.resize_img(target_img1)
        
        target_img2 = self.read_image(target_file2)
        target_img2 = self.resize_img(target_img2)
        
        target_img3 = self.read_image(target_file3)
        target_img3 = self.resize_img(target_img3)
        
        target_img4 = self.read_image(target_file4)
        target_img4 = self.resize_img(target_img4)
        
        target_img = np.stack([target_img1, target_img2, target_img3, target_img4], axis=3)

        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.transform(target_img)

        if self.combine_output:
            return torch.cat([target_img, input_img], 0)

        return {'input':input_img, 'target':target_img}

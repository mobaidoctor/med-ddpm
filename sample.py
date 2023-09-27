#-*- coding:utf-8 -*-
from diffusion_model.trainer import GaussianDiffusion, num_to_groups
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from torchvision.transforms import Compose, Lambda
from utils.dtypes import LabelEnum
import nibabel as nib
import torchio as tio
import numpy as np
import argparse
import torch
import os
import glob

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfolder', type=str, default="dataset/whole_head/mask")
parser.add_argument('-e', '--exportfolder', type=str, default="exports/")
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--depth_size', type=int, default=128)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=3)
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('-w', '--weightfile', type=str, default="model/model_128.pt")
args = parser.parse_args()

exportfolder = args.exportfolder
inputfolder = args.inputfolder
input_size = args.input_size
depth_size = args.depth_size
batchsize = args.batchsize
weightfile = args.weightfile
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_samples = args.num_samples
in_channels = args.num_class_labels
out_channels = 1
device = "cuda"

mask_list = sorted(glob.glob(f"{inputfolder}/*.nii.gz"))
print(len(mask_list))


def resize_img_4d(input_img):
    h, w, d, c = input_img.shape
    result_img = np.zeros((input_size, input_size, depth_size, in_channels-1))
    if h != input_size or w != input_size or d != depth_size:
        for ch in range(c):
            buff = input_img.copy()[..., ch]
            img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
            cop = tio.Resize((input_size, input_size, depth_size))
            img = np.asarray(cop(img))[0]
            result_img[..., ch] += img
        return result_img
    else:
        return input_img

def label2masks(masked_img):
    result_img = np.zeros(masked_img.shape + (in_channels-1,))
    result_img[masked_img==LabelEnum.BRAINAREA.value, 0] = 1
    result_img[masked_img==LabelEnum.TUMORAREA.value, 1] = 1
    return result_img


input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.permute(3, 0, 1, 2)),
    Lambda(lambda t: t.unsqueeze(0)),
    Lambda(lambda t: t.transpose(4, 2))
])

model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).cuda()


diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    depth_size = depth_size,
    timesteps = args.timesteps,   # number of steps
    loss_type = 'L1', 
    with_condition=True,
).cuda()
diffusion.load_state_dict(torch.load(weightfile)['ema'])
print("Model Loaded!")

# +
img_dir = exportfolder + "/image"   
msk_dir = exportfolder + "/mask"   
os.makedirs(img_dir, exist_ok=True)
os.makedirs(msk_dir, exist_ok=True)

for k, inputfile in enumerate(mask_list):
    left = len(mask_list) - (k+1)
    print("LEFT: ", left)
    ref = nib.load(inputfile)
    msk_name = inputfile.split('/')[-1]
    refImg = ref.get_fdata()
    img = label2masks(refImg)
    img = resize_img_4d(img)
    input_tensor = input_transform(img)
    batches = num_to_groups(num_samples, batchsize)
    steps = len(batches)
    sample_count = 0
    
    print(f"All Step: {steps}")
    counter = 0
    
    for i, bsize in enumerate(batches):
        print(f"Step [{i+1}/{steps}]")
        condition_tensors, counted_samples = [], []
        for b in range(bsize):
            condition_tensors.append(input_tensor)
            counted_samples.append(sample_count)
            sample_count += 1

        condition_tensors = torch.cat(condition_tensors, 0).cuda()
        all_images_list = list(map(lambda n: diffusion.sample(batch_size=n, condition_tensors=condition_tensors), [bsize]))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = all_images.unsqueeze(1)
        all_images = all_images.transpose(5, 3)
        sampleImages = all_images.cpu()#.numpy()
        
        for b, c in enumerate(counted_samples):
            counter = counter + 1
            sampleImage = sampleImages[b][0]
            sampleImage = sampleImage.numpy()
            sampleImage=sampleImage.reshape(refImg.shape)
            nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
            nib.save(nifti_img, os.path.join(img_dir, f'{counter}_{msk_name}'))
            nib.save(ref, os.path.join(msk_dir, f'{counter}_{msk_name}'))
        torch.cuda.empty_cache()
    print("OK!")

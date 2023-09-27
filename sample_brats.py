#-*- coding:utf-8 -*-
from diffusion_model.trainer_brats import GaussianDiffusion, num_to_groups
from diffusion_model.trainer_brats import GaussianDiffusion, Trainer
from diffusion_model.unet_brats import create_model
from torchvision.transforms import Compose, Lambda
from utils.dtypes_brats import LabelEnum
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
parser.add_argument('-i', '--inputfolder', type=str, default="dataset/brats2021/seg")
parser.add_argument('-e', '--exportfolder', type=str, default=f"exports/")
parser.add_argument('--input_size', type=int, default=192)
parser.add_argument('--depth_size', type=int, default=144)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=2)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('-w', '--weightfile', type=str, default=f"model/model_brats.pt")
args = parser.parse_args()

# +
exportfolder = args.exportfolder
inputfolder = args.inputfolder
input_size = args.input_size
depth_size = args.depth_size
batchsize = args.batchsize
weightfile = args.weightfile
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_samples = args.num_samples
in_channels = 4+4
out_channels = 4
device = "cuda"

os.makedirs(f"{exportfolder}/t1", exist_ok=True)
os.makedirs(f"{exportfolder}/t1ce", exist_ok=True)
os.makedirs(f"{exportfolder}/t2", exist_ok=True)
os.makedirs(f"{exportfolder}/flair", exist_ok=True)
os.makedirs(f"{exportfolder}/seg", exist_ok=True)
# -

mask_list = sorted(glob.glob(f"{inputfolder}/*.nii.gz"))
print(len(mask_list))


def resize_img_4d(input_img):
    h, w, d, c = input_img.shape
    result_img = np.zeros((input_size, input_size, depth_size, 4))
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
    result_img = np.zeros(masked_img.shape + (4,))   # ( (H, W, D) + (2,)  =  (H, W, D, 2)  -> (B, 2, H, W, D))
    result_img[masked_img==LabelEnum.TUMORAREA1.value, 0] = 1
    result_img[masked_img==LabelEnum.TUMORAREA2.value, 1] = 1
    result_img[masked_img==LabelEnum.TUMORAREA3.value, 2] = 1
    result_img[masked_img==LabelEnum.BRAINAREA.value, 3] = 1
    return result_img


# +
def processImg(img):
    t1 = tio.ScalarImage(img)
    subject = tio.Subject(image = t1)
    transforms = tio.RescaleIntensity((0, 1)), tio.CropOrPad((240, 240, 155))  
    transform = tio.Compose(transforms)
    fixed = transform(subject)
    fixed.image.save(img)
    
def processMsk(msk):
    t1 = tio.LabelMap(msk)
    subject = tio.Subject(mask = t1)
    transform = tio.CropOrPad((240, 240, 155))   
    fixed = transform(subject)
    fixed.mask.save(msk)


# -

input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
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
    loss_type = 'l1',    # L1 or L2
    with_condition=True,
    channels=out_channels
).cuda()
weight = torch.load(weightfile, map_location='cuda')
diffusion.load_state_dict(weight['ema'])
print("Model Loaded!")

for k, inputfile in enumerate(mask_list):
    left = len(mask_list) - (k+1)
    print("LEFT IMAGES: ", left)
    img = nib.load(inputfile).get_fdata()
    img = label2masks(img)
    img = resize_img_4d(img)
    input_tensor = input_transform(img)

    batches = num_to_groups(num_samples, batchsize)
    steps = len(batches)
    sample_count = 0
    
    if not os.path.exists(exportfolder):
        os.makedirs(exportfolder)
    
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
        
        t1_images = all_images[:, 0, ...]
        t1ce_images = all_images[:, 1, ...]
        t2_images = all_images[:, 2, ...]
        flair_images = all_images[:, 3, ...]

        t1_images = t1_images.transpose(3, 1)
        t1ce_images = t1ce_images.transpose(3, 1)
        t2_images = t2_images.transpose(3, 1)
        flair_images = flair_images.transpose(3, 1)
        
        ref = nib.load(inputfile)
        name = inputfile.split("/")[-1]
        name = name.split("_seg.nii.gz")[0]
        
        for b, c in enumerate(counted_samples):
            counter = counter + 1
            sampleImage = t1_images[b, :, :, :].cpu().numpy()
            sampleImage=sampleImage.reshape([input_size, input_size, depth_size])
            nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
            nib.save(nifti_img, f"{exportfolder}/t1/{counter}_{name}_t1.nii.gz")
            processImg(f"{exportfolder}/t1/{counter}_{name}_t1.nii.gz")
            
            sampleImage = t1ce_images[b, :, :, :].cpu().numpy()
            sampleImage=sampleImage.reshape([input_size, input_size, depth_size])
            nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
            nib.save(nifti_img, f"{exportfolder}/t1ce/{counter}_{name}_t1ce.nii.gz")
            processImg(f"{exportfolder}/t1ce/{counter}_{name}_t1ce.nii.gz")
            
            sampleImage = t2_images[b, :, :, :].cpu().numpy()
            sampleImage=sampleImage.reshape([input_size, input_size, depth_size])
            nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
            nib.save(nifti_img, f"{exportfolder}/t2/{counter}_{name}_t2.nii.gz")
            processImg(f"{exportfolder}/t2/{counter}_{name}_t2.nii.gz")
            
            sampleImage = flair_images[b, :, :, :].cpu().numpy()
            sampleImage=sampleImage.reshape([input_size, input_size, depth_size])
            nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
            nib.save(nifti_img, f"{exportfolder}/flair/{counter}_{name}_flair.nii.gz")
            processImg(f"{exportfolder}/flair/{counter}_{name}_flair.nii.gz")
            
            mask = ref.get_fdata()
            mask[mask==4.] = 0.
            mask[mask==2.] = 5.
            mask[mask==1.] = 2.
            mask[mask==5.] = 1.
            nifti_img = nib.Nifti1Image(mask, affine=ref.affine)
            nib.save(nifti_img, f"{exportfolder}/seg/{counter}_{name}_seg.nii.gz")
            processMsk(f"{exportfolder}/seg/{counter}_{name}_seg.nii.gz")
            
        torch.cuda.empty_cache()
    print("OK!")



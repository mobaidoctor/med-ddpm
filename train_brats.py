#-*- coding:utf-8 -*-
# +
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer_brats import GaussianDiffusion, Trainer
from diffusion_model.unet_brats import create_model
from dataset_brats import NiftiImageGenerator, NiftiPairImageGenerator
import argparse
import torch

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# -

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--seg_folder', type=str, default="dataset/brats2021/seg/")
parser.add_argument('-t1', '--t1_folder', type=str, default="dataset/brats2021/t1/")
parser.add_argument('-t2', '--t1ce_folder', type=str, default="dataset/brats2021/t1ce/")
parser.add_argument('-t3', '--t2_folder', type=str, default="dataset/brats2021/t2/")
parser.add_argument('-t4', '--flair_folder', type=str, default="dataset/brats2021/flair/")
parser.add_argument('--input_size', type=int, default=192)
parser.add_argument('--depth_size', type=int, default=144)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=2)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100000) # epochs parameter specifies the number of training iterations
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('--save_and_sample_every', type=int, default=1000)
parser.add_argument('--with_condition', action='store_true')
parser.add_argument('-r', '--resume_weight', type=str, default="model/model_brats.pt")
args = parser.parse_args()

seg_folder = args.seg_folder
t1_folder = args.t1_folder
t1ce_folder = args.t1ce_folder
t2_folder = args.t2_folder
flair_folder = args.flair_folder
input_size = args.input_size
depth_size = args.depth_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
save_and_sample_every = args.save_and_sample_every
with_condition = args.with_condition
resume_weight = args.resume_weight

# input tensor: (B, 1, H, W, D)  value range: [-1, 1]
transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.permute(3, 0, 1, 2)),
    Lambda(lambda t: t.transpose(3, 1)),
])

input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.permute(3, 0, 1, 2)),
    Lambda(lambda t: t.transpose(3, 1)),
])

if with_condition:
    dataset = NiftiPairImageGenerator(
        seg_folder,
        t1_folder,
        t1ce_folder,
        t2_folder,
        flair_folder,
        input_size=input_size,
        depth_size=depth_size,
        transform=input_transform if with_condition else transform,
        target_transform=transform,
        full_channel_mask=True
    )
else:
    print("Please modify your code to unconditional generation")

in_channels = 4+4 if with_condition or with_pairwised else 1
out_channels = 4


model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    depth_size = depth_size,
    timesteps = args.timesteps,   # number of steps
    loss_type = 'l1',    # L1 or L2
    with_condition=with_condition,
    channels=out_channels
).cuda()

if len(resume_weight) > 0:
    weight = torch.load(resume_weight, map_location='cuda')
    diffusion.load_state_dict(weight['ema'])
    print("Model Loaded!")

trainer = Trainer(
    diffusion,
    dataset,
    image_size = input_size,
    depth_size = depth_size,
    train_batch_size = args.batchsize,
    train_lr = 1e-5,
    train_num_steps = args.epochs,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,#True,                       # turn on mixed precision training with apex
    save_and_sample_every = save_and_sample_every,
    results_folder = './results_brats',
    with_condition=with_condition
)

trainer.train()

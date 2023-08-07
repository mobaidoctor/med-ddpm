import torch
import numpy as np
import glob
from inference_utils import *

# +
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
set_cuda_params()
device = torch.device("cuda")

model = load_model("weights/3dcddpm_net.pth", device)
inputfolder = "../dataset/mask"
img_dir = "export/ddim/image"
msk_dir = "export/ddim/mask"
sampling_step = 10
os.makedirs(img_dir, exist_ok=True)
os.makedirs(msk_dir, exist_ok=True)

mask_list = sorted(glob.glob(f"{inputfolder}/*.nii.gz"))
print(len(mask_list))
# -

for mask in mask_list:
    name = mask.split('/')[-1]
    input_tensor = load_input_tensor(mask)
    diffusion = make_diffusion("./weights/3dcddpm_params.pth", 250, sampling_step)
    wrap = Wrap(model, input_tensor.to(device)).to(device)
    with torch.no_grad():
        vis = diffusion.p_sample_loop(wrap, shape, progress=True)
        sampleImage = vis.cpu().numpy()[0]
        ref = nib.load(mask)
        refImg = ref.get_fdata()
        sampleImage=sampleImage.reshape(refImg.shape)
        nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
        nib.save(nifti_img, os.path.join(img_dir, f'{name}'))
        refImg = refImg.astype(np.int8)
        refImg[refImg==1.]=0
        refImg[refImg==2.]=1
        nifti_img = nib.Nifti1Image(refImg, affine=ref.affine)
        nib.save(nifti_img, os.path.join(msk_dir, f'{name}'))

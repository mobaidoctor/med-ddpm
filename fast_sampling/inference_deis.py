import torch
import numpy as np
import glob
from inference_utils import *
from th_deis import DisVPSDE, get_sampler
from guided_diffusion.gaussian_diffusion import _extract_into_tensor

# +
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
set_cuda_params()
device = torch.device("cuda")

model = load_model("weights/3dcddpm_net.pth", device)
inputfolder = "../dataset/mask"
img_dir = "export/ipndm/image"
msk_dir = "export/ipndm/mask"
sampling_step = 10
os.makedirs(img_dir, exist_ok=True)
os.makedirs(msk_dir, exist_ok=True)

mask_list = sorted(glob.glob(f"{inputfolder}/*.nii.gz"))
mask_idx = 0
inputfile = mask_list[mask_idx]
input_tensor = load_input_tensor(inputfile)
diffusion_full = make_diffusion("weights/3dcddpm_params.pth", 250, sampling_step)
wrap = Wrap(model, input_tensor.to(device)).to(device)


# -

def predict_xstart_from_eps(self, x_t, t, eps):
    assert x_t.shape == eps.shape
    return (
        _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
        - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
    )

def predict_eps(self, eps, x_t, t):
    x_start = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
    x_start = x_start.clamp_(-1., 1.)
    eps_new =  (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start)/_extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    return eps_new

def eps_fn(x, scalar_t):
    vec_t = (torch.ones(x.shape[0])).float().to(x) * scalar_t
    with torch.no_grad():
        r = wrap(x, vec_t)
        r = predict_eps(diffusion_full, r, x, scalar_t)
        return r

num_step = 10
vpsde = DisVPSDE(diffusion_full.alphas_cumprod)

sampler_fn = get_sampler(
    vpsde,
    num_step,
    eps_fn,
    order=3,
    method="ipndm",
)

noise = torch.randn(shape).cuda()
vis_deis = sampler_fn(noise)
sampleImage = vis_deis.cpu().numpy()[0]
ref = nib.load(inputfile)
refImg = ref.get_fdata()
name = inputfile.split('/')[-1]
sampleImage=sampleImage.reshape(refImg.shape)
nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
nib.save(nifti_img, os.path.join(img_dir, f'{name}'))
refImg = refImg.astype(np.int8)
refImg[refImg==1.]=0
refImg[refImg==2.]=1
nifti_img = nib.Nifti1Image(refImg, affine=ref.affine)
nib.save(nifti_img, os.path.join(msk_dir, f'{name}'))
print("Done!")
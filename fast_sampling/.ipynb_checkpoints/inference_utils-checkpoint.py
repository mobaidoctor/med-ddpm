import os
import torch
import numpy as np
from guided_diffusion.unet_3dcddpm import UNetModel
import nibabel as nib
import torchio as tio
from torchvision.transforms import Compose, Lambda
from enum import IntEnum, Enum
from torch import nn
# import cv2

input_size = 128
depth_size = 128
num_channels = 32
num_res_blocks = 1
in_channels = 3
out_channels = 1
shape = [1, depth_size, input_size, input_size]


def set_cuda_params():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:51"

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="14",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    in_channels=1,
    out_channels=1,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    NUM_CLASSES = 1

    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(1*out_channels if not learn_sigma else 2*out_channels),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )

def load_model(weights_file, device):

    model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).to(device)
    model.load_state_dict(torch.load(weights_file, map_location=device)["model"])
    return model

def load_input_tensor(inputfile):

    class LabelEnum(IntEnum):
        BACKGROUND = 0
        TUMORAREA = 2
        BRAINAREA = 1

    def resize_img_4d(input_img):
        h, w, d, c = input_img.shape
        result_img = np.zeros((input_size, input_size, depth_size, 2))
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
        result_img = np.zeros(masked_img.shape + (2,))  # ( (H, W, D) + (2,)  =  (H, W, D, 2)  -> (B, 2, H, W, D))
        result_img[masked_img == LabelEnum.TUMORAREA.value, 0] = 1
        result_img[masked_img == LabelEnum.BRAINAREA.value, 1] = 1
        return result_img

    input_transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: (t * 2) - 1),
        Lambda(lambda t: t.permute(3, 0, 1, 2)),
        Lambda(lambda t: t.unsqueeze(0)),
    ])

    img = nib.load(inputfile).get_fdata()
    img = label2masks(img)
    img = resize_img_4d(img)
    input_tensor = input_transform(img)
    return input_tensor

def make_diffusion(params_file, steps, timestep_respacing):
    from guided_diffusion.respace import SpacedDiffusion, space_timesteps
    import guided_diffusion.gaussian_diffusion as gd

    diffusion_betas = torch.load(params_file, map_location=torch.device("cpu"))["betas"]

    loss_type = gd.LossType.RESCALED_MSE

    betas = diffusion_betas.numpy()
    rescale_timesteps = True

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(steps, str(timestep_respacing)),
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.RESCALED_MSE,
        rescale_timesteps=rescale_timesteps,
    )
    return diffusion

class Wrap(nn.Module):

    def __init__(self, net, cond):
        super().__init__()
        self.net = net
        self.condition = cond

    def forward(self, x, t):

        with torch.no_grad():
            x = x.unsqueeze(1)
            x = torch.cat([x, self.condition], 1)
            x = self.net(x, t)
            x = x.squeeze(1)
        return x

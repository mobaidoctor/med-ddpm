#-*- coding:utf-8 -*-
# This code was taken from: https://github.com/assafshocher/resizer by Assaf Shocher (edited by Sodoo)
from .dtypes import FilterMethods
from math import pi
import numpy as np
import torch.nn as nn
import torch

def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) +
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))

def lanczos2(x):
    return (((np.sin(pi * x) * np.sin(pi * x / 2) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 2) + np.finfo(np.float32).eps))
            * (abs(x) < 2))

def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0

def lanczos3(x):
    return (((np.sin(pi * x) * np.sin(pi * x / 3) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 3) + np.finfo(np.float32).eps))
            * (abs(x) < 3))

def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))


class LowPassFilter(nn.Module):
    def __init__(self,
            in_shape: tuple,     # (H, W, D)
            output_shape: tuple, # (H, W, D)
            scale_factor: int = 4,
            method: FilterMethods = FilterMethods.CUBIC
        ):
        super().__init__()

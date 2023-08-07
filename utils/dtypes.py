#-*- coding:utf-8 -*-
from enum import IntEnum, Enum

class LabelEnum(IntEnum):
    BACKGROUND = 0
    TUMORAREA = 2
    BRAINAREA = 1

class FilterMethods(Enum):
    CUBIC = "CUBIC"
    LANCZOS2 = "LANCZOS2"
    LANCZOS3 = "LANCZOS3"
    BOX = "BOX"
    LINEAR = "LINEAR"

from enum import Enum

class MaskCorruption(Enum):
    # 4 types: misposition, shift, dilation and shrinkage
    MISPOSITION = 0
    SHIFT = 1
    DILATION = 2
    SHRINK = 3
import os
import gc
import math
import time
import random
import requests
import itertools
import logging
# import wandb
from datetime import datetime


import torch
from torch import nn
from torch.nn import functional
# from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
ToPILImage = transforms.ToPILImage()

# from torchsummary import summary
import digitake

from tqdm import tqdm
#from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
# import cv2

from src.wsdan import WSDAN


import numpy as np

EPSILON = 1e-12

"""## 3.1 Create Custom Layers & Modules

### 3.1.1 Create Bilinear Attention Pooling Layer
"""




if __name__ == '__main__':

    print("@@ torch.__version__:", torch.__version__)

    #

    """## 2.2 Global config for training environment and reproducibility"""

    USE_GPU = False#True
    digitake.model.set_reproducible(2565)

    if USE_GPU:
        # GPU settings
        assert torch.cuda.is_available(), "Don't forget to turn on gpu runtime!"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print("@@ device:", device)

    #

    # pretrain = 'resnet' #@param ["resnet", "densenet", "inception", "vgg"]
    pretrain = 'densenet' #@param ["resnet", "densenet", "inception", "vgg"]

    num_classes = 2
    num_attention_maps = 32


    print('\n\n@@ ======== Calling `net = WSDAN(...)`')
    net = WSDAN(num_classes=num_classes, M=num_attention_maps, net=pretrain, pretrained=True)




    print('\n\n@@ ======== done')

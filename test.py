import torch
print('@@ torch.__version__:', torch.__version__)

from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image

from tqdm import tqdm
#from tqdm.notebook import tqdm

import logging
logging.basicConfig(level=logging.INFO)

import matplotlib
print('@@ matplotlib.__version__:', matplotlib.__version__)

import os
import random
import time
from glob import glob


BASE_DIR = '.'
malignant_paths = glob(os.path.join(BASE_DIR,'Dataset_train_test_val/Train/Malignant', '*.png'))
benign_paths = glob(os.path.join(BASE_DIR, 'Dataset_train_test_val/Train/Benign', '*.png'))

val_malignant_paths = glob(os.path.join(BASE_DIR,'Dataset_train_test_val/Val/Malignant', '*.png'))
val_benign_paths = glob(os.path.join(BASE_DIR, 'Dataset_train_test_val/Val/Benign', '*.png'))

test_malignant_paths = glob(os.path.join(BASE_DIR,'Dataset_train_test_val/Test/Malignant', '*.png'))
test_benign_paths = glob(os.path.join(BASE_DIR, 'Dataset_train_test_val/Test/Benign', '*.png'))

print('@@ len(malignant_paths):', len(malignant_paths))
print('@@ len(benign_paths):', len(benign_paths))
print('@@ len(val_malignant_paths):', len(val_malignant_paths))
print('@@ len(val_benign_paths):', len(val_benign_paths))
print('@@ len(test_malignant_paths):', len(test_malignant_paths))
print('@@ len(test_benign_paths):', len(test_benign_paths))

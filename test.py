import os
import gc
import math
import time
import random
import requests
import itertools

import logging
logging.basicConfig(level=logging.INFO)

# import wandb
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
ToPILImage = transforms.ToPILImage()

# from torchsummary import summary
import digitake

from tqdm import tqdm
#from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
# import cv2

import numpy as np

from src.wsdan import WSDAN
from src.metric import TopKAccuracyMetric
from src.transform import ThyroidDataset, get_transform, get_transform_center_crop, transform_fn
from src.augment import batch_augment


#@@ def test(**kwargs):
def test(net, data_loader, visualize, ckpt=None):  # @@
  #@@ data_loader = kwargs['data_loader']
  #@@ visualize = kwargs['visualize']
  global name

  if visualize:  # @@
      savepath = f"classifier/result_{name}/"
      if not os.path.exists(savepath):
        os.mkdir(savepath)

  # Load ckpt and get state_dict
  #@@ if kwargs['ckpt']:
  #@@   ckpt = kwargs['ckpt']
  if ckpt is not None:  # @@
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)
    logging.info('Network loaded from {}'.format(ckpt))

  ToPILImage = transforms.ToPILImage()

  raw_accuracy = TopKAccuracyMetric()
  ref_accuracy = TopKAccuracyMetric()
  raw_accuracy.reset()

  results = []
  net.eval()

  with torch.no_grad():

      pbar = tqdm(total=len(data_loader), unit=' batches')
      pbar.set_description('Test data')

      for i, (X, y, p) in enumerate(data_loader):
          # obtain data for testing
          X = X.to(device)
          y = y.to(device)

          mean = X.mean((0,2,3)).view(1, 3, 1, 1)
          std = X.std((0,2,3)).view(1, 3, 1, 1)

          MEAN = mean.cpu()     # torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
          STD = std.cpu()           # torch.tensor([0.1, 0.1, 0.1]).view(1, 3, 1, 1)
          # MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
          # STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

          ##################################
          # Raw Image
          ##################################
          y_pred_raw, _, attention_maps = net(X)

          ##################################
          # Attention Cropping
          ##################################
          crop_image = batch_augment(X, attention_maps, mode='crop', theta=0.85, padding_ratio=0.05)

          # crop images forward
          y_pred_crop, _, _ = net(crop_image)
          importance =  torch.abs(y_pred_raw[0] - y_pred_raw[1])

          y_pred = (y_pred_raw + (y_pred_crop * 2 * importance)) / 3.

          if visualize:
              channel = 3

              # reshape attention maps
              print(f"Input Shape:{X.shape} vs Attention Shape: {attention_maps.shape}")
              A = attention_maps.expand(-1, 4, 8, 8) if channel == 4 else attention_maps
              print(f"New Attention: {A.shape}, size={(X.size(2), X.size(3))}")
              attention_maps = functional.interpolate(A, size=(X.size(2), X.size(3)))
              attention_maps = attention_maps.cpu() / attention_maps.max().item()

              # get heat attention maps
              heat_threshold = 0.5
              heat_attention_maps = generate_heatmap(attention_maps, threshold=heat_threshold)

              # raw_image, heat_attention, raw_attention
              raw_image = X.cpu() * STD + MEAN

              print(f"X:{raw_image.shape} vs HEAT Shape: {heat_attention_maps.shape}")
              #H = heat_attention_maps.repeat(1, 4, 1, 1)
              grays_dim = heat_attention_maps.shape[:]
              grays_dim = [grays_dim[0], 1, grays_dim[2], grays_dim[3]]
              ones = torch.ones(grays_dim)

              H = torch.hstack([heat_attention_maps, ones]) if channel == 4 else heat_attention_maps
              heat_attention_image = raw_image * 0.5 + H *0.5

              raw_attention_image = raw_image * attention_maps

              for batch_idx in range(X.size(0)):
                  rimg = ToPILImage(raw_image[batch_idx])
                  raimg = ToPILImage(raw_attention_image[batch_idx])
                  haimg = ToPILImage(heat_attention_image[batch_idx])
                  rimg.save(os.path.join(savepath, '%03d_raw.png' % (i * batch_size + batch_idx)))
                  raimg.save(os.path.join(savepath, '%03d_raw_atten.png' % (i * batch_size + batch_idx)))
                  haimg.save(os.path.join(savepath, '%03d_heat_atten.png' % (i * batch_size + batch_idx)))

          results = (X, crop_image, y_pred, y, p, heat_attention_image, y_pred_crop)

          # Top K
          epoch_raw_acc = raw_accuracy(y_pred_raw, y)
          epoch_ref_acc = ref_accuracy(y_pred, y)

          # end of this batch
          batch_info = 'Val Acc: Raw ({:.2f}), Refine ({:.2f})'.format(
              epoch_raw_acc[0], epoch_ref_acc[0])
          pbar.update()
          pbar.set_postfix_str(batch_info)
          torch.cuda.empty_cache()

      pbar.close()
  return results



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

    test_ds_path_no = digitake.preprocess.build_dataset({
      'malignant': ['Test/Malignant'],
      'benign': ['Test/Benign'],
    }, root='Dataset_train_test_val')

    print('@@ test_ds_path_no:', test_ds_path_no)
    print("@@ len(test_ds_path_no['malignant']):", len(test_ds_path_no['malignant']))
    print("@@ len(test_ds_path_no['benign']):", len(test_ds_path_no['benign']))

    #

    # pretrain = 'resnet' #@param ["resnet", "densenet", "inception", "vgg"]
    pretrain = 'densenet' #@param ["resnet", "densenet", "inception", "vgg"]

    target_resize = 250
    batch_size = 8 #@param ["8", "16", "4", "1"] {type:"raw"}

    num_classes = 2
    num_attention_maps = 32

    #@@workers = 2
    workers = 0  # @@

    #

    # No Markers
    test_dataset_no = ThyroidDataset(
        phase='test',
        dataset=test_ds_path_no,
        transform=get_transform(target_resize, phase='basic'),
        with_alpha_channel=False)

    test_loader_no = DataLoader(
        test_dataset_no,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=workers,
        pin_memory=True)

    #

    print('\n\n@@ ======== Calling `net = WSDAN(...)`')
    net = WSDAN(num_classes=num_classes, M=num_attention_maps, net=pretrain, pretrained=True)

    net.to(device)

    #

    print('\n\n@@ ======== Calling `test()`')
    visualize = False
    ckpt = "densenet_224_16_lr-1e5_n5_220905-1309_78.571.ckpt"
    results = test(net, test_loader_no, visualize, ckpt=ckpt)
    print('@@ results:', results)

    #

    print('\n\n@@ ======== print_scores(results)')
    print_scores(results)

    _enable_plot = 0  # @@
    print(f'\n\n@@ ======== print_auc(results, enable_plot={_enable_plot})')
    print_auc(results, len(test_dataset_no), enable_plot=_enable_plot)

    #

    print('\n\n@@ ======== done')

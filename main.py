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
# from torch import nn
# from torch.nn import functional
# from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

# from torchsummary import summary
import digitake


import matplotlib.pyplot as plt
# import cv2

import numpy as np

from src.wsdan import WSDAN
from src.transform import ThyroidDataset, get_transform##, get_transform_center_crop, transform_fn
# from src.train import train        # "Training"
# from src.validate import validate  # "Validation"


def get_device():
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


def demo_test():
    print('@@ demo_test(): ^^')
    from src.test import test  # "Prediction"

    device = get_device()
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

    # name = "xx"
    # savepath = f"classifier/result_{name}/"
    # if not os.path.exists(savepath):
    #     os.mkdir(savepath)

    ckpt = "WSDAN_densenet_224_16_lr-1e5_n1-remove_220828-0837_85.714.ckpt"
    #ckpt = "WSDAN_doppler_densenet_224_16_lr-1e5_n5_220905-1309_78.571.ckpt"

    results = test(device, net, batch_size, test_loader_no, ckpt, savepath='./result')

    print('@@ results:', results)

    #

    if 1:  #  legacy
        from src.legacy import print_scores, print_auc

        print('\n\n@@ ======== print_scores(results)')
        print_scores(results)

        _enable_plot = 0  # @@
        print(f'\n\n@@ ======== print_auc(results, enable_plot={_enable_plot})')
        print_auc(results, len(test_dataset_no), enable_plot=_enable_plot)

    #

    print('@@ demo_test(): vv')


def demo_doppler_comp():
    print('@@ demo_doppler_comp(): ^^')
    from src.doppler import doppler_comp, get_iou, plot_comp

    path_doppler = './Siriraj/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0011_1_p0022.png'
    path_markers = './Siriraj/Markers_Train/Benign/benign_nodule1_0001-0100_c0011_2_p0022.png'
    path_markers_label = './Siriraj/Markers_Train_Markers_Labels/Benign/benign_nodule1_0001-0100_c0011_2_p0022.txt'

    bbox_doppler, bbox_markers, border_img_doppler, border_img_markers = doppler_comp(
        path_doppler, path_markers, path_markers_label)
    print('@@ bbox_doppler:', bbox_doppler)
    print('@@ bbox_markers:', bbox_markers)

    iou = get_iou(bbox_doppler, bbox_markers)
    print('@@ iou:', iou)

    if 0:
        plot_comp(border_img_doppler, border_img_markers, path_doppler, path_markers)

#           ./Siriraj/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0076_1_p0152.png
#                ./Siriraj/Markers_Train/Benign/benign_nodule1_0001-0100_c0076_2_p0152.png
# ./Siriraj/Markers_Train_Markers_Labels/Benign/benign_nodule1_0001-0100_c0076_2_p0152.txt
#
#           ./Siriraj/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0022_1_p0044.png
#                ./Siriraj/Markers_Train/Benign/benign_nodule1_0001-0100_c0022_2_p0044.png
# ./Siriraj/Markers_Train_Markers_Labels/Benign/benign_nodule1_0001-0100_c0022_2_p0044.txt
#
#           ./Siriraj/Doppler_Train_Crop/Benign/benign_nodule3_0001-0030_c0024_2_p0071.png
#                ./Siriraj/Markers_Train/Benign/benign_nodule3_0001-0030_c0024_1_p0071.png
# ./Siriraj/Markers_Train_Markers_Labels/Benign/benign_nodule3_0001-0030_c0024_1_p0071.txt
#
#           ./Siriraj/Doppler_Train_Crop/Benign/benign_nodule2_0001-0016_c0001_3_p0002.jpg
#                ./Siriraj/Markers_Train/Benign/benign_nodule2_0001-0016_c0001_1_p0002.png
# ./Siriraj/Markers_Train_Markers_Labels/Benign/benign_nodule2_0001-0016_c0001_1_p0002.txt
#
#           ./Siriraj/Doppler_Train_Crop/Benign/benign_siriraj_0001-0160_c0128_2_p0089.png
#                ./Siriraj/Markers_Train/Benign/benign_siriraj_0001-0160_c0128_1_p0088.png
# ./Siriraj/Markers_Train_Markers_Labels/Benign/benign_siriraj_0001-0160_c0128_1_p0088.txt
#
#           ./Siriraj/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0008_1_p0016.png
#                ./Siriraj/Markers_Train/Benign/benign_nodule1_0001-0100_c0008_2_p0016.png
# ./Siriraj/Markers_Train_Markers_Labels/Benign/benign_nodule1_0001-0100_c0008_2_p0016.txt
#
#           ./Siriraj/Doppler_Train_Crop/Malignant/malignant_siriraj_0001-0124_c0110_3_p0257.png
#                ./Siriraj/Markers_Train/Malignant/malignant_siriraj_0001-0124_c0110_2_p0256.png
# ./Siriraj/Markers_Train_Markers_Labels/Malignant/malignant_siriraj_0001-0124_c0110_2_p0256.txt
#
#           ./Siriraj/Doppler_Train_Crop/Malignant/malignant_nodule3_0001-0030_c0004_3_p0011.png
#                ./Siriraj/Markers_Train/Malignant/malignant_nodule3_0001-0030_c0004_1_p0011.png
# ./Siriraj/Markers_Train_Markers_Labels/Malignant/malignant_nodule3_0001-0030_c0004_1_p0011.txt

    print('@@ demo_doppler_comp(): vv')


if __name__ == '__main__':
    print("@@ torch.__version__:", torch.__version__)

    if 0:  # the "Prediction" flow of 'WSDAN_Pytorch_Revised_v1_01_a.ipynb'
        demo_test()

    if 0:  # the "Traning/Validation" flow of 'WSDAN_Pytorch_Revised_v1_01_a.ipynb'
        demo_train()  # TBA

    if 1:  # adaptation of 'compare.{ipynb,py}' exported from https://colab.research.google.com/drive/1kxMFgo1LyVqPYqhS6_UJKUsVvA2-l9wk
        demo_doppler_comp()
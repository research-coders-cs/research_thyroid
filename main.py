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


ARTIFACTS_OUTPUT = './output'

def mk_artifact_dir(dirname):
    path = f'{ARTIFACTS_OUTPUT}/{dirname}'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

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

    return device

def show_data_loader(data_loader, plt_show=False):
    x = enumerate(data_loader)

    try:
        i, v = next(x)

        shape = v[0].shape
        batch_size = shape[0]
        channel = shape[1]
        w = shape[2]
        h = shape[3]

        print(shape)
        print(f"X contains {batch_size} images with {channel}-channels of size {w}x{h}")
        print(f"y is a {type(v[1]).__name__} of", v[1].tolist())
        print()
        for k in v[2]:
            print(f"{k}=", v[2][k])

    except StopIteration:
        print('StopIteration')

    return channel, batch_size, w, h

def demo_thyroid_test():
    print('\n\n\n\n@@ demo_thyroid_test(): ^^')
    from src.thyroid_test import test  # "Prediction"

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

    # pretrain = 'resnet'
    pretrain = 'densenet'

    target_resize = 250
    batch_size = 8 #@param ["8", "16", "4", "1"] {type:"raw"}

    num_classes = 2
    num_attention_maps = 32

    #@@workers = 2
    workers = 0  # @@
    print('@@ workers:', workers)

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

    net = WSDAN(num_classes=num_classes, M=num_attention_maps, net=pretrain, pretrained=True)
    net.to(device)

    #

    #ckpt = "WSDAN_densenet_224_16_lr-1e5_n1-remove_220828-0837_85.714.ckpt"
    #ckpt = "WSDAN_doppler_densenet_224_16_lr-1e5_n5_220905-1309_78.571.ckpt"

    #ckpt = "./out--train--workers_0/densenet_250_8_lr-1e5_n4_55.000"
    #ckpt = "./out--train--workers_2/densenet_250_8_lr-1e5_n4_75.000"
    #!!!! todo use the model with best score !!!!

    results = test(device, net, batch_size, test_loader_no, ckpt,
                   savepath=mk_artifact_dir('demo_thyroid_test'))
    # print('@@ results:', results)

    #

    if 1:  #  legacy
        from src.legacy import print_scores, print_auc

        print('\n\n@@ ======== print_scores(results)')
        print_scores(results)

        _enable_plot = 0  # @@
        print(f'\n\n@@ ======== print_auc(results, enable_plot={_enable_plot})')
        print_auc(results, len(test_dataset_no), enable_plot=_enable_plot)

    #

    print('@@ demo_thyroid_test(): vv')


def demo_thyroid_train():
    print('\n\n\n\n@@ demo_thyroid_train(): ^^')
    from src.thyroid_train import training

    device = get_device()
    print("@@ device:", device)

    #

    train_ds_path = digitake.preprocess.build_dataset({
      'malignant': ['Train/Malignant'],
      'benign': ['Train/Benign'],
    }, root='Dataset_train_test_val')
    #print(train_ds_path)
    print(len(train_ds_path['malignant']), len(train_ds_path['benign']))  # @@ 20 21

    val_ds_path = digitake.preprocess.build_dataset({
      'malignant': ['Val/Malignant'],
      'benign': ['Val/Benign'],
    }, root='Dataset_train_test_val')
    #print(val_ds_path)
    print(len(val_ds_path['malignant']), len(val_ds_path['benign']))  # @@ 10 10

    # @@ to be used later
    dropper_train_ds_path = digitake.preprocess.build_dataset({
      'malignant': ['Doppler_Train_Crop/Malignant'],
      'benign': ['Doppler_Train_Crop/Benign'],
    }, root='Siriraj_sample_doppler_comp')
    print("@@ dropper_train_ds_path['malignant']", dropper_train_ds_path['malignant'])

    #

    # pretrain = 'resnet'
    pretrain = 'densenet'

    target_resize = 250
    batch_size = 8 #@param ["8", "16", "4", "1"] {type:"raw"}

    number = 4 #@param ["1", "2", "3", "4", "5"] {type:"raw", allow-input: true}

    num_classes = 2
    num_attention_maps = 32

    workers = 2
    print('@@ workers:', workers)

    lr = 0.001 #@param ["0.001", "0.00001"] {type:"raw"}
    lr_ = "lr-1e5" #@param ["lr-1e3", "lr-1e5"]

    start_epoch = 0
    total_epochs = 5

    run_name = f"{pretrain}_{target_resize}_{batch_size}_{lr_}_n{number}"
    print('@@ run_name:', run_name)

    #

    train_dataset = ThyroidDataset(
        phase='train',
        dataset=train_ds_path,
        transform=get_transform(target_resize, phase='basic'),
        with_alpha_channel=False  # if False, it will load image as RGB(3-channel)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    channel, _, _, _ = show_data_loader(train_loader)
    print('@@ channel:', channel)

    #

    validate_dataset = ThyroidDataset(
        phase='val',
        dataset=val_ds_path,
        transform=get_transform(target_resize, phase='basic'),
        with_alpha_channel=False
      )

    validate_loader = DataLoader(
        validate_dataset,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )

    #

    net = WSDAN(num_classes=num_classes, M=num_attention_maps, net=pretrain, pretrained=True)
    net.to(device)
    feature_center = torch.zeros(num_classes, num_attention_maps * net.num_features).to(device)

    #

    logs = {
        'epoch': 0,
        'train/loss': float("Inf"),
        'val/loss': float("Inf"),
        'train/raw_topk_accuracy': 0.,
        'train/crop_topk_accuracy': 0.,
        'train/drop_topk_accuracy': 0.,
        'val/topk_accuracy': 0.
    }

    learning_rate = logs['lr'] if 'lr' in logs else lr

    opt_type = 'SGD'
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.99)

    if 0:  # @@
        wandb.init(
            # Set the project where this run will be logged
            # project=f"Wsdan_Thyroid_{total_epochs}epochs_RecheckRemove_Upsampling_v2",
            project=f"Wsdan_Thyroid",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=run_name,
            # Track hyperparameters and run metadata
            config={
            "learning_rate": learning_rate,
            "architecture": f"WS-DAN-{pretrain}",
            "optimizer": opt_type,
            "dataset": "Thyroid",
            "train-data-augment": f"{channel}-channel",
            "epochs": f"{total_epochs - start_epoch}({start_epoch}->{total_epochs})" ,
        })

    #

    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'
        .format(total_epochs, batch_size, len(train_dataset), len(validate_dataset)))

    training(device, net, feature_center, batch_size, train_loader, validate_loader,
             optimizer, scheduler, run_name, logs, start_epoch, total_epochs,
             savepath=mk_artifact_dir('demo_thyroid_train'))

    #

    print('@@ demo_thyroid_train(): vv')


def demo_doppler_comp():
    print('\n\n\n\n@@ demo_doppler_comp(): ^^')

    from src.doppler import doppler_comp, get_iou, plot_comp, get_sample_paths
    savepath = mk_artifact_dir('demo_doppler_comp')

    for path_doppler, path_markers, path_markers_label in get_sample_paths():
        print('\n@@ -------- calling doppler_comp() for')
        print(f'  {os.path.basename(path_doppler)} vs')
        print(f'  {os.path.basename(path_markers)}')

        bbox_doppler, bbox_markers, border_img_doppler, border_img_markers = doppler_comp(
            path_doppler, path_markers, path_markers_label)
        print('@@ bbox_doppler:', bbox_doppler)
        print('@@ bbox_markers:', bbox_markers)

        iou = get_iou(bbox_doppler, bbox_markers)
        print('@@ iou:', iou)

        plt = plot_comp(border_img_doppler, border_img_markers, path_doppler, path_markers)
        stem = os.path.splitext(os.path.basename(path_doppler))[0]
        fname = f'{savepath}/comp-doppler-{stem}.jpg'
        plt.savefig(fname, bbox_inches='tight')
        print('@@ saved -', fname)

    print('@@ demo_doppler_comp(): vv')


if __name__ == '__main__':
    print("@@ torch.__version__:", torch.__version__)

    if 0:  # adaptation of 'compare.{ipynb,py}' exported from https://colab.research.google.com/drive/1kxMFgo1LyVqPYqhS6_UJKUsVvA2-l9wk
        demo_doppler_comp()  # TODO - renaming

    if 1:  # the "Traning/Validation" flow of 'WSDAN_Pytorch_Revised_v1_01_a.ipynb'
        demo_thyroid_train()

    if 0:  # the "Prediction" flow of 'WSDAN_Pytorch_Revised_v1_01_a.ipynb' - https://colab.research.google.com/drive/1LN4KjBwtq6hUG42LtSLCmIVPasehKeKq
        demo_thyroid_test()  # TODO - generate 'confusion_matrix_test-*.png', 'test-*.png'

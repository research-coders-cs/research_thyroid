import torch
from torch.utils.data import DataLoader

from digitake.preprocess import build_dataset

from .transform import ThyroidDataset, get_transform##, get_transform_center_crop, transform_fn
from .utils import mk_artifact_dir, get_device


WSDAN_NUM_CLASSES = 2

TRAIN_DS_PATH_DEFAULT = build_dataset({
    'benign': ['Train/Benign'],
    'malignant': ['Train/Malignant'],
}, root='Dataset_train_test_val')  # 21 20

VALIDATE_DS_PATH_DEFAULT = build_dataset({
    'benign': ['Val/Benign'],
    'malignant': ['Val/Malignant'],
}, root='Dataset_train_test_val')  # 10 10

TEST_DS_PATH_DEFAULT = build_dataset({
    'benign': ['Test/Benign'],
    'malignant': ['Test/Malignant'],
}, root='Dataset_train_test_val')  # 10 10

MODEL_DEFAULT = 'densenet121'


def doppler_compare():
    import os
    from ..net.doppler import doppler_comp, get_iou, plot_comp, get_sample_paths
    import matplotlib.pyplot as plt
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

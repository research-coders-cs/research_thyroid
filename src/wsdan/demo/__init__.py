import torch
from torch.utils.data import DataLoader

from digitake.preprocess import build_dataset

from ..net import WSDAN, net_train, net_test

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


def train():
    pass


def train_with_doppler():
    pass


def test(ckpt, model=MODEL_DEFAULT, ds_path=TEST_DS_PATH_DEFAULT,
        target_resize=250, batch_size=8, num_attention_maps=32):
    from .utils import show_data_loader
    from .stats import print_scores, print_auc, print_poa

    print('\n\n@@ demo_thyroid_test(): ^^')
    print("@@ model:", model)
    print("@@ target_resize:", target_resize)
    print("@@ batch_size:", batch_size)

    device = get_device()
    print("@@ device:", device)

    #print('@@ ds_path:', ds_path)
    print("@@ lens ds_path:", len(ds_path['benign']), len(ds_path['malignant']))

    test_dataset = ThyroidDataset(
        phase='test',
        dataset=ds_path,
        transform=get_transform(target_resize, phase='basic'),
        with_alpha_channel=False)

    #@@workers = 2
    workers = 0  # @@
    print('@@ workers:', workers)

    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),  # @@
        shuffle=False,
        num_workers=workers,
        pin_memory=True)

    #

    net = WSDAN(num_classes=WSDAN_NUM_CLASSES, M=num_attention_maps, model=model, pretrained=True)
    net.to(device)

    results = net_test.test(device, net, batch_size, test_loader, ckpt,
        savepath=mk_artifact_dir('demo_thyroid_test'))
    # print('@@ results:', results)

    if 1:
        print('\n\n@@ ======== print_scores(results)')
        print_scores(results)

    if 0:
        _enable_plot = 0  # @@
        print(f'\n\n@@ ======== print_auc(results, enable_plot={_enable_plot})')
        print_auc(results, len(test_dataset), enable_plot=_enable_plot)

    if 1:
        print(f'\n\n@@ ======== print_poa(results)')
        print_poa(results)

import torch
from torch.utils.data import DataLoader

from digitake.preprocess import build_dataset
from src.wsdan import WSDAN
from src.transform import ThyroidDataset, get_transform##, get_transform_center_crop, transform_fn
from src.utils import mk_artifact_dir, get_device, show_data_loader
from src.stats import print_scores, print_auc, print_poa
from src import thyroid_train, thyroid_test

import os
import logging
logging.basicConfig(level=logging.INFO)


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


def demo_thyroid_test(ckpt, model=MODEL_DEFAULT, ds_path=TEST_DS_PATH_DEFAULT,
        target_resize=250, batch_size=8, num_attention_maps=32):

    print('\n\n@@ demo_thyroid_test(): ^^')
    print("@@ model:", model)
    print("@@ target_resize:", target_resize)
    print("@@ batch_size:", batch_size)

    device = get_device()
    print("@@ device:", device)

    #

    print('@@ ds_path:', ds_path)
    print("@@ len(ds_path['malignant']):", len(ds_path['malignant']))
    print("@@ len(ds_path['benign']):", len(ds_path['benign']))

    #

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

    results = thyroid_test.test(device, net, batch_size, test_loader, ckpt,
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


def _demo_thyroid_train(with_doppler, model, train_ds_path, validate_ds_path, savepath):
    print('\n\n@@ _demo_thyroid_train(): ^^')

    device = get_device()
    print("@@ device:", device)

    #

    print('@@ with_doppler:', with_doppler)
    print('@@ model:', model)
    print('@@ savepath:', savepath)

    #print('@@ train_ds_path:', train_ds_path)
    print(len(train_ds_path['benign']), len(train_ds_path['malignant']))

    print('@@ validate_ds_path:', validate_ds_path)
    print(len(validate_ds_path['benign']), len(validate_ds_path['malignant']))

    #

    target_resize = 250
    batch_size = 8 #@param ["8", "16", "4", "1"] {type:"raw"}

    number = 4 #@param ["1", "2", "3", "4", "5"] {type:"raw", allow-input: true}

    workers = 2
    print('@@ workers:', workers)

    lr = 0.001 #@param ["0.001", "0.00001"] {type:"raw"}
    lr_ = "lr-1e5" #@param ["lr-1e3", "lr-1e5"]

    #total_epochs = 1
    #total_epochs = 2
    total_epochs = 40

    run_name = f"{model}_{target_resize}_{batch_size}_{lr_}_n{number}"
    print('@@ run_name:', run_name)

    #

    from src.doppler import to_doppler  # !!!!

    train_dataset = ThyroidDataset(
        phase='train',
        dataset=train_ds_path,
        transform=get_transform(target_resize, phase='basic'),
    #==== @@ orig
        with_alpha_channel=False  # if False, it will load image as RGB(3-channel)
    #==== @@ WIP w.r.t. 'digitake/src/digitake/preprocess/thyroid.py'
        # mask_dict=to_doppler if with_doppler else None,  # !!!!
        # with_alpha_channel=with_doppler  # !!!! TODO debug with `True`
    #====
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    if 0:
        print('@@ show_data_loader(train_loader) -------- ^^')
        _channel, _, _, _ = show_data_loader(train_loader)  # only the first batch shown
        print('@@ show_data_loader(train_loader) -------- $$')

    #

    validate_dataset = ThyroidDataset(
        phase='val',
        dataset=validate_ds_path,
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

    num_attention_maps = 32  # @@ cf. 16 in 'main_legacy.py'
    net = WSDAN(num_classes=WSDAN_NUM_CLASSES, M=num_attention_maps, model=model, pretrained=True)
    net.to(device)
    feature_center = torch.zeros(WSDAN_NUM_CLASSES, num_attention_maps * net.num_features).to(device)

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

    START_EPOCH = 0

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
            "architecture": f"WS-DAN-{model}",
            "optimizer": opt_type,
            "dataset": "Thyroid",
            "train-data-augment": f"{channel}-channel",
            "epochs": f"{total_epochs - START_EPOCH}({START_EPOCH}->{total_epochs})" ,
        })

    #

    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'
        .format(total_epochs, batch_size, len(train_dataset), len(validate_dataset)))

    ckpt = thyroid_train.train(
        device, net, feature_center, batch_size, train_loader, validate_loader,
        optimizer, scheduler, run_name, logs, START_EPOCH, total_epochs,
        with_doppler=with_doppler, savepath=savepath)
    print('@@ done; ckpt:', ckpt)

    return ckpt


def demo_thyroid_train(
        model=MODEL_DEFAULT,
        train_ds_path=TRAIN_DS_PATH_DEFAULT,
        validate_ds_path=VALIDATE_DS_PATH_DEFAULT):
    return _demo_thyroid_train(False, model, train_ds_path, validate_ds_path,
        mk_artifact_dir('demo_thyroid_train'))

def demo_thyroid_train_with_doppler(
        model=MODEL_DEFAULT,
        train_ds_path=TRAIN_DS_PATH_DEFAULT,
        validate_ds_path=VALIDATE_DS_PATH_DEFAULT):
    return _demo_thyroid_train(True, model, train_ds_path, validate_ds_path,
        mk_artifact_dir('demo_thyroid_train_with_doppler'))


def demo_doppler_compare():
    print('\n\n\n\n@@ demo_doppler_comp(): ^^')

    from src.doppler import doppler_comp, get_iou, plot_comp, get_sample_paths
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

    print('@@ demo_doppler_comp(): $$')


if __name__ == '__main__':
    print("@@ torch.__version__:", torch.__version__)

    if 0:  # adaptation of 'compare.{ipynb,py}' exported from https://colab.research.google.com/drive/1kxMFgo1LyVqPYqhS6_UJKUsVvA2-l9wk
        demo_doppler_compare()

    if 0:
        # ckpt = 'ttt/51/output/demo_thyroid_train/densenet_250_8_lr-1e5_n4_75.000'  # 0.800
        # demo_thyroid_test(ckpt)  # TODO - generate 'confusion_matrix_test-*.png', 'test-*.png'

        ckpt = 'densenet_224_8_lr-1e5_n4_95.968.ckpt'  # 0.9xx, LGTM
        demo_thyroid_test(ckpt, 'densenet121', TEST_DS_PATH_DEFAULT, 224, 8)

    if 1:
        #model = 'densenet121'
        model = 'resnet34'

        #

        train_ds_path = build_dataset({
            'benign': ['Markers_Train_Remove_Markers/Benign_Remove/train'],
            'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/train'],
        }, root='Dataset_doppler_100c')  # 70% 70% (doppler matched)

        validate_ds_path = build_dataset({
            'benign': ['Markers_Train_Remove_Markers/Benign_Remove/validate'],
            'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/validate'],
        }, root='Dataset_doppler_100c')  # 30% 30% (doppler matched)

        ckpt = demo_thyroid_train(model, train_ds_path, validate_ds_path)
        #ckpt = demo_thyroid_train_with_doppler(model, train_ds_path, validate_ds_path)

        #

        test_ds_path = build_dataset({
            'benign': ['Markers_Train_Remove_Markers/Benign_Remove/test'],
            'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/test'],
        }, root='Dataset_doppler_100c')  # max max (doppler not matched)

        demo_thyroid_test(ckpt, model, test_ds_path)

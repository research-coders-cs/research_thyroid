import torch
from torch.utils.data import DataLoader

import digitake
from src.wsdan import WSDAN
from src.transform import ThyroidDataset, get_transform##, get_transform_center_crop, transform_fn
from src.utils import mk_artifact_dir, get_device, show_data_loader
from src import thyroid_train, thyroid_test

import os
import logging
logging.basicConfig(level=logging.INFO)


WSDAN_NUM_CLASSES = 2

def demo_thyroid_test(ckpt,
        model='densenet121', target_resize=250, batch_size=8, num_attention_maps=32):

    print('\n\n@@ demo_thyroid_test(): ^^')
    print("@@ model:", model)
    print("@@ target_resize:", target_resize)
    print("@@ batch_size:", batch_size)

    device = get_device()
    print("@@ device:", device)

    #

    test_ds_path = digitake.preprocess.build_dataset({
      'malignant': ['Test/Malignant'],
      'benign': ['Test/Benign'],
    }, root='Dataset_train_test_val')  # No Markers

    print('@@ test_ds_path:', test_ds_path)
    print("@@ len(test_ds_path['malignant']):", len(test_ds_path['malignant']))
    print("@@ len(test_ds_path['benign']):", len(test_ds_path['benign']))

    #

    test_dataset = ThyroidDataset(
        phase='test',
        dataset=test_ds_path,
        transform=get_transform(target_resize, phase='basic'),
        with_alpha_channel=False)

    #@@workers = 2
    workers = 0  # @@
    print('@@ workers:', workers)

    test_loader_no = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),  # @@
        shuffle=False,
        num_workers=workers,
        pin_memory=True)

    #

    net = WSDAN(num_classes=WSDAN_NUM_CLASSES, M=num_attention_maps, model=model, pretrained=True)
    net.to(device)

    results = thyroid_test.test(device, net, batch_size, test_loader_no, ckpt,
        savepath=mk_artifact_dir('demo_thyroid_test'))
    # print('@@ results:', results)

    #

    if 1:  #  legacy
        from src.legacy import print_scores, print_auc

        print('\n\n@@ ======== print_scores(results)')
        print_scores(results)

        _enable_plot = 0  # @@
        print(f'\n\n@@ ======== print_auc(results, enable_plot={_enable_plot})')
        # print_auc(results, len(test_dataset), enable_plot=_enable_plot)

    #

    print('@@ demo_thyroid_test(): $$')


def _demo_thyroid_train(with_doppler, savepath):
    print('\n\n@@ _demo_thyroid_train(): ^^')

    device = get_device()
    print("@@ device:", device)

    #

    print('@@ with_doppler:', with_doppler)
    print('@@ savepath:', savepath)

    train_ds_path = None
    if with_doppler:
        train_ds_path = digitake.preprocess.build_dataset({
            # @@ TODO update with "Markers_Train_Remove_Markers" instead !!!!
            'malignant': ['Markers_Train/Malignant'],
            'benign': ['Markers_Train/Benign'],
        }, root='Siriraj_sample_doppler_comp')
        #print(train_ds_path)
        print(len(train_ds_path['malignant']), len(train_ds_path['benign']))  # @@ 2 7
    else:
        train_ds_path = digitake.preprocess.build_dataset({
            'malignant': ['Train/Malignant'],
            'benign': ['Train/Benign'],
        }, root='Dataset_train_test_val')
        #print(train_ds_path)
        print(len(train_ds_path['malignant']), len(train_ds_path['benign']))  # @@ 20 21

    #

    val_ds_path = digitake.preprocess.build_dataset({
      'malignant': ['Val/Malignant'],
      'benign': ['Val/Benign'],
    }, root='Dataset_train_test_val')
    #print(val_ds_path)
    print(len(val_ds_path['malignant']), len(val_ds_path['benign']))  # @@ 10 10

    #

    # model = 'resnet50'
    model = 'densenet121'

    target_resize = 250
    batch_size = 8 #@param ["8", "16", "4", "1"] {type:"raw"}

    number = 4 #@param ["1", "2", "3", "4", "5"] {type:"raw", allow-input: true}

    workers = 2
    print('@@ workers:', workers)

    lr = 0.001 #@param ["0.001", "0.00001"] {type:"raw"}
    lr_ = "lr-1e5" #@param ["lr-1e3", "lr-1e5"]

    start_epoch = 0
    #total_epochs = 5
    total_epochs = 10  # @@

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

    print('@@ show_data_loader(train_loader) -------- ^^')
    _channel, _, _, _ = show_data_loader(train_loader)
    print('@@ show_data_loader(train_loader) -------- $$')

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
            "epochs": f"{total_epochs - start_epoch}({start_epoch}->{total_epochs})" ,
        })

    #

    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'
        .format(total_epochs, batch_size, len(train_dataset), len(validate_dataset)))

    ckpt = thyroid_train.train(
        device, net, feature_center, batch_size, train_loader, validate_loader,
        optimizer, scheduler, run_name, logs, start_epoch, total_epochs,
        savepath=savepath)
    print('@@ done; ckpt:', ckpt)

    return ckpt


def demo_thyroid_train():
    return _demo_thyroid_train(False, mk_artifact_dir('demo_thyroid_train'))

def demo_thyroid_train_with_doppler():
    return _demo_thyroid_train(True, mk_artifact_dir('demo_thyroid_train_with_doppler'))


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

    if 1:
        # seemingly unlearned ...
        # ckpt = "WSDAN_densenet_224_16_lr-1e5_n1-remove_220828-0837_85.714.ckpt"
        # ckpt = "WSDAN_doppler_densenet_224_16_lr-1e5_n5_220905-1309_78.571.ckpt"
        # demo_thyroid_test(ckpt, 'densenet121', 224, 16)

        # ckpt = 'ttt/51/output/demo_thyroid_train/densenet_250_8_lr-1e5_n4_75.000'  # 0.800
        # demo_thyroid_test(ckpt)

        ckpt = 'densenet_224_8_lr-1e5_n4_95.968.ckpt'  # 0.9xx, LGTM
        demo_thyroid_test(ckpt, 'densenet121', 224, 8)

        # ?? trained via different src ??
        #---- ng
        #  - ?? INFO:root:WSDAN: Some params were not loaded: INFO:root:features.conv0.weight, features.norm0.weight, features.norm0.bias, features.norm0.running_mean, features.norm0.running_var, ...
        # ckpt = 'densenet121_batch4_epoch100.ckpt'  # num_attentions: 32 per 'densenet121_batch4_epoch100.log'
        # demo_thyroid_test(ckpt, 'densenet121', 320, 4)  # nonesense results
        #---- ng
        # ckpt = 'resnet34_batch4_epoch100.ckpt'  # num_attentions: 32
        # demo_thyroid_test(ckpt, 'resnet34', 400, 4)  # 0.650

    if 0:
        ckpt = demo_thyroid_train()
        demo_thyroid_test(ckpt)  # TODO - generate 'confusion_matrix_test-*.png', 'test-*.png'

    if 0:
        ckpt = demo_thyroid_train_with_doppler()
        demo_thyroid_test(ckpt)

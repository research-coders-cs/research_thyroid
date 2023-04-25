import torch
# from torch.utils.data import DataLoader

from digitake.preprocess import build_dataset

import wsdan  # via 'research-thyroid-wsdan' pkg

# from wsdan.net import WSDAN, net_train, net_test

# both
# from wsdan.demo.transform import ThyroidDataset, get_transform##, get_transform_center_crop, transform_fn
# from wsdan.demo.utils import mk_artifact_dir, get_device

# test only
# from wsdan.demo.utils import mk_artifact_dir, get_device, show_data_loader
# from wsdan.demo.stats import print_scores, print_auc, print_poa

from wsdan.demo import test as demo_test
from wsdan.demo import train as demo_train
from wsdan.demo import train_with_doppler as demo_train_with_doppler
from wsdan.demo import MODEL_DEFAULT, \
    TRAIN_DS_PATH_DEFAULT, VALIDATE_DS_PATH_DEFAULT, TEST_DS_PATH_DEFAULT


import logging
logging.basicConfig(level=logging.INFO)


def _demo_thyroid_train(with_doppler, model, train_ds_path, validate_ds_path, savepath):
    print('\n\n@@ _demo_thyroid_train(): ^^')

    device = get_device()
    print("@@ device:", device)

    print('@@ with_doppler:', with_doppler)
    print('@@ model:', model)
    print('@@ savepath:', savepath)

    #print('@@ train_ds_path:', train_ds_path)
    print("@@ lens train_ds_path:", len(train_ds_path['benign']), len(train_ds_path['malignant']))

    #print('@@ validate_ds_path:', validate_ds_path)
    print("@@ lens validate_ds_path:", len(validate_ds_path['benign']), len(validate_ds_path['malignant']))

    target_resize = 250
    batch_size = 8 #@param ["8", "16", "4", "1"] {type:"raw"}

    number = 4 #@param ["1", "2", "3", "4", "5"] {type:"raw", allow-input: true}

    workers = 2
    print('@@ workers:', workers)

    lr = 0.001 #@param ["0.001", "0.00001"] {type:"raw"}
    lr_ = "lr-1e5" #@param ["lr-1e3", "lr-1e5"]

    #total_epochs = 1
    total_epochs = 2
    #total_epochs = 40

    run_name = f"{model}_{target_resize}_{batch_size}_{lr_}_n{number}"
    print('@@ run_name:', run_name)

    #

    from wsdan.net.doppler import to_doppler  # !!!!

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
        pin_memory=True)

    if 0:
        print('@@ show_data_loader(train_loader) -------- ^^')
        _channel, _, _, _ = show_data_loader(train_loader)  # only the first batch shown
        print('@@ show_data_loader(train_loader) -------- $$')

    #

    validate_dataset = ThyroidDataset(
        phase='val',
        dataset=validate_ds_path,
        transform=get_transform(target_resize, phase='basic'),
        with_alpha_channel=False)

    validate_loader = DataLoader(
        validate_dataset,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=workers,
        pin_memory=True)

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

    ckpt = net_train.train(
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




if __name__ == '__main__':
    print("@@ torch.__version__:", torch.__version__)

    if 0:  # adaptation of 'compare.{ipynb,py}' exported from https://colab.research.google.com/drive/1kxMFgo1LyVqPYqhS6_UJKUsVvA2-l9wk
        wsdan.demo.doppler_compare()

    if 1:
        # ckpt = 'ttt/51/output/demo_thyroid_train/densenet_250_8_lr-1e5_n4_75.000'  # 0.800
        # demo.test(ckpt)  # TODO - generate 'confusion_matrix_test-*.png', 'test-*.png'

        ckpt = 'densenet_224_8_lr-1e5_n4_95.968.ckpt'  # 0.9xx, LGTM
        demo_test(ckpt, 'densenet121', TEST_DS_PATH_DEFAULT, 224, 8)

    if 0:
        #model = 'densenet121'
        model = 'resnet34'

        train_ds_path = build_dataset({
            'benign': ['Markers_Train_Remove_Markers/Benign_Remove/train'],
            'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/train'],
        }, root='Dataset_doppler_100d')  # 70% 70% (doppler matched)

        validate_ds_path = build_dataset({
            'benign': ['Markers_Train_Remove_Markers/Benign_Remove/validate'],
            'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/validate'],
        }, root='Dataset_doppler_100d')  # 30% 30% (doppler matched)

        #ckpt = demo_thyroid_train(model, train_ds_path, validate_ds_path)
        ckpt = demo_thyroid_train_with_doppler(model, train_ds_path, validate_ds_path)

        test_ds_path = build_dataset({
            'benign': ['Markers_Train_Remove_Markers/Benign_Remove/test'],
            'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/test'],
        }, root='Dataset_doppler_100d')  # 75 75

        demo_thyroid_test(ckpt, model, test_ds_path)

    if 0:  # experiment - 'heatmap-compare-doppler_100c-TrueFalse--rounds--tests_100d'
        # ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-False/output/demo_thyroid_train/resnet34_250_8_lr-1e5_n4'
        # ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-False/run-2/output/demo_thyroid_train/resnet34_250_8_lr-1e5_n4'
        # ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-False/run-3/output/demo_thyroid_train/resnet34_250_8_lr-1e5_n4'
        # ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-True/output/demo_thyroid_train_with_doppler/resnet34_250_8_lr-1e5_n4'
        ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-True/run-2/output/demo_thyroid_train_with_doppler/resnet34_250_8_lr-1e5_n4'
        # ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-True/run-3/output/demo_thyroid_train_with_doppler/resnet34_250_8_lr-1e5_n4'

        demo_thyroid_test(ckpt, 'resnet34', build_dataset({
            'benign': ['Markers_Train_Remove_Markers/Benign_Remove/test'],
            'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/test'],
        }, root='Dataset_doppler_100d'))

    if 0:  # demo - acc 0.65-0.68
        ckpt = 'WSDAN_doppler_100d-resnet34_250_8_lr-1e5_n4.ckpt'
        test_ds_path = build_dataset({
            'benign': ['Markers_Train_Remove_Markers/Benign_Remove/test'],
            'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/test'],
        }, root='Dataset_doppler_100d')

        demo_thyroid_test(ckpt, 'resnet34', test_ds_path, 250, 8)

    if 0:  # experiment - default
        model = 'resnet34'
        ####ckpt = demo_thyroid_train(model)
        ckpt = demo_train(model)

        demo_test(ckpt, model)

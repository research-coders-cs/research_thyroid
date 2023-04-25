from . import DataLoader, ThyroidDataset, get_transform, \
    mk_artifact_dir, get_device, \
    WSDAN_NUM_CLASSES, MODEL_DEFAULT, TEST_DS_PATH_DEFAULT
from .utils import show_data_loader
from .stats import print_scores, print_auc, print_poa
from ..net import WSDAN, net_test

def run(ckpt, model=MODEL_DEFAULT, ds_path=TEST_DS_PATH_DEFAULT,
        target_resize=250, batch_size=8, num_attention_maps=32):

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

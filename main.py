import wsdan  # 'research-thyroid-wsdan' pkg (i.e. 'src/wsdan/*')
from wsdan.demo import test as demo_test
from wsdan.demo import train as demo_train
from wsdan.demo import train_with_doppler as demo_train_with_doppler
from wsdan.digitake.preprocess import build_dataset

import logging
logger = logging.getLogger('@@')
logger.setLevel(level=logging.DEBUG if 1 else logging.INFO)


# TODO - `print()` -> `logging.info()` in 'src/wsdan/**/*.py'; output control in Colab
if __name__ == '__main__':

    if 0:  # adaptation of 'compare.{ipynb,py}' exported from https://colab.research.google.com/drive/1kxMFgo1LyVqPYqhS6_UJKUsVvA2-l9wk
        wsdan.demo.doppler_compare()

    if 0:
        # ckpt = 'ttt/51/output/demo_train/densenet_250_8_lr-1e5_n4_75.000'  # 0.800
        # demo.test(ckpt)  # TODO - generate 'confusion_matrix_test-*.png', 'test-*.png'

        from wsdan.demo import TEST_DS_PATH_DEFAULT
        ckpt = 'densenet_224_8_lr-1e5_n4_95.968.ckpt'  # 0.9xx, LGTM
        demo_test(ckpt, 'densenet121', TEST_DS_PATH_DEFAULT, 224, 8)

    if 0:
        total_epochs = 40
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

        #ckpt = demo_train(total_epochs, model, train_ds_path, validate_ds_path)
        ckpt = demo_train_with_doppler(total_epochs, model, train_ds_path, validate_ds_path)

        test_ds_path = build_dataset({
            'benign': ['Markers_Train_Remove_Markers/Benign_Remove/test'],
            'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/test'],
        }, root='Dataset_doppler_100d')  # 75 75

        demo_test(ckpt, model, test_ds_path)

    if 0:  # experiment - 'heatmap-compare-doppler_100c-TrueFalse--rounds--tests_100d'
        # ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-False/output/demo_train/resnet34_250_8_lr-1e5_n4'
        # ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-False/run-2/output/demo_train/resnet34_250_8_lr-1e5_n4'
        # ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-False/run-3/output/demo_train/resnet34_250_8_lr-1e5_n4'
        # ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-True/output/demo_train_with_doppler/resnet34_250_8_lr-1e5_n4'
        ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-True/run-2/output/demo_train_with_doppler/resnet34_250_8_lr-1e5_n4'
        # ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-True/run-3/output/demo_train_with_doppler/resnet34_250_8_lr-1e5_n4'

        demo_test(ckpt, 'resnet34', build_dataset({
            'benign': ['Markers_Train_Remove_Markers/Benign_Remove/test'],
            'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/test'],
        }, root='Dataset_doppler_100d'))

    if 0:  # demo - acc 0.65-0.68
        ckpt = 'WSDAN_doppler_100d-resnet34_250_8_lr-1e5_n4.ckpt'
        test_ds_path = build_dataset({
            'benign': ['Markers_Train_Remove_Markers/Benign_Remove/test'],
            'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/test'],
        }, root='Dataset_doppler_100d')

        demo_test(ckpt, 'resnet34', test_ds_path, 250, 8)

    if 1:  # experiment - default
        total_epochs = 10
        model = 'resnet34'

        ckpt = demo_train(total_epochs, model)
        demo_test(ckpt, model)

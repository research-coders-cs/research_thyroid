# !! pipenv run python3 -m pip install --force-reinstall .  # for `import wsdan` to work
# !! pipenv run scripts/plot_ep_time_from_log.py

import PIL
from PIL import Image

import sys
import numpy as np
import datetime
import time

if 1:
    from wsdan.demo.plot_if import is_colab, get_plt, plt_show
    plt = get_plt()

    from wsdan.demo.stats import print_auc
else:  # standalone
    import matplotlib.pyplot as plt
    def is_colab():
        return False
    def plt_show(plt):
        print('@@ plt_show(): \'q\' to close interactively')
        plt.show()

if is_colab():
    import os
    os.chdir('/content/drive/MyDrive/colab/foo')

import torch
#print(torch.__version__)

#---- ^^ plot_*()

def plt_img_tensor(file):
    img = Image.open(file).convert('RGB')
    img = np.array(img)
    #print('@@ img.shape:', img.shape)  # e.g. (420, 420, 3)

    x = torch.tensor(img, dtype=torch.float32)
    x /= 255  # min-max normalization
    #print(x.min(), x.max())  # e.g. tensor(0.) tensor(1.)

    plt.imshow(x)
    plt_show(plt)

def plt_ep_val(logs, mode='time'):
    if mode != 'time' and mode != 'acc':
        raise Exception("`mode` should be one of 'time', 'acc'")

    for log in logs:
        deltas = log_to_deltas(log, mode)
        xs_ep = range(0, len(deltas))  # (ep0, ep1, ep2, ...) i.e. (0, 1, 2, ...)
        ys_val = deltas.values()  # [val0, val1, val2, ...]

        plt.plot(xs_ep, ys_val, label=f'{log.split("/")[-1]}')

        plt.xlim([xs_ep[0], xs_ep[-1]])

        if mode == 'time':
            plt.ylim([0, max(ys_val) + 5])
        elif mode == 'acc':
            plt.ylim([55, 102])

    plt.xlabel('Epoch Index')

    if mode == 'time':
        plt.ylabel('Processing Time (s)')
    elif mode == 'acc':
        ##plt.ylabel('ValAcc (%)')
        plt.ylabel('Model Accuracy (%)')

    ##plt.legend(loc='upper right')
    plt.legend(loc='lower right')
    plt_show(plt)

def plt_auc():
    pred = torch.tensor([
        [ 0.6650, -0.4284],
        [ 0.2460, -0.1261],
        [-0.4980,  0.5372],
        [-0.0229, -0.0511],
        [ 0.5294, -0.2512],
        [ 0.2646,  0.2131],
        [-1.3620,  0.9549],
        [-1.1261,  0.5606],
        [-0.9976,  0.1080],
        [ 0.5730, -0.4583],
        [-0.0655,  0.3107],
        [-0.1167, -0.2246],
        [ 0.4521, -0.5234],
        [-0.1683,  0.4976],
        [ 0.6475, -0.1778],
        [-0.5879,  0.0949],
        [-0.0213,  0.1973],
        [ 0.2243, -0.4667],
        [-0.7021,  0.7416],
        [-0.1003,  0.1517]], dtype=torch.float32)
    true = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    if 0:
        print('@@ test_size:', test_size)
        print('@@ pred:', pred)  # results[2]
        print('@@ true:', true)  # results[3]

    results = [None, None, pred, true]
    test_size = 20
    print_auc(results, test_size, plot=True, plot_savepath='.')

#---- $$ plot_*()

def log_to_deltas(log, mode):
    with open(log) as file:
        deltas = {}
        acc_best = 0.0

        for line in file:
            sec = None
            if line.startswith('Epoch '):  # Epoch 2.001/20: 100%|██████████| 63/63 [02:12<00:00,  2.10s/ batches, Loss 3.2508, Raw Acc (69.48), Crop Acc (60.44), Drop Acc (63.05), Val Loss 4.0830, Val Acc (67.07)]

                seg = line.split('<')[0]  # Epoch 2.001/20: 100%|██████████| 63/63 [02:12
                ep = seg.split('/')[0]  # Epoch 2.001

                tm = time.strptime(seg.split('[')[1],'%M:%S')  # 02:12 -> tm
                sec = datetime.timedelta(minutes=tm.tm_min, seconds=tm.tm_sec).total_seconds()

            if mode == 'time' and sec is not None:
                deltas[ep] = int(sec)
            elif mode == 'acc' and 'Val Acc (' in line:
                seg = line.split('Val Acc (')[1]  # "67.07)]"
                acc = float(seg.split(')')[0])  # 67.07

                #deltas[ep] = acc
                #====
                if acc > acc_best:
                    acc_best = acc
                    deltas[ep] = acc
                else:
                    deltas[ep] = acc_best

        # print('@@ deltas:', deltas)
        print('@@ len(deltas):', len(deltas))
        return deltas


if __name__ == '__main__':
    if 0:
        try:
            log = sys.argv[1]  # e.g. results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r1
        except:
            print(f'Usage: python3 {sys.argv[0]} <log file>')
            exit()

        plt_ep_val([log])
        exit()

    if 0:
        plt_img_tensor('test.png')
        exit()

    if 0:
        plt_auc()
        exit()

    # plt_ep_val([
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r1',
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r2',
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r3',
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r4',
    # ], 'time')

    # plt_ep_val([
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r1',
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r2',
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r3',
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r4',
    # ], 'acc')

    """
    plt_ep_val([
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r5',
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r6',
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r7',
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r8',
        'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r5',
        'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r6',
        'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r7',
        'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r8',
    ], 'time')  # ep_time_nd_d_5678.png

    plt_ep_val([
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r5',
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r6',
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r7',
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r8',
    ], 'acc')  # ep_model_acc_nd_5678.png

    plt_ep_val([
        'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r5',
        'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r6',
        'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r7',
        'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r8',
    ], 'acc')  # ep_model_acc_d_5678.png
    """

    plt_ep_val([
        'results--doppler_100g-TrueFalse/log-d-crop-only-r1',
        'results--doppler_100g-TrueFalse/log-d-crop-only-r2',
        'results--doppler_100g-TrueFalse/log-d-crop-only-r3',
    ], 'time')  # ep_time_d_crop_only_123.png
    plt_ep_val([
        'results--doppler_100g-TrueFalse/log-d-crop-only-r1',
        'results--doppler_100g-TrueFalse/log-d-crop-only-r2',
        'results--doppler_100g-TrueFalse/log-d-crop-only-r3',
    ], 'acc')  # ep_model_acc_d_crop_only_123.png

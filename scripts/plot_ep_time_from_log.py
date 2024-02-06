try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

import PIL
from PIL import Image

import sys
import numpy as np
import datetime
import time
if IN_COLAB:
    import os
    os.chdir('/content/drive/MyDrive/colab/foo')

import torch
#print(torch.__version__)

#---- ^^
import matplotlib.pyplot as plt

def plt_show(plt):
    if not IN_COLAB:
        print('@@ plt_show(): \'q\' to close interactively')
    plt.show()

def plt_img_tensor(file):
    img = Image.open(file).convert('RGB')
    img = np.array(img)
    #print('@@ img.shape:', img.shape)  # e.g. (420, 420, 3)

    x = torch.tensor(img, dtype=torch.float32)
    x /= 255  # min-max normalization
    #print(x.min(), x.max())  # e.g. tensor(0.) tensor(1.)

    plt.imshow(x)
    plt_show(plt)

def plt_ep_sec(logs, mode='acc'):
    if mode != 'time' and mode != 'acc':
        raise Exception("`mode` should be one of 'time', 'acc'")

    for log in logs:
        deltas = log_to_deltas(log, mode)
        xs_ep = range(0, len(deltas))  # (ep0, ep1, ep2, ...) i.e. (0, 1, 2, ...)
        ys_sec = deltas.values()  # [sec0, sec1, sec2, ...]

        plt.plot(xs_ep, ys_sec, label=f'{log.split("/")[-1]}')

        plt.xlim([xs_ep[0], xs_ep[-1]])

        if mode == 'time':
            plt.ylim([0, max(ys_sec) + 5])
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
#---- $$

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

        plt_ep_sec([log])
        exit()

    if 0:
        plt_img_tensor('test.png')
        exit()

    # plt_ep_sec([
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r1',
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r2',
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r3',
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r4',
    # ], 'time')
    #
    # plt_ep_sec([
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r1',
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r2',
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r3',
    #     'results--doppler_100g-TrueFalse/log-d-ep20-kfold3-r4',
    # ], 'acc')

    plt_ep_sec([
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r5',
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r6',
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r7',
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r8',
    ], 'acc')

    plt_ep_sec([
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r5',
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r6',
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r7',
        'results--doppler_100g-TrueFalse/log-nd-ep20-kfold3-r8',
    ], 'time')
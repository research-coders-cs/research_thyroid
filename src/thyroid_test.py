import torch
from torch.nn import functional

from torchvision import transforms
ToPILImage = transforms.ToPILImage()

from .metric import TopKAccuracyMetric
from .augment import batch_augment

import os
import logging

from tqdm import tqdm
#from tqdm.notebook import tqdm


def generate_heatmap(attention_maps, threshold=0.5):
    print("Total attentions map. =", attention_maps.shape)

    amax = attention_maps.max()
    threshold=attention_maps.mean()
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...]/amax)  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < threshold).float() + \
        (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= threshold).float())  # G
    heat_attention_maps.append((1. - attention_maps[:, 0, ...]))  # B
    return torch.stack(heat_attention_maps, dim=1)


def test(device, net, batch_size, data_loader, ckpt, savepath=None):
    logging.info('Network loading from {}'.format(ckpt))

    ckpt_dict = torch.load(ckpt)

    print('@@ ckpt:', ckpt)
    for key, val in ckpt_dict.items():  # @@
        print('@@ ckpt_dict - key:', key)
        if key == 'logs': print('  ', val)
        if key == 'feature_center': print('  ', val)

    state_dict = ckpt_dict['state_dict']
    # for key, _ in state_dict.items(): print('@@ state_dict - key:', key)  # @@

    net.load_state_dict(state_dict)
    #exit()  # @@ !!

    raw_accuracy = TopKAccuracyMetric()
    ref_accuracy = TopKAccuracyMetric()
    raw_accuracy.reset()

    results = []
    net.eval()

    with torch.no_grad():

        pbar = tqdm(total=len(data_loader), unit=' batches')
        pbar.set_description('Test data')

        for i, (X, y, p) in enumerate(data_loader):
            paths = p['path']  # @@

            # obtain data for testing
            X = X.to(device)
            y = y.to(device)

            mean = X.mean((0,2,3)).view(1, 3, 1, 1)
            std = X.std((0,2,3)).view(1, 3, 1, 1)

            MEAN = mean.cpu()     # torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
            STD = std.cpu()           # torch.tensor([0.1, 0.1, 0.1]).view(1, 3, 1, 1)
            # MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            # STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

            ##################################
            # Raw Image
            ##################################
            y_pred_raw, _, attention_maps = net(X)

            ##################################
            # Attention Cropping
            ##################################
            crop_image = batch_augment(
                X, paths, attention_maps, savepath,
                mode='crop', theta=0.85, padding_ratio=0.05)

            # crop images forward
            y_pred_crop, _, _ = net(crop_image)
            importance =  torch.abs(y_pred_raw[0] - y_pred_raw[1])

            y_pred = (y_pred_raw + (y_pred_crop * 2 * importance)) / 3.

            channel = 3

            # reshape attention maps
            print(f"Input Shape:{X.shape} vs Attention Shape: {attention_maps.shape}")
            A = attention_maps.expand(-1, 4, 8, 8) if channel == 4 else attention_maps
            print(f"New Attention: {A.shape}, size={(X.size(2), X.size(3))}")
            attention_maps = functional.interpolate(A, size=(X.size(2), X.size(3)))
            attention_maps = attention_maps.cpu() / attention_maps.max().item()

            # get heat attention maps
            heat_threshold = 0.5
            heat_attention_maps = generate_heatmap(attention_maps, threshold=heat_threshold)

            # raw_image, heat_attention, raw_attention
            raw_image = X.cpu() * STD + MEAN

            print(f"X:{raw_image.shape} vs HEAT Shape: {heat_attention_maps.shape}")
            #H = heat_attention_maps.repeat(1, 4, 1, 1)
            grays_dim = heat_attention_maps.shape[:]
            grays_dim = [grays_dim[0], 1, grays_dim[2], grays_dim[3]]
            ones = torch.ones(grays_dim)

            H = torch.hstack([heat_attention_maps, ones]) if channel == 4 else heat_attention_maps
            heat_attention_image = raw_image * 0.5 + H *0.5

            raw_attention_image = raw_image * attention_maps

            if savepath is not None:
                for idx in range(X.size(0)):
                    rimg = ToPILImage(raw_image[idx])
                    raimg = ToPILImage(raw_attention_image[idx])
                    haimg = ToPILImage(heat_attention_image[idx])
                    rimg.save(os.path.join(savepath, '%03d_raw.png' % (i * batch_size + idx)))
                    raimg.save(os.path.join(savepath, '%03d_raw_atten.png' % (i * batch_size + idx)))
                    haimg.save(os.path.join(savepath, '%03d_heat_atten.png' % (i * batch_size + idx)))

            results = (X, crop_image, y_pred, y, p, heat_attention_image, y_pred_crop)

            # Top K
            epoch_raw_acc = raw_accuracy(y_pred_raw, y)
            epoch_ref_acc = ref_accuracy(y_pred, y)

            # end of this batch
            batch_info = 'Val Acc: Raw ({:.2f}), Refine ({:.2f})'.format(
                epoch_raw_acc[0], epoch_ref_acc[0])
            pbar.update()
            pbar.set_postfix_str(batch_info)
            torch.cuda.empty_cache()

        pbar.close()

    return results
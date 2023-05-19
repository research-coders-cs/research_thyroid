import torch
from torch.nn import functional

from torchvision import transforms
ToPILImage = transforms.ToPILImage()

import numpy as np
import random
import os
from .doppler import resolve_hw_slices


# https://github.com/GuYuc/WS-DAN.PyTorch/blob/87779124f619ceeb445ddfb0246c8a22ff324db4/eval.py#L37
def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)

def get_raw_image(batch_image):
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return batch_image * STD + MEAN

def dump_heatmap(savepath, prefix, raw_image, atten_map, imgH, imgW, batch_index):
    rimg = ToPILImage(raw_image[batch_index])
    rimg.save(os.path.join(savepath, f'{prefix}_raw.png'))

    _attention_maps = functional.interpolate(
        atten_map, size=(imgH, imgW), mode='bilinear')
    _attention_maps = _attention_maps.cpu() / _attention_maps.max().item()
    heat_attention_map = generate_heatmap(_attention_maps)

    heat_attention_image = (raw_image * 0.3) + (heat_attention_map.cpu() * 0.7)
    himg = ToPILImage(heat_attention_image[batch_index])
    himg.save(os.path.join(savepath, f'{prefix}_heat_atten.png'))

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def img_gpu_to_cpu(img):
    img_full = img.cpu().permute(1, 2, 0).numpy()
    img_full = NormalizeData(img_full) * 255
    return img_full

def batch_augment(images, paths, attention_map, savepath=None, use_doppler=False,
                  mode='crop', theta=0.5, padding_ratio=0.1):
    print('@@ images.size():', images.size())
    raw_image = get_raw_image(images.cpu())  # @@
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for idx in range(batches):
            atten_map = attention_map[idx:idx + 1]
            if savepath is not None:  # @@ debug
                dump_heatmap(savepath, f'debug_crop_idx_{idx}', raw_image, atten_map, imgH, imgW, idx)
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = functional.interpolate(atten_map, size=(imgH, imgW), mode='bilinear') >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            #-------- @@
            print(f'@@ [idx={idx}] crop: ({width_min}, {height_min}), ({width_max}, {height_max})')

            if use_doppler:
                bbox_crop = np.array([
                    width_min, height_min,
                    width_max, height_max], dtype=np.float32)
                train_img_copy = np.array(
                    img_gpu_to_cpu(images[idx])).astype(np.uint8).copy()
                train_img_path = paths[idx]

                sh, sw = resolve_hw_slices(
                    bbox_crop, train_img_copy, train_img_path, idx, (imgH, imgW), savepath)
            else:
                sh, sw = slice(height_min, height_max), slice(width_min, width_max)
            #-------- @@

            crop_images.append(functional.interpolate(
                #images[idx:idx + 1, :, height_min:height_max, width_min:width_max],
                images[idx:idx + 1, :, sh, sw],  # @@
                size=(imgH, imgW), mode='bilinear'))

        crop_images = torch.cat(crop_images, dim=0)
        print("@@ crop_images.shape:", crop_images.shape)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for idx in range(batches):
            atten_map = attention_map[idx:idx + 1]
            if savepath is not None:  # @@ debug
                dump_heatmap(savepath, f'debug_drop_idx_{idx}', raw_image, atten_map, imgH, imgW, idx)

            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()
            drop_masks.append(functional.interpolate(
                atten_map, size=(imgH, imgW), mode='bilinear') < theta_d)

        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()

        print("drop_images : ", drop_images.shape)
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)

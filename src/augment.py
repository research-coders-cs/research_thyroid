import torch
from torch.nn import functional

import numpy as np
import cv2
import os
import random


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def img_gpu_to_cpu(img):
    img_full = img.cpu().permute(1, 2, 0).numpy()
    img_full = NormalizeData(img_full) * 255
    return img_full

def batch_augment(images, attention_map, savepath, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for idx in range(batches):
            atten_map = attention_map[idx:idx + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = functional.interpolate(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            print('crop: ', (height_min, width_min), ((height_min + height_max), (width_min + width_max)))

            if 1:  # @@
                img = img_gpu_to_cpu(images[idx])
                img = np.array(img).astype(np.uint8).copy()

                img_ = cv2.rectangle(img,
                    (width_min, height_min),
                    ((width_min + width_max), (height_min + height_max)),
                    (0, 0, 255), 1)
                cv2.imwrite(os.path.join(savepath, f'debug_crop_idx_{idx}_as_bbox.jpg'), img_)

                img_ = img_[height_min:height_max, width_min:width_max, :].copy()
                cv2.imwrite(os.path.join(savepath, f'debug_crop_idx_{idx}.jpg'), img_)

            crop_images.append(functional.interpolate(
                images[idx:idx + 1, :, height_min:height_max, width_min:width_max],
                size=(imgH, imgW)))

        crop_images = torch.cat(crop_images, dim=0)
        print("@@ crop_images.shape:", crop_images.shape)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for idx in range(batches):
            atten_map = attention_map[idx:idx + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()
            drop_masks.append(functional.interpolate(
                atten_map, size=(imgH, imgW)) < theta_d)

        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()

        print("drop_images : ", drop_images.shape)
        # cv2_imshow(img_gpu_to_cpu(drop_images[0]))
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)

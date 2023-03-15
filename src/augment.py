import torch
from torch.nn import functional

from .doppler import detect_doppler, get_iou, to_doppler

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

def batch_augment(images, paths, attention_map, savepath,
                  mode='crop', theta=0.5, padding_ratio=0.1):
    print('@@ images.size():', images.size())
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

            print(f'[idx={idx}] crop: ', (height_min, width_min), ((height_min + height_max), (width_min + width_max)))

            if 1:  # ======== TODO refactor ^^, cleanup `doppler_train_loader` stuff
                bbox_crop = np.array([
                    width_min,
                    height_min,
                    width_min + width_max,
                    height_min + height_max], dtype=np.float32)

                path_doppler = to_doppler[paths[idx]] if paths[idx] in to_doppler else None
                if path_doppler is not None:  # @@
                    # get doppler bbox (scaled)
                    raw = cv2.imread(path_doppler)
                    bbox_raw = detect_doppler(raw)
                    bbox = np.array([
                        bbox_raw[0] * imgW / raw.shape[1],
                        bbox_raw[1] * imgH / raw.shape[0],
                        bbox_raw[2] * imgW / raw.shape[1],
                        bbox_raw[3] * imgH / raw.shape[0]], dtype=np.float32)

                    iou = get_iou(bbox, bbox_crop)
                    debug_fname_jpg = f'debug_crop_doppler_{idx}_iou_%0.4f.jpg' % iou
                    print('@@ debug_fname_jpg:', debug_fname_jpg)

                    if 1:  # debug dump
                        train_img_copy = np.array(
                            img_gpu_to_cpu(images[idx])).astype(np.uint8).copy()

                        # superpose doppler bbox
                        cv2.rectangle(train_img_copy,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            (255, 255, 0), 1)

                        # superpose crop bbox
                        img_ = cv2.rectangle(train_img_copy,
                            (width_min, height_min),
                            ((width_min + width_max), (height_min + height_max)),
                            (0, 0, 255), 1)
                        cv2.imwrite(os.path.join(
                            savepath, debug_fname_jpg), img_)

                        # crop patch image, ok
                        # img_ = img_[height_min:height_max, width_min:width_max, :].copy()
                        # cv2.imwrite(os.path.join(savepath, f'debug_crop_idx_{idx}.jpg'), img_)
            # ======== TODO refactor vv

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

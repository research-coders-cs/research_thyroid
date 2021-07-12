from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import torch

# imagenet mean and std
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


##################################
# transform in dataset to target size
##################################
def get_transform(target_size, phase='train'):
    transform_dict = {
        'train':
            transforms.Compose([
                transforms.Resize(size=(int(target_size[0] * 1.1), int(target_size[1] / 1.1))),
                transforms.RandomRotation(90, interpolation=Image.BILINEAR),
                transforms.RandomCrop(target_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.126, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            ]),
        'val':
            transforms.Compose([
                transforms.Resize(size=(int(target_size[0] / 0.9), int(target_size[1] / 0.9))),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            ]),
        'test':
            transforms.Compose([
                transforms.Resize(size=(int(target_size[0] / 0.9), int(target_size[1] / 0.9))),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            ])
    }

    if phase in transform_dict:
        return transform_dict[phase]
    else:
        raise "Unknown pharse specified"

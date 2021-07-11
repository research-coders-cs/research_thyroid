from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import torch


##################################
# transform in dataset to target size
##################################
def get_transform(resize, phase='train'):
    transform_dict = {
        'train':
            transforms.Compose([
                transforms.Resize(size=(int(resize[0] / 0.9), int(resize[1] / 0.9))),
                transforms.RandomRotation(90, interpolation=Image.BILINEAR),
                transforms.RandomCrop(resize),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.126, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            ]),
        'val':
            transforms.Compose([
                transforms.Resize(size=(int(resize[0] / 0.9), int(resize[1] / 0.9))),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            ]),
        'test':
            transforms.Compose([
                transforms.Resize(size=(int(resize[0] / 0.9), int(resize[1] / 0.9))),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            ])
    }

    if phase in transform_dict:
        return transform_dict[phase]
    else:
        raise "Unknown pharse specified"

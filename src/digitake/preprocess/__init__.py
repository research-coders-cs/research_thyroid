from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import glob

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


def build_train_validation_set(datasource, val_size, root="", ext="*.png"):
    """
        datasource: dictionary with key as label of data, and value is a list of image path
        val_size: validation size, must be greater than total datasource size of each class's dataset
    """
    datasets = {}
    total_dataset = 0
    for key in datasource:
        datasets[key] = glob.glob(os.path.join(root, datasource[key], ext))
        path_count = len(datasets[key])
        total_dataset += path_count
        print(f'Found total {path_count} for {key}')
        assert val_size < path_count, f'The dataset {key} is smaller than validation size'

    training_set = {}
    validation_set = {}
    total_training_set = 0
    total_validation_set = 0

    print("--" * 25)

    for key in datasets:
        ds_size = len(datasets[key])  # size of each dataset

        training_set[key] = datasets[key][val_size:]  # Cut from val_size-th onward
        t_size = len(training_set[key])
        total_training_set += t_size
        print(f'Total train {key} size = {t_size}/{ds_size} ({t_size / ds_size:0.2f})')

        validation_set[key] = datasets[key][:val_size]  # Validation data is the first n elements from the dataset
        v_size = len(validation_set[key])
        total_validation_set += v_size
        print(f'Total validation {key} size = {v_size}/{ds_size} ({v_size / ds_size:0.2f})')

    print("--" * 25)
    print(f"Total dataset size = {total_dataset}")

    return training_set, validation_set

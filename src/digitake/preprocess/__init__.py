import os
import torchvision.transforms as transforms
import glob
from torchvision.transforms.functional import InterpolationMode
from typing import Dict

# imagenet mean and std
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


####################################################################
# transform in dataset to target size
####################################################################
def get_transform(target_size, phase='train'):
    """
    Predefined transformation pipe for the dataset
    :param target_size: tuple of (W,H) result image from the pipe
    :param phase: train/val/test phase of different transformation e.g. test will not need RandomCrop
    :return: a transformation function to target_size
    """
    if type(target_size) is int:
        target_size = (target_size, target_size)

    assert type(target_size) is tuple, "target_size must be tuple of (W:int, H:int) or int if square is needed"

    # enlarge 10% bigger for the later cropping
    enlarge = transforms.Resize(size=(int(target_size[0] * 1.1), int(target_size[1] * 1.1)))

    # ImageNet normalizer
    imagenet_normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

    transform_dict = {
        'train':
            transforms.Compose([
                enlarge,
                transforms.RandomRotation(45, interpolation=InterpolationMode.BILINEAR, expand=True),
                transforms.CenterCrop(target_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomPerspective(0.2),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.126, contrast=0.2)
                ], p=0.5),
                transforms.ToTensor(),
                imagenet_normalize
            ]),
        'val':
            transforms.Compose([
                enlarge,
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                imagenet_normalize
            ]),
        'test':
            transforms.Compose([
                enlarge,
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                imagenet_normalize
            ])
    }

    if phase in transform_dict:
        return transform_dict[phase]
    else:
        raise Exception("Unknown phase specified")


def build_dataset(datasource: Dict[str, str], root="", ext="*.png"):
    """
    Build dataset by consuming data from root/<datasource-key>

    :param datasource: the datasource dictionary that maps from label to path or list of paths
    eg. datasource = {
        'malignant': 'Malignant_Markers_Crop',
        'benign': 'Benign_Markers_Crop'
    }
    or
    datasource = {
        'malignant': ['path/a/malignant', 'path/b/malignant'],
        'benign': ['path/a/benign', 'path/b/benign']
    }
    :param root: the root path to be prepended to datasource-key, default is emptu
    :param ext: the file extension to search for
    :return: a dictionary of data split by corresponding label name e.g. { 'benign', 'malignant'}
    """
    datasets = {}
    for key in datasource:
        if isinstance(datasource[key], list):
            files = []
            for path in datasource[key]:
                files += glob.glob(os.path.join(root, path, ext))
            datasets[key] = files
        else:
            datasets[key] = glob.glob(os.path.join(root, datasource[key], ext))

    return datasets


def explain_dataset(ds):
    total_dataitem = 0
    s = ""
    for key in ds:
        path_count = len(ds[key])
        total_dataitem += path_count
        s += f'Found total {path_count} for {key}\n'
    s += f"Total dataitem: {total_dataitem}\n"
    return s


def build_train_validation_set(datasource, val_size, root="", ext="*.png"):
    """
    This function loop over each datasource's key and append it to <root> to make a search path to scan for
    files with given extension. The list of file then, will be splitted into training and validation set.

    The difference to `build_dataset` is the train/validation split(In fact, this method call build_dataset
    to begin with, then loop over keys to split train/validation data)

    :param datasource: dictionary with key as the class of data, and value is an image path
    :param val_size: validation size, must be greater than total datasource size of each class's dataset
    :param root: the root path to be prepended to datasource, default is empty
    :param ext: the file extension to search for
    :return a dictionary of data split by corresponding class name e.g. { 'benign', 'malignant'}
    """
    # Build dataset according to datasource dict[class: path]. e.g. { 'benign': 'benign/folder' }
    datasets = build_dataset(datasource, root=root, ext=ext)

    training_set = {}
    validation_set = {}
    total_training_set = 0
    total_validation_set = 0

    total_dataset = 0
    for key in datasets:    # for each class
        ds_size = len(datasets[key])  # size of each dataset
        total_dataset += ds_size

        assert val_size < ds_size, f"The size of '{key}'({ds_size}) is smaller than the specific val_size({val_size})"
        training_set[key] = datasets[key][val_size:]  # Cut from val_size-th onward
        t_size = len(training_set[key])
        total_training_set += t_size
        #print(f'Total train {key} size = {t_size}/{ds_size} ({t_size / ds_size:0.2f})')

        validation_set[key] = datasets[key][:val_size]  # Validation data is the first n elements from the dataset
        v_size = len(validation_set[key])
        total_validation_set += v_size
        #print(f'Total validation {key} size = {v_size}/{ds_size} ({v_size / ds_size:0.2f})')

    return training_set, validation_set



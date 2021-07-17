import os
from PIL import Image
import torchvision.transforms as transforms
import glob
__package__ = ['thyroid_dataset']

# imagenet mean and std
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


##################################
# transform in dataset to target size
##################################
def get_transform(target_size, phase='train'):
    """
    Pre-defined transformation pipe for the dataset
    :param target_size: tuple of (W,H) result image from the pipe
    :param phase: train/val/test phase of different transformation e.g. test will not need RandomCrop
    :return: a transformation function to target_size
    """


    # enlarge 10% bigger for the later cropping
    enlarge = transforms.Resize(size=(int(target_size[0] * 1.1), int(target_size[1] / 1.1))),

    # ImageNet normalizer
    imagenet_normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

    transform_dict = {
        'train':
            transforms.Compose([
                enlarge,
                transforms.RandomRotation(90, interpolation=Image.BILINEAR),
                transforms.RandomCrop(target_size),
                transforms.RandomHorizontalFlip(0.5),
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
        raise Exception("Unknown pharse specified")


def build_train_validation_set(datasource, val_size, root="", ext="*.png"):
    """
    :param datasource: dictionary with key as label of data, and value is a list of image path
    :param val_size: validation size, must be greater than total datasource size of each class's dataset
    :param root: the root path to be prepended to datasource, default is emptu
    :param ext: the file extension to search for
    :return a dictionary of data split by corresponding class name e.g. { 'benign', 'malignant'}
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

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from . import build_train_validation_set, get_transform


class ThyroidDataset(Dataset):
    """
    Dataset for Thyroid Image
    """
    datasource_root = 'thyroid-ds3'  # base path to be prepend to each data source
    datasource_paths = {
        'malignant': 'Malignant_Markers_Crop',
        'benign': 'Benign_Markers_Crop'
    }

    training_params = {
        'val_size': 24,
        'test_size': 24
    }

    def __init__(self, phase, target_size):
        self.phase = phase
        train, val = build_train_validation_set(
            ThyroidDataset.datasource_paths,
            val_size=ThyroidDataset.training_params['val_size'],
            root=ThyroidDataset.datasource_root
        )

        if phase == 'train':
            self.dataset = sum(train.values(), [])  # monoid flatten
        elif phase == 'val':
            self.dataset = sum(val.values(), [])
        elif phase == 'test':
            self.dataset = sum(val.values(), [])

        if type(target_size) is int:
            target_size = (target_size, target_size)

        assert type(target_size) is tuple, "target_size must be tuple of (W:int, H:int) or int if square is needed"

        self.transform = get_transform(target_size, phase)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> T_co:
        return self.dataset[index]

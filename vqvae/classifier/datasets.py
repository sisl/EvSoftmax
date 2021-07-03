# Code heavily based on: https://github.com/ritheshkumar95/pytorch-vqvae

import os
import csv
import torch.utils.data as data
from PIL import Image
from pathlib import Path

def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Borrowed from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class MiniImagenet(data.Dataset):

    def __init__(self, root, train=False, valid=False, test=False,
                 transform=None, target_transform=None, download=False, dataset='miniimagenet'):
        super(MiniImagenet, self).__init__()
        self.root = root
        self.train = train
        self.valid = valid
        self.test = test
        self.transform = transform
        self.target_transform = target_transform
        self.encoding = {}
        self.base_folder = get_project_root()
        if not (((train ^ valid ^ test) ^ (train & valid & test))):
            raise ValueError('One and only one of `train`, `valid` or `test` '
                'must be True (train={0}, valid={1}, test={2}).'.format(train,
                valid, test))

        if dataset == 'miniimagenet':
            self.image_folder = os.path.join(self.base_folder, os.path.expanduser(root), 'images')
            if train:
                filename = 'train_small.csv'
                print("yes train!", train)
            elif valid:
                filename = 'val_small.csv'
            elif test:
                filename = 'val_small.csv'
            else:
                raise ValueError('Unknown split.')
        elif dataset == 'miniimagenetgenerated':
            self.image_folder = os.path.join(self.base_folder, os.path.expanduser(root))
            filename = 'miniimagenet_generated.csv'


        self.split_filename = os.path.join(self.base_folder, os.path.expanduser(root), filename)
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use `download=True` '
                               'to download it')

        # Extract filenames and labels
        self._data = []
        with open(self.split_filename, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip the header
            for line in reader:
                self._data.append(tuple(line))
        self._fit_label_encoding()


    def _fit_label_encoding(self):
        _, labels = zip(*self._data)
        unique_labels = sorted(set(labels), key=labels.index)
        self.class_to_idx = dict((label, idx)
            for (idx, label) in enumerate(unique_labels))
        print("encoder:", self.class_to_idx)

    def __getitem__(self, index):
        filename, label = self._data[index]
        image = pil_loader(os.path.join(self.image_folder, filename))
        label = self.class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def _check_exists(self):
        print(self.split_filename, os.path.exists(self.split_filename))
        return (os.path.exists(self.image_folder) 
            and os.path.exists(self.split_filename))

    def __len__(self):
        return len(self._data)
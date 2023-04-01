import csv
import os
from typing import TYPE_CHECKING

import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader
if TYPE_CHECKING:
    from yaecs import Configuration


class BaselineDataset(Dataset):
    """
    Baseline pytorch dataset for the competition data.
    Note that it does not leverage mask information, augment the data nor normalise it in any way.
    """

    def __init__(self, config: 'Configuration'):
        data_directory = os.path.dirname(os.path.abspath(config.csv_path))
        self.s1_folder = os.path.join(data_directory, "s1")
        self.s2_folder = os.path.join(data_directory, "s2")
        self.masks_folder = os.path.join(data_directory, "s2-mask")

        with open(config.csv_path) as path_list:
            self.data_paths = [[os.path.basename(path) for path in row] for row in csv.reader(path_list, delimiter=",")][1:]

        self.device = config.device
        self.mode = config.mode

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index: int) -> dict:
        """ Reads a sample from the disk and returns it in the form of a dictionary. """

        paths = self.data_paths[index]
        s1 = [load_image(os.path.join(self.s1_folder, i)) for i in paths]
        if self.mode == "infer":
            s2 = [load_image(os.path.join(self.s2_folder, i)) for i in paths[:-1]]
            masks = [load_image(os.path.join(self.masks_folder, i)) for i in paths[:-1]]
            label = None
            label_mask = None
        else:
            s2 = [load_image(os.path.join(self.s2_folder, i)) for i in paths]
            masks = [load_image(os.path.join(self.masks_folder, i)) for i in paths]
            label = s2.pop(-1)
            label_mask = masks.pop(-1)

        return {
            "paths": paths,
            "s1": torch.from_numpy(np.concatenate(s1, axis=-1)).to(self.device),
            "s2": torch.from_numpy(np.concatenate(s2, axis=-1)).to(self.device),
            "masks": torch.from_numpy(np.concatenate(masks, axis=-1)).to(self.device),
            "label": torch.from_numpy(label).to(self.device) if label is not None else [0],
            "label_mask": torch.from_numpy(label_mask).to(self.device) if label_mask is not None else [0],
        }


def get_loader(config: 'Configuration') -> DataLoader:
    """ Get a loader for the data for batching and shuffling. """
    dataset = BaselineDataset(config)
    return DataLoader(dataset,
                      batch_size=config.batch_size,
                      shuffle=config.shuffle_data,
                      num_workers=config.num_workers)


def load_image(path: str) -> np.ndarray:
    """ Loads TIFF images and returns a numpy array with shape (256, 256, channels). """
    image = io.imread(path)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    return image

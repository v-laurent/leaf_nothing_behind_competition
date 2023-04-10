import csv
import os
from typing import TYPE_CHECKING

import numpy as np
from skimage import io
import torch
from torch.utils.data import (
    Dataset, 
    DataLoader, 
    random_split
)
from torchvision.transforms import (
    Compose, 
    RandomRotation, 
    Normalize 
)

if TYPE_CHECKING:
    from yaecs import Configuration

CORRUPTED_CLASSES = [0,1,7,8,9]
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

class S1ToS2Dataset(Dataset):

    def __init__(self, config: 'Configuration', transform=None):
        data_directory = os.path.dirname(os.path.abspath(config.csv_path))
        self.s1_folder = os.path.join(data_directory, "s1")
        self.s2_folder = os.path.join(data_directory, "s2")
        self.masks_folder = os.path.join(data_directory, "s2-mask")
        
        with open(config.csv_path) as path_list:
            self.data_paths = [[os.path.basename(path) for path in row] for row in csv.reader(path_list, delimiter=",")][1:]

        self.device = config.device
        self.mode = config.mode
        self.transform = transform

    def __len__(self):
        if self.mode == "infer":
            return len(self.data_paths)
        else:
            return len(self.data_paths)*2

    def __getitem__(self, index: int) -> dict:
        if self.mode == "infer":
            paths = self.data_paths[index]
            s1 = [load_image(os.path.join(self.s1_folder, i)) for i in paths]
            s2 = [load_image(os.path.join(self.s2_folder, i)) for i in paths[:-1]]
        else:
            paths = self.data_paths[index//2]
            s1 = load_image(os.path.join(self.s1_folder, paths[index%2])) 
            s2 = load_image(os.path.join(self.s2_folder, paths[index%2]))
            mask = load_image(os.path.join(self.masks_folder, paths[index%2]))
            mask = np.where(np.isin(mask, CORRUPTED_CLASSES),0,1)
            image = torch.from_numpy(np.concatenate([s1,s2, mask], axis=-1).T)
            if self.transform:
                image = self.transform(image)

            #output format C,H,W    
            return {
                "paths": paths,
                "X": image[:2,:,:].float().to(self.device),
                "y": torch.unsqueeze(image[2,:,:], dim=0).float().to(self.device),
                'mask': torch.unsqueeze(image[3,:,:], dim=0).float().to(self.device)
            }
             
def get_dataset(config: 'Configuration') -> Dataset:
    if config.model == "S1ToS2Model":
        max_rotation = 0 if config.mode == "infer_on_train" else 30
        return S1ToS2Dataset(
            config, 
            transform=Compose([
                Normalize(
                    mean=[-17.97, -10.70, 1.45, 0], 
                    std=[3.53, 2.77, 1.50, 1]
                ),
                RandomRotation(degrees=max_rotation),
            ])
        )
    else:
        return BaselineDataset(config)
        
        
def get_loader(config: 'Configuration') -> DataLoader:
    dataset = get_dataset(config)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
                        train_dataset,
                        batch_size=config.batch_size,
                        shuffle=config.shuffle_data,
                        num_workers=config.num_workers
                    )
    val_dataloader = DataLoader(
                        val_dataset,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers
                    )
    return train_dataloader, val_dataloader


def load_image(path: str) -> np.ndarray:
    """ Loads TIFF images and returns a numpy array with shape (256, 256, channels). """
    image = io.imread(path)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    return image

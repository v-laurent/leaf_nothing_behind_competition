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
class WholeDataset(Dataset):

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
        return len(self.data_paths)

    def __getitem__(self, index: int) -> dict:
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
            masks = np.where(np.isin(masks, CORRUPTED_CLASSES),0,1)
            
            image = torch.from_numpy(np.concatenate([s1[2], *s2, *masks], axis=-1).T)
            if self.transform:
                image = self.transform(image)

        #output format C,H,W  
        return {
            "paths": paths,
            "S1_t0": image[:2,:,:].float().to(self.device),
            "S2_and_Mask_t2": torch.cat([
                    torch.unsqueeze(image[2,:,:], dim=0), 
                    torch.unsqueeze(image[5,:,:], dim=0)
                ],axis=0).float().to(self.device),
            "S2_and_Mask_t1": torch.cat([
                    torch.unsqueeze(image[3,:,:], dim=0), 
                    torch.unsqueeze(image[6,:,:], dim=0)
                ],axis=0).float().to(self.device),
            "y":  torch.unsqueeze(image[4,:,:], dim=0).float().to(self.device),
            "y_mask":  torch.unsqueeze(image[7,:,:], dim=0).float().to(self.device),
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
            #TODO: not working yet
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
    elif config.model == "Unet":
        max_rotation = 0 if config.mode == "infer_on_train" else 30
        return WholeDataset(
            config, 
            transform=Compose([
                Normalize(
                    mean=[-18.22, -10.90, 1.41, 1.34, 1.30, 0, 0, 0], 
                    std=[3.61, 2.81, 1.41, 1.47, 1.32, 1, 1, 1]
                ),
                RandomRotation(degrees=max_rotation),
            ])
        )
    else:
         raise ValueError(f"Unknown model '{config.model}'.")
        
        
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

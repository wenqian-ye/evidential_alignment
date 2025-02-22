
import os
import pandas as pd
from pathlib import Path
from utils import log
from .register import register_dataset
import torch
import numpy as np
from PIL import Image
import warnings

def generate_metadata_imagenetbg(data_path):
    log("Generating metadata for ImagenetBG...")
    bg_dir = Path(data_path)
    dirs = {
        'train': 'in9l/train',
        'val': 'in9l/val',
        'test': 'original/val',
        'mixed_rand': 'mixed_rand/val',
        'only_fg': 'only_fg/val',
        'no_fg': 'no_fg/val',
    }
    classes = {
        0: 'dog',
        1: 'bird',
        2: 'wheeled vehicle',
        3: 'reptile',
        4: 'carnivore',
        5: 'insect',
        6: 'musical instrument',
        7: 'primate',
        8: 'fish'
    }

    all_data = []
    for dir in dirs:
        for label in classes:
            label_folder = f'0{label}_{classes[label]}'
            folder_path = bg_dir/dirs[dir]/label_folder
            for img_path in folder_path.glob('*.JPEG'):
                all_data.append({
                    'split': dir,
                    'filename': str(img_path.relative_to(bg_dir)),
                    'y': label,
                    'a': 0
                })

    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(bg_dir, "metadata.csv"), index=False)
    return df

@register_dataset("imagenet-bg")
class ImageNet9BG(torch.utils.data.Dataset):
    def __init__(
        self,
        basedir,
        split,
        transform=None,
        sel_indexes=None,
        seed=None
    ):
        metadata_path = os.path.join(basedir, "metadata.csv")
        # if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        # else:
        #     metadata_df = generate_metadata_imagenetbg(basedir)
        
        self.metadata_df = metadata_df[metadata_df["split"] == split]
        total = len(metadata_df)
        split_total = len(self.metadata_df)

        if sel_indexes is not None:
            if seed is None:
                raise ValueError("seed is not specified")
            self.metadata_df = self.metadata_df.iloc[sel_indexes]
            ratio = len(sel_indexes) / split_total
            assert ratio <= 1.0, "incorrect sel_indexes" 
            log(f"{split} ({ratio*100:.2f}%): {len(self.metadata_df)} ({len(self.metadata_df)/split_total*100:.2f}%)")
        else:
            log(f"{split}: {split_total} ({split_total/total*100:.2f}%)")
        self.y_array = self.metadata_df["y"].values
        self.filename_array = self.metadata_df["filename"].values
        self.n_classes = np.unique(self.y_array).size

        self.basedir = basedir
        self.transform = transform
        self.group_array = self.y_array
        self.confounder_array = self.y_array
        self.split = split


    def __getitem__(self, index):
        path, target = self.filename_array[index], self.y_array[index]
        img_path = os.path.join(self.basedir, path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
    
        return img, target, target, target
      

    def __len__(self):
        return len(self.y_array)
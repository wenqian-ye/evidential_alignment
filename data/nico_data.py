import os


import numpy as np
import json
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from .register import register_dataset, register_transform
from torch import nn
import pandas as pd
from PIL import Image
import warnings
from config import NICO_DATA_FOLDER, NICO_CXT_DIC_PATH, NICO_CLASS_DIC_PATH
from utils import log

# https://github.com/Wangt-CN/CaaM
# https://drive.google.com/drive/folders/17-jl0fF9BxZupG75BtpOqJaB6dJ2Pv8O?usp=sharing
TRAINING_DIST = {'dog': ['on_grass', 'in_water', 'in_cage', 'eating', 'on_beach', 'lying', 'running'],
                 'cat': ['on_snow', 'at_home', 'in_street', 'walking', 'in_river', 'in_cage', 'eating'],
                 'bear': ['in_forest', 'black', 'brown', 'eating_grass', 'in_water', 'lying', 'on_snow'],
                 'bird': ['on_ground', 'in_hand', 'on_branch', 'flying', 'eating', 'on_grass', 'standing'],
                 'cow': ['in_river', 'lying', 'standing', 'eating', 'in_forest', 'on_grass', 'on_snow'],
                 'elephant': ['in_zoo', 'in_circus', 'in_forest', 'in_river', 'eating', 'standing', 'on_grass'],
                 'horse': ['on_beach', 'aside_people', 'running', 'lying', 'on_grass', 'on_snow', 'in_forest'],
                 'monkey': ['sitting', 'walking', 'in_water', 'on_snow', 'in_forest', 'eating', 'on_grass'],
                 'rat': ['at_home', 'in_hole', 'in_cage', 'in_forest', 'in_water', 'on_grass', 'eating'],
                 'sheep': ['eating', 'on_road', 'walking', 'on_snow', 'on_grass', 'lying', 'in_forest']}


def prepare_metadata():
    cxt_dic = json.load(open(NICO_CXT_DIC_PATH, 'r'))
    class_dic = json.load(open(NICO_CLASS_DIC_PATH, 'r'))
    cxt_index2name = {i: n for n, i in cxt_dic.items()}
    class_index2name = {i: n for n, i in class_dic.items()}

    labels = []
    contexts = []
    context_names = []
    label_names = []
    file_names = []
    splits = []
    for split_id, split in enumerate(["train", "val", "test"]):
        all_file_name = os.listdir(os.path.join(NICO_DATA_FOLDER, split))
        for file_name in all_file_name:
            label, context, index = file_name.split('_')
            file_names.append(os.path.join(split, file_name))
            contexts.append(int(context))
            context_names.append(cxt_index2name[int(context)])
            label_names.append(class_index2name[int(label)])
            labels.append(int(label))
            splits.append(split_id)

    labels_unique = sorted(list(set(labels)))
    contexts_unique = sorted(list(set(contexts)))
    label2unique = {l: i for i, l in enumerate(labels_unique)}
    context2unique = {c: i for i, c in enumerate(contexts_unique)}
    uniquelabel2name = {
        label2unique[l]: class_index2name[l] for l in labels_unique}
    uniquecontext2name = {
        context2unique[c]: cxt_index2name[c] for c in contexts_unique}

    name2uniquelabel = {n: l for l, n in uniquelabel2name.items()}
    name2uniquecontext = {n: c for c, n in uniquecontext2name.items()}

    meta_data_path = os.path.join(NICO_DATA_FOLDER, "metadata.csv")
    with open(meta_data_path, "w") as f:
        f.write("img_id,img_filename,y,label_name,split,context,context_name\n")
        for i in range(len(file_names)):
            file_name = file_names[i]
            label = label2unique[labels[i]]
            label_name = label_names[i]
            split_id = splits[i]
            context = context2unique[contexts[i]]
            context_name = context_names[i]
            f.write(
                f"{i},{file_name},{label},{label_name},{split_id},{context},{context_name}\n")
    return meta_data_path

def prepare_metadata_v2():
    select_dict = {'bear':['on_tree', 'white'],             # [val, test]
               'bird':['on_shoulder', 'in_hand'], 
               'cat':['on_tree', 'in_street'], 
               'cow':['spotted', 'standing'], 
               'dog':['running', 'in_street'], 
               'elephant':['in_circus', 'in_street'], 
               'horse':['running', 'in_street'], 
               'monkey':['climbing', 'sitting'], 
               'rat':['running', 'in_hole'], 
               'sheep':['at_sunset', 'on_road']}
    nico_metadata = prepare_metadata()
    dir_path = os.path.dirname(nico_metadata)
    new_nico_metadata = os.path.join(dir_path, "sel_metadata.csv")
    context_pos = 6
    label_name_pos = 3
    split_pos = 4
    with open(new_nico_metadata, "w") as fout:
        with open(nico_metadata, "r") as f:
            for i,line in enumerate(f):
                if i == 0:
                    fout.write(line)
                    continue
                eles = line.split(',')
                ctx = eles[context_pos].strip()
                label_name = eles[label_name_pos].strip()
                if select_dict[label_name][0] == ctx: # for validation
                    eles[split_pos] = str(1)
                elif select_dict[label_name][1] == ctx: # for test
                    eles[split_pos] = str(2)
                else:
                    eles[split_pos] = str(0)
                fout.write(','.join(eles))

def get_transform_nico(train, augment_data=True):
    mean = [0.52418953, 0.5233741, 0.44896784]
    std = [0.21851876, 0.2175944, 0.22552039]
    if train and augment_data:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform

@register_dataset("nico")
class NICO_dataset(Dataset):
    def __init__(self, basedir, split, transform=None, sel_indexes=None, seed=None):
        super(NICO_dataset, self).__init__()
        assert split in ["train", "val", "test"], f"invalida split = {split}"
        self.basedir = basedir
        metadata_df = pd.read_csv(os.path.join(basedir, "sel_metadata.csv"))
        total = len(metadata_df)
        split_info = metadata_df["split"].values
        print(len(metadata_df))
        split_i = ["train", "val", "test"].index(split)
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        split_total = len(self.metadata_df)
        if sel_indexes is not None:
            if seed is None:
                raise ValueError("seed is not specified")
            ratio = len(sel_indexes) / len(self.metadata_df)
            assert ratio <= 1.0, "incorrect sel_indexes" 
            self.metadata_df = self.metadata_df.iloc[sel_indexes]
            log(f"{split} ({ratio*100:.2f}%): {len(self.metadata_df)} ({len(self.metadata_df)/split_total*100:.2f}%)")

        self.y_array = self.metadata_df["y"].values
        sel_indexes = self.metadata_df["img_id"].values
        labelnames = self.metadata_df["label_name"].values
        self.labelname2index = {}
        for i in range(len(self.y_array)):
            self.labelname2index[labelnames[i]] = self.y_array[i]

        self.confounder_array = self.metadata_df["context"].values
        contextnames = self.metadata_df["context_name"].values
        self.contextname2index = {}
        for i in range(len(self.confounder_array)):
            self.contextname2index[contextnames[i]] = self.confounder_array[i]
        self.filename_array = self.metadata_df["img_filename"].values
       
        print(len(self.y_array))
        self.n_classes = np.unique(self.y_array).size
        self.n_places = np.unique(self.confounder_array).size
        
        self.group_array = (
            self.y_array * self.n_places + self.confounder_array
        ).astype("int")
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
            (
                torch.arange(self.n_groups).unsqueeze(1)
                == torch.from_numpy(self.group_array)
            )
            .sum(1)
            .float()
        )

        self.transform = transform
    

   

    def __getitem__(self, idx):
        img_path = os.path.join(self.basedir, self.filename_array[idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        y = self.y_array[idx]
        p = self.confounder_array[idx]
        g = self.group_array[idx]

        return img, y, g, p
        

    def __len__(self):
        return len(self.y_array)

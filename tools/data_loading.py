# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import sys
sys.path.append(os.path.join(__file__, "../../research"))
from skimage import io
import numpy as np
from torch.utils.data import Dataset
from config import Config


def get_name_of_classes():
    """
    Read names of classes from file
    Returns
    -------
        list of names of classes
    """
    with open('../config.cfg') as f:
        cfg = Config(f)
    with open(cfg.list_of_classes) as f:
       names_of_classes = f.readlines()
    names_of_classes = [name[0:len(name) - 1] for name in names_of_classes]
    names_of_classes.sort()
    return names_of_classes


class NewDataset(Dataset):

    def __init__(self, root_dir, transforms=None):
        np.random.seed(100)
        self.transform = transforms
        self.all_data_table = []
        path_func = np.vectorize(os.path.join)
        temp = os.listdir(root_dir)
        subdirs = path_func(root_dir, temp)
        self.names_of_classes = np.array(get_name_of_classes())
        self.samples_per_class = {}
        for class_label in self.names_of_classes:
            self.samples_per_class[class_label] = 0
        for subdir in subdirs:
            for class_label in self.names_of_classes:
                class_path = os.path.join(subdir, class_label)
                file_names = np.array(os.listdir(class_path))
                np.random.shuffle(file_names)
                file_names = file_names[:file_names.shape[0] // 5]
                triplets = []
                for file_name in file_names:
                    idx, = np.where(self.names_of_classes == class_label)
                    idx = idx[0]
                    triplets.append((os.path.join(class_path, file_name), idx))
                    self.samples_per_class[class_label] += 1.0
                self.all_data_table = self.all_data_table + triplets



    def __len__(self):
        return len(self.all_data_table)

    def __getitem__(self, idx):
        color_image = io.imread(self.all_data_table[idx][0])
        sample = {'color': color_image,
                  'label': self.all_data_table[idx][1]}
        if self.transform:
           sample = self.transform(sample)
        return sample

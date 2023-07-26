import torch
from torch.utils.data import Dataset as DS
import pandas as pd
import numpy as np
import os

from helpers.data import get_label, read_pts_file, assign_labels


class Dataset(DS):
    def __init__(self, root_folder, suffix, transform=None, classes_to_use=None, data_augmentation=False):
        super().__init__()
        if classes_to_use is None:
            classes_to_use = ['cube', 'pyramid']
        self.root_folder = root_folder
        self.suffix = suffix
        self.data_augmentation = data_augmentation
        if 'all' in classes_to_use:
            self.classes_to_use = os.listdir(root_folder)
        else:
            self.classes_to_use = classes_to_use
        self.transform = transform
        #if self.suffix == '.pts':
        self.labels = assign_labels(root_folder)
        # Find all files in the root folder with the given suffix
        self.filepaths = []
        for subdir, dirs, files in os.walk(self.root_folder):
            for file in files:
                if file.endswith(self.suffix) and subdir.split('/')[-1] in self.classes_to_use:
                    self.filepaths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        if 'csv' in self.suffix:
        # Load data from file
            arr = np.array(pd.read_csv(filepath))
        elif 'pts' in self.suffix:
            arr = read_pts_file(filepath)
        else:
            raise NotImplementedError("This data type is not implemented at the moment.")
        #target = (self.highest_shape, arr.shape[-1])
        #padding = [(0, target[0] - arr.shape[0]), (0, 0)]
        #padded_arr = np.pad(arr, padding)
        # Process data into a tensor
        #tensor_data = torch.tensor(padded_arr, dtype=torch.float)

        # Save the tensor to a folder
        #save_folder = os.path.join(self.root_folder, 'processed_data')
        #if not os.path.exists(save_folder):
        #    os.makedirs(save_folder)
        #save_filepath = os.path.join(save_folder, f'{self.filename}_{index}.pt')
        #torch.save(tensor_data, save_filepath)

        cls = os.path.dirname(filepath).split('/')[-1]
        if cls in self.classes_to_use:
            label = self.labels[cls]
        else:
            raise NotImplementedError("This label is not included in the current dataset.")

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            arr[:, [0, 2]] = arr[:, [0, 2]].dot(rotation_matrix)  # random rotation
            arr += np.random.normal(0, 0.02, size=arr.shape)  # random jitter

        sample = {'pc': arr,
                  'label': np.array(label),
                  'path': filepath}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


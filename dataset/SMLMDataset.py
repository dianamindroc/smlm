import torch
from torch.utils.data import Dataset as DS
import pandas as pd
import numpy as np
import os

class Dataset(DS):
    def __init__(self, root_folder, filename, suffix):
        super().__init__()
        self.root_folder = root_folder
        self.filename = filename
        self.suffix = suffix

        # Find all files in the root folder with the given suffix
        self.filepaths = []
        for subdir, dirs, files in os.walk(self.root_folder):
            for file in files:
                if file.endswith(self.suffix):
                    self.filepaths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        filepath = self.filepaths[index]

        # Load data from file
        arr = np.array(pd.read_csv(filepath))

        # Process data into a tensor
        tensor_data = torch.tensor(arr, dtype=torch.float)

        # Save the tensor to a folder
        save_folder = os.path.join(self.root_folder, 'processed_data')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_filepath = os.path.join(save_folder, f'{self.filename}_{index}.pt')
        torch.save(tensor_data, save_filepath)

        return tensor_data
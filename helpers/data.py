import glob
import os
import torch.nn.functional as F
import pandas as pd

def add_folder_label_to_csv(csv_folder, folder1_name, folder2_name):
    for filename in os.listdir(csv_folder):
        file_path = os.path.join(csv_folder, filename)

        if os.path.isfile(file_path) and filename.endswith('.csv'):
            df = pd.read_csv(file_path)
            folder_name = os.path.basename(os.path.dirname(file_path))

            if folder_name == folder1_name:
                df['label'] = 0
            elif folder_name == folder2_name:
                df['label'] = 1

            labeled_file_path = os.path.join(csv_folder, f'labeled_{filename}')
            df.to_csv(labeled_file_path, index=False)

            print(f'Label added to {filename} and saved as labeled_{filename}.')


def get_label(name, classes):
    if classes[0] == name:
        label = 0
    else:
        label = 1
    return label


def pad_tensor(current_shape, target_shape):

    # Calculate the amount of padding needed along the second dimension
    padding = target_shape[0] - current_shape.size(0)

    # Pad the tensor with zeros along the second dimension
    padded_tensor = F.pad((0, 0, 0, padding, 0, 0), )

    return padded_tensor


def get_highest_shape(root_folder, classes):
    highest = 0
    for cl in classes:
        folder = os.path.join(root_folder, cl)
        files = os.listdir(folder)
        for file in files:
            if len(pd.read_csv(os.path.join(folder, file))) > highest:
                highest = len(pd.read_csv(os.path.join(folder, file)))
    return highest

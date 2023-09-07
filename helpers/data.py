import glob
import os

import numpy as np
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


def assign_labels(root_folder):
    classes = {}
    class_label = 0

    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            classes[folder] = class_label
            class_label += 1

    return classes


def pad_tensor(current_shape, target_shape):

    # Calculate the amount of padding needed along the second dimension
    padding = target_shape[0] - current_shape.size(0)

    # Pad the tensor with zeros along the second dimension
    padded_tensor = F.pad((0, 0, 0, padding, 0, 0), )

    return padded_tensor


def get_highest_shape(root_folder, classes):
    #csv_file = os.path.join(os.path.dirname(os.path.abspath(os.curdir)), 'resources', 'highest_shape.csv')
    csv_file = os.path.join('/home/squirrel/coding/smlm', 'resources', 'highest_shape.csv')
    dataset_name = os.path.basename(root_folder)
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        try:
            highest = df.loc[df['dataset'] == dataset_name, 'highest_shape']
            return int(highest)
        except:
            print('Dataset not found in the CSV file.')
    else:
        print(f"CSV file '{csv_file}' does not exist.")

    highest = 0
    if 'all' in classes:
        cls = os.listdir(root_folder)
    else:
        cls = classes
    for cl in cls:
        folder = os.path.join(root_folder, cl)
        files = os.listdir(folder)
        for file in files:
            if len(pd.read_csv(os.path.join(folder, file))) > highest:
                highest = len(pd.read_csv(os.path.join(folder, file)))
    update_csv(csv_file, dataset_name, highest)
    return highest



def read_pts_file(file_path):
    with open(file_path, 'r') as file:
        # Skip the first two lines (header information)
        next(file)
        next(file)

        # Read the remaining lines and extract the coordinates
        points = []
        for line in file:
            x, y, z = map(float, line.split())
            points.append([x, y, z])

    return np.array(points)


import pandas as pd

def update_csv(csv_file, dataset_name, highest_shape):
    data = {'dataset': [dataset_name],
            'highest_shape': [highest_shape]}

    df = pd.DataFrame(data)

    if not pd.api.types.is_string_dtype(df['dataset']):
        df['dataset'] = df['dataset'].astype(str)

    if not pd.api.types.is_numeric_dtype(df['highest_shape']):
        df['highest_shape'] = df['highest_shape'].astype(int)

    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False)
    else:
        existing_df = pd.read_csv(csv_file)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_csv(csv_file, index=False)

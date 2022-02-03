"""
Class for reading raw localization data
"""

import numpy as np
import pandas as pd
import os

class Reader():
    def __init__(self, folder_path):
        """
        Initialize the class.
        :param folder_path: path to main folder containing data
        """
        self.path = folder_path

    def get_folders(self, type = 'all'):
        """
        Gets folders containing data based on the desired type (2D, 3D, all)
        :param type: string mentioning the type of data that we want to read from the folder, either '2D' or '3D' or 'all'
        :return: returns a list with folders
        """
        self.folders = [file for file in os.listdir(self.path) if type in file]
        return self.folders

    def get_files_from_folder(self, folder_number = None, path = None):
        """
        Gets the files inside the folder, either directly from a path or from a folder number from the get_folder method.
        :param folder_number: int mentioning which folder from the folder list (from get_folder method) if path is None
        :param path: string, path to folder to read files from if folder_number is None
        :return: returns a list with the files
        """
        if folder_number is None:
            if path is None:
                raise ValueError(f"Please enter a path or a folder number")
            self.files = os.listdir(path)
        else:
            self.files = os.listdir(os.path.join(self.path,self.folders[folder_number]))
        return self.files

    def set_file(self, file):
        """
        Set the file to be used for further operations
        :param file: string, path to the file
        :return: None
        """
        self.file = file

    def read_csv(self):
        """
        Reads a .csv file containing raw localization data
        :return: returns pandas DataFrame
        """
        data = pd.read_csv(self.file)
        return data

    def read_txt(self):
        """
        Reads a .txt file containing raw localization data. Converts it to a DataFrame.
        :return: returns pandas DataFrame
        """
        data = pd.read_csv(self.file, skiprows=[0])
        data = data[data.columns[0].str.split(' ', 4, expand=True)]
        return data

    def get_xyz_array(self, df):
        """
        Converts the df to a numpy array containing only the x,y,z coordinates.
        :param df: DataFrame to be converted to array
        :return: returns the numpy array
        """
        arr = np.array(df[['x','y','z']])
        return arr

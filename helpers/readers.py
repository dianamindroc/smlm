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

    def get_folders(self, type : str = 'all') -> str:
        """
        Gets folders containing data based on the desired type (2D, 3D, all)
        :param type: string mentioning the type of data that we want to read from the folder, either '2D' or '3D' or 'all'
        :return: returns a list with folders
        """
        if type == 'all':
            self.folders = [os.listdir(self.path)]
        else:
            self.folders = [file for file in os.listdir(self.path) if type in file]
        return self.folders

    def set_folder(self, folder):
        """
        Sets working folder
        :param folder:
        :return: None
        """
        self.folder = folder

    def get_files_from_folder(self, folder_number : int = None, path : str = None) -> list:
        """
        Gets the files inside the folder, either directly from a path or from a folder number from the get_folder method.
        :param folder_number: int mentioning which folder from the folder list (from get_folder method) if path is None
        :param path: path to folder to read files from if folder_number is None
        :param type: string to mention the extension of files to get from folder, e.g. images .png or localizations .csv
        :return: returns a list with the files
        """
        if folder_number is None:
            if path is None:
                raise ValueError(f"Please enter a path or a folder number")
            self.files = [file for file in os.listdir(self.path)]
        else:
            self.files = os.listdir(os.path.join(self.path,self.folders[folder_number]))
        self.set_folder(os.path.join(self.path, self.folders[folder_number]))
        return self.files

    def set_file(self, file):
        """
        Set the file to be used for further operations
        :param file: path to the file
        :return: None
        """
        self.file = file

    def filter(self, type : str) -> list:
        """
        Filters the files in the folder based on the desired type (.png, .csv, .txt, ...)
        :param type: extension of the desired file type
        :return: filtered list of files
        """
        if type is None:
            return self.files
        else:
            self.files = [file for file in self.files if file.endswith(type)]

    def read_csv(self) -> pd.DataFrame:
        """
        Reads a .csv file containing raw localization data and renames columns to unified version
        :return: returns pandas DataFrame
        """
        data = pd.read_csv(os.path.join(self.folder, self.file))
        self.data = data
        return data

    def read_txt(self) -> pd.DataFrame:
        """
        Reads a .txt file containing raw localization data. Converts it to a DataFrame and renames columns to unified version
        :return: returns pandas DataFrame
        """
        data = pd.read_csv(os.path.join(self.path, self.file, skiprows=[0]))
        data = data[data.columns[0].str.split(' ', 4, expand=True)]
        self.data = data
        return data

    def get_xyz_array(self) -> np.array:
        """
        Converts the df to a numpy array containing only the x,y,z coordinates.
        :param df: DataFrame to be converted to array
        :return: returns the numpy array
        """
        arr = np.array(self.data[['x','y','z']])
        self.array = arr
        return arr

"""
Class for reading raw localization data
"""

import pandas as pd
import os


class Reader():
    def __init__(self, folder_path):
        """
        Initialize the class.
        :param folder_path: path to main folder containing data
        """
        self.path = folder_path

    def get_folders(self, which: str = 'all') -> str:
        """
        Gets folders containing data based on the desired type (2D, 3D, all)
        :param which: string mentioning the type of data that we want to read from the folder, either '2D' or '3D' or 'all'
        :return: returns a list with folders
        """
        if which == 'all':
            self.folders = [file for file in os.listdir(self.path)]
        else:
            self.folders = [file for file in os.listdir(self.path) if which in file]
        return self.folders

    def set_folder(self, folder):
        """
        Sets working folder
        :param folder:
        :return: None
        """
        self.folder = folder

    def get_files_from_folder(self, folder_number: int = None, path: str = None) -> list:
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
            self.files = os.listdir(os.path.join(self.path, self.folders[folder_number]))
        self.set_folder(os.path.join(self.path, self.folders[folder_number]))
        return self.files

    def set_file(self, file):
        """
        Set the file to be used for further operations
        :param file: path to the file
        :return: None
        """
        self.file = file

    def filter(self, which: str) -> list:
        """
        Filters the files in the folder based on the desired type (.png, .csv, .txt, ...)
        :param which: extension of the desired file type
        :return: filtered list of files
        """
        if which is None:
            return self.files
        else:
            self.files = [file for file in self.files if file.endswith(which)]

    def read_csv(self) -> pd.DataFrame:
        """
        Reads a .csv file containing raw localization data
        #TODO: rename columns to unified version
        :return: returns pandas DataFrame
        """
        data = pd.read_csv(os.path.join(self.folder, self.file))
        self.data = data
        return data

    def read_txt(self, columns: int) -> pd.DataFrame:
        """
        Reads a .txt file containing raw localization data. Converts it to a DataFrame
        :param columns: the number of columns to split the text file into
        #TODO: rename columns to unified version
        :return: returns pandas DataFrame
        """
        data = pd.read_csv(os.path.join(self.folder, self.file), skiprows=[0])
        data = data[data.columns[0]].str.split(' ', columns, expand=True)
        self.data = data
        return data

    def extract_xyz(self, column_names: list) -> pd.DataFrame:
        """
        Extracts a data frame from the working data frame containing only the x,y,z coordinates.
        :param column_names: list which explains the names of the original columns
        :return: data frame
        """
        df = pd.DataFrame()
        df[['x', 'y', 'z']] = self.data[column_names].astype(float)
        self.df_xyz = df
        return self.df_xyz

    def extract_xy(self, column_indices: list) -> pd.DataFrame:
        """
        Extract a data frame from the working data frame containing only the x and y coordinates.
        :param column_indices: list containing int that are the column indices containing x and y coordinates in original data frame.
        :return: data frame
        """
        df = pd.DataFrame()
        df[['x', 'y']] = self.data[column_indices].astype(float)
        self.df_xy = df
        return self.df_xy
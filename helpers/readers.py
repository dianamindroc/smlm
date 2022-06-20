import pandas as pd
import os


class Reader:
    """
    Class used to read raw localization data from .csv or .txt files in a systematic manner.
    """
    def __init__(self, folder_path: str):
        """
        Initialize the class.
        :param folder_path: path to main folder containing data
        """
        self.path = folder_path
        self.folder = self.path
        self.folders = []
        self.files = []
        self.data = pd.DataFrame()
        self.file = ''
        self.df_xyz = pd.DataFrame()
        self.df_xy = pd.DataFrame()

    def get_folders(self, which: str = 'all') -> [str]:
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

    def set_folder(self, folder: int = None):
        """
        Sets the working folder
        :param folder: number of folder inside the main folder; if None, then the main folder is
        considered the working folder.
        :return: None
        """
        if folder is None:
            print('Working folder will be the main folder.')
        else:
            self.folder = os.path.join(self.path, self.folders[folder])

    def get_files_from_folder(self, folder_number: int = None, path: str = None) -> [str]:
        """
        Gets the files inside the folder, either directly from a path or from a folder number from the get_folder method.
        :param folder_number: int mentioning which folder from the folder list (from get_folder method) if path is None
        :param path: absolute path to folder to read files from if folder_number is None
        :return: returns a list with the files
        """
        if folder_number is None:
            if path is None:
                if self.folder is None:
                    raise ValueError(f"Please enter a path or a folder number")
                else:
                    self.files = os.listdir(self.folder)
            else:
                self.files = os.listdir(path)
                self.folder = path
        else:
            self.files = os.listdir(os.path.join(self.path, self.folders[folder_number]))
            self.set_folder(folder_number)
        return self.files

    def set_file(self, file: int):
        """
        Set the file to be used for further operations
        :param file: int to mention which file from files list to set as working file.
        :return: None
        """
        self.file = os.path.join(self.folder, self.files[file])

    def filter(self, which: str) -> [str]:
        """
        Filters the files in the folder based on the desired type (.png, .csv, .txt, ...)
        :param which: extension of the desired file type
        :return: filtered list of files
        """
        if which is None:
            return self.files
        else:
            self.files = [file for file in self.files if file.endswith(which)]
            return self.files

    def read_csv(self) -> pd.DataFrame:
        """
        Reads a .csv file containing raw localization data
        :return: pandas DataFrame
        """
        data = pd.read_csv(os.path.join(self.folder, self.file))
        self.data = data
        return data

    def read_txt(self, columns: int) -> pd.DataFrame:
        """
        Reads a .txt file containing raw localization data. Converts it to a DataFrame
        :param columns: the number of columns to split the text file into
        #TODO: rename columns to unified version
        :return: pandas DataFrame
        """
        data = pd.read_csv(os.path.join(self.folder, self.file), skiprows=[0])
        data = data[data.columns[0]].str.split(' ', columns, expand=True)
        self.data = data
        return data

    def extract_xyz(self, column_names: list, save: bool = False) -> pd.DataFrame:
        """
        Extracts a data frame from the working data frame containing only the x,y,z coordinates.
        :param column_names: list which explains the names of the original columns which contain x,y,z coordinates.
        :return: pandas DataFrame
        """
        df = pd.DataFrame()
        df[['x', 'y', 'z']] = self.data[column_names].astype(float)
        self.df_xyz = df
        if save:
            self.df_xyz.to_csv(os.path.join(self.path, self.folder, self.file, '_xyz.csv'), columns=['x', 'y', 'z'],
                               sep=';', index=False)
        return self.df_xyz

    def extract_xy(self, column_indices: list, save: bool = False) -> pd.DataFrame:
        """
        Extract a data frame from the working data frame containing only the x and y coordinates.
        :param column_indices: list containing int that are the column indices containing x and y coordinates
        in original data frame.
        :param save: boolean indicating whether saving of the extracted data frame is desired. if True,
        :return: pandas DataFrame
        """
        df = pd.DataFrame()
        df[['x', 'y']] = self.data[column_indices].astype(float)
        self.df_xy = df
        if save:
            self.df_xy.to_csv(os.path.join(self.path, self.folder, self.file, '_xyz.csv'), columns=['x', 'y'],
                              sep=';', index=False)
        return self.df_xy

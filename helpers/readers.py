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

def read_mat_file(file, save = False, path=None):
    # this is only for the Heydarian dataset
    import scipy
    data = scipy.io.loadmat(file)
    list = []
    for i, npc in enumerate(data['particles'][0]):
        npc_to_save = npc[0][0][0]
        list.append(npc_to_save)
        if save:
            df = pd.DataFrame(npc_to_save, columns=['x','y','z'])
            name = 'npc_' + str(i)
            if path is None:
                print('Path was not set. Saving in repository home directory.')
                path = os.path.join(os.path.abspath(os.curdir), 'NPCs')
                os.mkdir(path)
            else:
                if not os.path.exists(path):
                    os.mkdir(path)
            df.to_csv(os.path.join(path,name) + '.csv', index=False)
    return list

class ReaderIMODfiles:
    """
    Class for extracting files containing point clouds generated with IMOD.
    """
    def __init__(self, path_folder : str, suffix : str = 'Localizations.txt'):
        self.suffix = suffix
        if 'train' in os.listdir(path_folder):
            self.path_folder = path_folder
            self.path_folder1 = os.path.join(path_folder, 'train')
            self.read(self.path_folder1)
            self.path_folder2 = os.path.join(path_folder, 'test')
            self.read(self.path_folder2)
        else:
            self.path_folder = path_folder
            self.read(self.path_folder)

    def read(self, path):
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                self._process_subfolders(full_path)


    def _process_subfolders(self, folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(self.suffix):
                df = pd.read_csv(os.path.join(folder_path,file), sep=' ')
                df.rename(columns={
                    'Pos_x': 'x',
                    'Pos_y': 'y',
                    'Pos_z': 'z'
                    # Add more columns as needed
                }, inplace=True)
                name = folder_path.split('/')[-1] + '.csv'
                save_path = os.path.join(self.path_folder, name)
                df.to_csv(save_path, index=False)

def convert_ply_to_csv(path, save = False):
    import open3d as o3d
    pcs = []
    #check if path is file or folder
    if os.path.isdir(path):
        for file in os.listdir(path):
            pc = o3d.io.read_point_cloud(os.path.join(path, file))
            df = pd.DataFrame(pc.points)
            if save:
                df.to_csv(os.path.join(path, file.split('.')[0] + '.csv'), index=False)
            else:
                pcs.append(df)
        return pcs
    else:
        pc = o3d.io.read_point_cloud(path)
        df = pd.DataFrame(pc.points)
        if save:
            df.to_csv(path.split('.')[0]+'.csv', index=False)
        else:
            return df






import os
import warnings

import numpy as np
import pandas as pd

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


def read_mat_file(file, save=False, path=None):
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


def convert_ply_to_csv(path, save=False):
    """
    Function to convert .ply files to .csv files. If it is a path, it processes all files.
    :param path: Path to data.
    :param save: boolean to decide if to save the df. if True, saves the df in the same folder as .csv
    :return: pandas DataFrame
    """
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


def read_pc(path_or_arr):
    """
    Quick script for reading a point cloud either from path (.csv or .ply) or directly array
    :param path_or_arr: path to point cloud or array
    :return: array containing point cloud data
    """
    import numpy as np
    if isinstance(path_or_arr, str) and path_or_arr.endswith('.csv'):
        return np.array(pd.read_csv(path_or_arr)[['x', 'y', 'z']])
    elif isinstance(path_or_arr, np.ndarray):
        return path_or_arr
    elif isinstance(path_or_arr, str) and path_or_arr.endswith('.ply'):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path_or_arr)
        return np.array(pcd.points)


def read_by_sample(path, sample):
    """
    Read point clouds (ground truth and respective input and output) by sample number in the specified path.
    :param path: path to read point clouds from.
    :param sample: int representing sample number in the specified path.
    :return: tuple containing ground truth, input and output with the specified number
    """
    import os
    import glob
    sample_int = int(sample)
    sample_str = str(sample_int)
    all_files = glob.glob(os.path.join(path, f'*{sample_str}*'))

    def matches_sample(fname):
        token = os.path.basename(fname).split('_')[0]
        try:
            return int(token) == sample_int
        except ValueError:
            return token == sample_str

    samples = [f for f in all_files if matches_sample(f)]

    gt = next((read_pc(s) for s in samples if 'gt' in s), None)
    inputt = next((read_pc(s) for s in samples if 'input' in s), None)
    output = next((read_pc(s) for s in samples if 'output' in s), None)
    attn = next((np.load(s)[0] for s in samples if 'attn' in s), None)
    attn_value = None if attn is None else np.array(attn)
    return {'gt':gt,
            'input':inputt,
            'output':output,
            'gtdf': pd.DataFrame(gt, columns=['x', 'y', 'z']),
            'inputdf': pd.DataFrame(inputt, columns=['x', 'y', 'z']),
            'outputdf':pd.DataFrame(output, columns=['x', 'y', 'z']),
            'attn':attn_value}


def read_samples(
    path,
    sample_numbers,
    suppress_force_all_finite_warning: bool = True,
):
    """
    Convenience wrapper to read several samples while optionally suppressing
    scikit-learn's FutureWarning about ``force_all_finite``.

    Args:
        path (str): directory containing the samples.
        sample_numbers (Iterable[int]): sample ids to load.
        suppress_force_all_finite_warning (bool): whether to silence the noisy warning.
    """
    if suppress_force_all_finite_warning:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*force_all_finite.*",
                category=FutureWarning,
            )
            return [read_by_sample(path, sample) for sample in sample_numbers]
    return [read_by_sample(path, sample) for sample in sample_numbers]


def read_pts_file(file_path):
    """
    Read a .pts file and return the points as a numpy array.
    :param file_path: Path to the .pts file
    Returns:
    numpy.ndarray: Array of points, shape (n_points, n_dimensions)
    """
    import numpy as np
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


def remove_half_plane(pcd, axis='x', keep_positive=True):
    import open3d as o3d
    import numpy as np
    """
    Remove half of a point cloud along a specified axis.

    Args:
    pcd (o3d.geometry.PointCloud): Input point cloud
    axis (str): Axis along which to cut ('x', 'y', or 'z')
    keep_positive (bool): If True, keep the positive half; if False, keep the negative half

    Returns:
    o3d.geometry.PointCloud: Point cloud with half removed
    """
    points = np.asarray(pcd.points)

    if axis.lower() == 'x':
        axis_index = 0
    elif axis.lower() == 'y':
        axis_index = 1
    elif axis.lower() == 'z':
        axis_index = 2
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    # Find the median along the specified axis
    median = np.median(points[:, axis_index])

    if keep_positive:
        mask = points[:, axis_index] >= median
    else:
        mask = points[:, axis_index] < median

    # Create a new point cloud with the selected half
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(points[mask])

    # Set the color of all points to light blue
    num_points = len(new_pcd.points)
    light_blue = np.array([0.6, 0.8, 1.0])  # RGB values for light blue
    light_blue_colors = np.tile(light_blue, (num_points, 1))
    new_pcd.colors = o3d.utility.Vector3dVector(light_blue_colors)

    return new_pcd

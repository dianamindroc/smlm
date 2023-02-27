# Script to run preprocessing of the data

import argparse
from preprocessing.preprocessing import Preprocessing
import timeit

if __name__ == '__main__':
    print('\n--------------------')
    print('Preprocessing SMLM data')
    parser = argparse.ArgumentParser(description='Preprocessing of SMLM Data')
    parser.add_argument('-d', '--data', type=str, help='Path to folder containing train and test data (expects train and test folders).', required=True)
    parser.add_argument('-ds', '--downsample', type=int, help='Specify factor to downsample by.', default=100)
    parser.add_argument('-dn', '--denoise', help='Specify whether to denoise the data. Default false.')
    parser.add_argument('-suf', '--suffix', help='Specify the suffix of the data', default='.txt')
    parser.add_argument('-n', '--name', help='Specify the base name of the files to search for', default='Localizations')
    args = parser.parse_args()
    path = args.data
    start_time = timeit.default_timer()
    prep = Preprocessing(path)
    prep.downsample(args.downsample)
    if args.denoise:
        prep.denoise()
        # prep.get_graphs()
    stop_time = timeit.default_timer()
    print('Data was preprocessed. It can be found in the preprocessed folder.')
    print('--------------------')

import argparse
import os
from helpers.pc_stats import load_and_analyze_point_clouds


# Script to quickly get the statistics of a dataset
def get_stats(directory_path, suffix='.csv', cluster_size=25, save=False):
    """
    Load point cloud data and compute statistics.
    :param directory_path: path to directory of point clouds
    :param suffix: suffix of file, default .csv
    :param cluster_size: size of cluster for HDBSCAN
    :return stats: dictionary of statistics
    """
    stats = load_and_analyze_point_clouds(directory_path, suffix, cluster_size)
    # If save=True, save the stats as a JSON file
    if save:
        import json
        save_path = os.path.join(directory_path, 'stats.json')

        # Save the stats dictionary as a JSON file
        with open(save_path, 'w') as json_file:
            json.dump(stats, json_file, indent=4)

    print(f"Statistics saved to {save_path}")
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get statistics from point clouds.')
    parser.add_argument('directory', type=str, help='Directory containing point clouds.')
    parser.add_argument('-s', '--suffix', type='str', help='Suffix to append to filenames.', default='.csv')
    parser.add_argument('-c','--cluster_size', type='int', help='Cluster size.', default=25)
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print('Directory {} does not exist.'.format(args.directory))
        exit(1)

    pc_stats = get_stats(args.directory, args.suffix, args.cluster_size)

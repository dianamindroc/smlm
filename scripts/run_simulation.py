# Script to run simulations from the command line

import argparse
from simulation.simulate_suresim_data import simulate
import yaml
import timeit

if __name__ == '__main__':
    print('\n--------------------')
    print('Simulating SMLM data')
    parser = argparse.ArgumentParser(description='SMLM simulator using SuReSIM')
    parser.add_argument('--config', type=str, help='Path to configuration file (yaml)', required=True)
    parser.add_argument('-s', '--samples', help='Number of desired simulation samples', default=15)
    parser.add_argument('-t', '--technique', help='Microscopy technique (choose from dstorm, palm. Default dstorm.)',
                        default='dstorm')
    parser.add_argument('-f', '--frames', help='Number of frames that would be recorded and then fitted. Default random choice.', default='random')
    parser.add_argument('-p', '--split', help='Split the data into 80/20 train and test folders. Default true.', default=True)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    samples = int(args.samples)
    start_time = timeit.default_timer()
    simulate(config, samples, args.technique, args.frames, args.split)
    stop_time = timeit.default_timer()
    print('Simulation finished. Simulation time of ',samples, 'samples was ',stop_time-start_time)
    print('--------------------')


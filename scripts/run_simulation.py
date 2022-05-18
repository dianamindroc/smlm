# Script to run simulations from the command line

import argparse
from simulator.simulate_data import simulate
import yaml

if __name__ == '__main__':
    print('\n--------------------')
    print('Simulating SMLM data')
    parser = argparse.ArgumentParser(description='SMLM simulator using SuReSIM')
    parser.add_argument('--config', type=str, help='Path to configuration file (yaml)', required=True)
    parser.add_argument('-s', '--samples', help='Number of desired simulation samples', default=15)
    parser.add_argument('-t', '--technique', help='Microscopy technique (choose from dstorm, palm. Default dstorm.)',
                        default='dstorm')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    samples = int(args.samples)
    simulate(config, samples, args.technique)
    print('Simulation finished.')
    print('--------------------')
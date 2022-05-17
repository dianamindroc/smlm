import argparse
from simulator.simulate_data import simulate
import yaml

if __name__ == '__main__':
    print('\n--------------------')
    print('Simulating SMLM data')
    parser = argparse.ArgumentParser(description='SMLM simulator')
    parser.add_argument('--config', type=str, help='Path to configuration file', required=True)
    parser.add_argument('-s', '--samples', help='Number of simulation samples', default=15)
    parser.add_argument('-t', '--technique', help='Microscopy technique (choose from dstorm, palm)', default='dstorm')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    samples = int(args.samples)
    simulate(config, samples, args.technique)
    print('Simulation finished.')
    print('--------------------')
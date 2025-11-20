import argparse
from train_pocafoldas import train


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run PocaFoldAS training")

    # Add arguments
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--fixed_alpha', type=float, default=0.001, help='Fixed alpha value (default: 0.001)')

    # Parse arguments
    args = parser.parse_args()

    # Run the training function with the parsed arguments
    train(args.config, exp_name=args.exp_name, fixed_alpha=args.fixed_alpha)


if __name__ == "__main__":
    main()

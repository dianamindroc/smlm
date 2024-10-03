import argparse
import os
import json
from simulation.simulate_dna_origami import SMLMDnaOrigami


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate DNA Origami Samples for SMLM")

    # Add command-line arguments
    parser.add_argument('-s', '--struct_type', type=str, required=True,
                        choices=['cube', 'pyramid', 'tetrahedron', 'sphere'],
                        help="Type of DNA origami structure to simulate (cube, pyramid, tetrahedron, sphere)")

    parser.add_argument('-n', '--number_samples', type=int, required=True,
                        help="Number of DNA origami samples to simulate")

    parser.add_argument('-stats', '--stats_file', type=str, required=False,
                        help="Path to a JSON file containing stats data")

    parser.add_argument('-rot', '--apply_rotation', type=bool, required=False, default=False,
                        help="Flag to apply random rotation to the model structure")

    # Parse command-line arguments
    args = parser.parse_args()

    # Check if stats file exists
    if args.stats_file:
        print(f"Error: Stats file {args.stats_file} does not exist.")
        return

    # Load stats from the provided file (assuming it is in JSON format)
    if args.stats_file:
        with open(args.stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = None

    # Instantiate the SMLMDnaOrigami class with the provided arguments
    dna_origami_simulator = SMLMDnaOrigami(
        struct_type=args.struct_type,
        number_dna_origami_samples=args.number_samples,
        stats=stats,
        apply_rotation=args.apply_rotation
    )

    # Run the simulation
    dna_origami_simulator.generate_all_dna_origami_smlm_samples()


if __name__ == "__main__":
    main()

"""
main.py
=======
Purpose : Command-line entry point for the CE 4011 frame analysis program.
          Uses matrix_lib (pure-Python) for all matrix operations.

Usage
-----
  python main.py <input_file> [output_file]

Inputs  : input_file  (str) – structured text file (see read_input.py)
          output_file (str, optional) – path to save results
Outputs : printed results; optional text file
"""

import sys
import os

# Ensure the Assignment2 directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

from read_input    import read_input
from frame_analysis import solve_frame
from write_output  import write_output


def main():
    """
    Main entry point.
    Inputs  : sys.argv  (command-line arguments)
    Outputs : None  (prints results to stdout, optionally writes file)
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_file> [output_file]")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(input_file):
        print(f"ERROR: Input file '{input_file}' not found.")
        sys.exit(1)

    print(f"\nReading: {input_file}")
    data = read_input(input_file)
    print(f"  Nodes: {data['NumNode']},  Elements: {data['NumElem']}")

    print("Solving...")
    result = solve_frame(data)
    print(f"  Free DOFs: {result['NumEq']}")
    print(f"  SkylineMatrix storage: {result['K_skyline'].storage_size()} entries")

    write_output(result, data, output_file)


if __name__ == '__main__':
    main()

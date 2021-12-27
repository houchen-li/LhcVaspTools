#!/usr/bin/env python

import argparse
from LhcVaspTools.BasicUtils import EnergyBands


def parseArgv() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="This script is used to plot bands")
    parser.add_argument("input_file_name", nargs="?", type=str,
                        help="input hdf5 file.")
    parser.add_argument("-o", "--output-file", nargs="?", type=str,
                        dest="output_file_name", required=True, help="output file.")
    options: argparse.Namespace = parser.parse_args()
    return options


def main() -> int:
    options: argparse.Namespace = parseArgv()
    input_file_name: str = options.input_file_name
    output_file_name: str = options.output_file_name
    energy_bands: EnergyBands = EnergyBands()
    energy_bands.readFile(input_file_name)
    energy_bands.plotFigure(output_file_name)
    return 0


if __name__ == "__main__":
    main()

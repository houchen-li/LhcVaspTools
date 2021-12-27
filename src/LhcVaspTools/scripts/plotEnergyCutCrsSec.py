#!/usr/bin/env python

import argparse
from LhcVaspTools.BasicUtils import EnergyCutCrsSec


def parseArgv() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="This script is used to produce Fermi surface cross section.")
    parser.add_argument("input_file_name", nargs="?", type=str, help="input hdf5 file.")
    parser.add_argument("-o", "--output-file", nargs="?", type=str, dest="output_file_name",
                        help="output file.")
    options: argparse.Namespace = parser.parse_args()
    return options


def main() -> int:
    options: argparse.Namespace = parseArgv()
    input_file_name: str = options.input_file_name
    output_file_name: str = options.output_file_name
    energy_cut_crs_sec: EnergyCutCrsSec = EnergyCutCrsSec()
    energy_cut_crs_sec.readFile(input_file_name)
    energy_cut_crs_sec.plotFigure(output_file_name)
    return 0


if __name__ == "__main__":
    main()

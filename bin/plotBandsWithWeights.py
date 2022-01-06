#!/usr/bin/env python

import argparse
from LhcVaspTools.BasicUtils import readDataFromJson
from LhcVaspTools.PhaseExts import EnergyBandsWithWeights


def parseArgv() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="This script is used to plot bands")
    parser.add_argument("input_file_name", nargs="?", type=str,
                        help="input hdf5 file.")
    parser.add_argument("-o", "--output-file", nargs="?", type=str,
                        dest="output_file_name", required=True, help="output file.")
    parser.add_argument('-af', '--atomic-orbits-file', nargs='?', type=str,
                        dest='atomic_orbits_file_name', help='atomic orbits file.')
    options: argparse.Namespace = parser.parse_args()
    return options


def main() -> int:
    options: argparse.Namespace = parseArgv()
    input_file_name: str = options.input_file_name
    output_file_name: str = options.output_file_name
    atomic_orbits_file_name: str = options.atomic_orbits_file_name
    if atomic_orbits_file_name is None:
        atomic_orbits: dict = None
    else:
        atomic_orbits: dict = readDataFromJson(atomic_orbits_file_name)
    energy_bands_with_weights: EnergyBandsWithWeights = EnergyBandsWithWeights()
    energy_bands_with_weights.readFile(input_file_name)
    energy_bands_with_weights.plotFigure(output_file_name, ylim=[-15., 15.], atomic_orbits=atomic_orbits)
    return 0


if __name__ == "__main__":
    main()

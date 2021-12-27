#!/usr/bin/env python

import argparse
from LhcVaspTools.OamExts import EnergyBandsWithOam


def parseArgv() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="This script is used to plot bands")
    parser.add_argument("input_file_name", nargs="?", type=str,
                        help="input hdf5 file.")
    parser.add_argument("-o", "--output-file", nargs="?", type=str,
                        dest="output_file_name", required=True, help="output figure file.")
    parser.add_argument('-c', '--component', nargs="?", type=str,
                        choices=['Lx', 'Ly', 'Lz'], dest='component', required=True,
                        help='the OAM component to plot.')
    options: argparse.Namespace = parser.parse_args()
    return options


def main() -> int:
    options: argparse.Namespace = parseArgv()
    input_file_name: str = options.input_file_name
    output_file_name: str = options.output_file_name
    component: str = options.component
    energy_bands_with_oam: EnergyBandsWithOam = EnergyBandsWithOam()
    energy_bands_with_oam.readFile(input_file_name)
    energy_bands_with_oam.ylim = (-3., 3.)
    energy_bands_with_oam.plotFigure(output_file_name, component)
    return 0


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
from LhcVaspTools.BasicUtils import readDataFromJson
from LhcVaspTools.OamExts import EnergyBandsWithOam


def parseArgv() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="This script is used to plot bands")
    parser.add_argument("input_file_name", metavar='INPUT_FILE_NAME',
                        nargs="?", type=str, help="input hdf5 file.")
    parser.add_argument("-o", "--output-file", nargs="?", type=str,
                        dest="output_file_name", required=True, help="output figure file.")
    parser.add_argument('-c', '--component', nargs="?", type=str,
                        choices=['Lx', 'Ly', 'Lz'], dest='component', required=True,
                        help='the OAM component to plot.')
    parser.add_argument('-bf', '--band-indices-file', nargs='?', type=str,
                        dest='band_indices_file_name', help='band indices file name.')
    parser.add_argument('-xl', '--xlim', nargs=2, type=float,
                        dest='xlim', help='xlim for the bands plot.')
    parser.add_argument('-yl', '--ylim', nargs=2, type=float,
                        dest='ylim', default=[-2., 1.], help='ylim for the bands plot.')
    options: argparse.Namespace = parser.parse_args()
    return options


def main() -> int:
    options: argparse.Namespace = parseArgv()
    input_file_name: str = options.input_file_name
    output_file_name: str = options.output_file_name
    component: str = options.component
    band_indices_file_name: str = options.band_indices_file_name
    xlim: list = options.xlim
    ylim: list = options.ylim
    if band_indices_file_name is None:
        band_indices: list = None
    else:
        band_indices: list = readDataFromJson(band_indices_file_name)
    energy_bands_with_oam: EnergyBandsWithOam = EnergyBandsWithOam()
    energy_bands_with_oam.readFile(input_file_name)
    energy_bands_with_oam.plotFigure(output_file_name, component,
                                     xlim=xlim, ylim=ylim, band_indices=band_indices)
    return 0


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
from LhcVaspTools.BasicUtils import readDataFromJson
from LhcVaspTools.PhaseExts import EnergyBandsWithWeights


def parseArgv() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="This script is used to plot bands")
    parser.add_argument("input_file_name", metavar='INPUT_FILE_NAME',
                        nargs="?", type=str, help="input hdf5 file.")
    parser.add_argument("-o", "--output-file", nargs="?", type=str,
                        dest="output_file_name", required=True, help="output file.")
    parser.add_argument('-cf', '--config-file', nargs='?', type=str,
                        dest='config_file_name', help='config file.')
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
    config_file_name: str = options.config_file_name
    xlim: list = options.xlim
    ylim: list = options.ylim
    if config_file_name is None:
        config: object = None
    else:
        config: object = readDataFromJson(config_file_name)
    energy_bands_with_weights: EnergyBandsWithWeights = EnergyBandsWithWeights()
    energy_bands_with_weights.readFile(input_file_name)
    energy_bands_with_weights.plotFigure(output_file_name, xlim=xlim, ylim=ylim, config=config)
    return 0


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
from numpy import ndarray
from LhcVaspTools.BasicUtils import saveData2Json, Vaspdata


def parseArgv() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=
                                                              'This script is used to determine the indices of cross \
                                                              bands out of \"vaspout.h5\".')
    parser.add_argument('input_file_name', nargs='?', type=str,
                        help='input \"vaspout.h5\" file.')
    parser.add_argument('-o', '--output-file', nargs='?', type=str,
                        dest='output_file_name', help="output file.")
    parser.add_argument('-e', '--energy-level', nargs='?', type=float,
                        dest='energy_level', required=True, help='the energy level where bands cross.')
    options: argparse.Namespace = parser.parse_args()
    return options


def main() -> int:
    options: argparse.Namespace = parseArgv()
    input_file_name: str = options.input_file_name
    output_file_name: str = options.output_file_name
    energy_level: float = options.energy_level
    vasp_data: Vaspdata = Vaspdata()
    vasp_data.readFile(input_file_name)
    cross_bands: list = Vaspdata.deterCrossBands(vasp_data, energy_level).tolist()
    if output_file_name is None:
        print('cross_bands:\t' + str(cross_bands))
    else:
        saveData2Json(cross_bands, output_file_name)
    return 0


if __name__ == "__main__":
    main()

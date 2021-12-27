#!/usr/bin/env python

import argparse
from LhcVaspTools.BasicUtils import saveData2Json, Vaspdata


def parseArgv() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=
                                                              'This script is used to extract Fermi\
                                                              energy out of \"vaspout.h5\"')
    parser.add_argument('input_file_name', nargs='?', type=str, help='input \"vaspout.h5\" file.')
    parser.add_argument('-o', '--output-file', nargs='?', type=str, dest='output_file_name',
                        help="output file.")
    options: argparse.Namespace = parser.parse_args()
    return options


def main() -> int:
    options: argparse.Namespace = parseArgv()
    input_file_name: str = options.input_file_name
    output_file_name: str = options.output_file_name
    vasp_data: Vaspdata = Vaspdata()
    vasp_data.readFile(input_file_name)
    if output_file_name is None:
        print('efermi:\t{0:.6f}'.format(vasp_data.efermi))
    else:
        saveData2Json(vasp_data.efermi, output_file_name)
    return 0


if __name__ == "__main__":
    main()

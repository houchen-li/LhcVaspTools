#!/usr/bin/env python

import argparse
from LhcVaspTools.BasicUtils import readDataFromJson, Vaspdata, EnergyBands


def parseArgv() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="This script is used to extract bands from \"vaspout.h5\".")
    parser.add_argument("input_file_name", nargs="?", type=str, help="input \"vaspout.h5\" file.")
    parser.add_argument("-o", "--output-file", nargs="?", type=str, dest="output_file_name",
                        help="output file.")
    parser.add_argument("-ef", "--efermi", nargs="?", type=float, dest="efermi",
                        required=True, help="Fermi energy from static computing.")
    parser.add_argument("-xl", "--xticklabels-file", nargs="?", type=str, dest="xticklabels_file_name",
                        help="xtick labels file")
    options: argparse.Namespace = parser.parse_args()
    return options


def main() -> int:
    options: argparse.Namespace = parseArgv()
    input_file_name: str = options.input_file_name
    output_file_name: str = options.output_file_name
    efermi: float = options.efermi
    xticklabels_file_name: str = options.xticklabels_file_name
    if xticklabels_file_name is not None:
        xticklabels: list = readDataFromJson(xticklabels_file_name)
    else:
        xticklabels: list = None
    vasp_data = Vaspdata()
    vasp_data.readFile(input_file_name)
    energy_bands: EnergyBands = EnergyBands()
    energy_bands.loadVaspdata(vasp_data)
    energy_bands.efermi = efermi
    energy_bands.xticklabels = xticklabels
    energy_bands.saveFile(output_file_name)
    return 0


if __name__ == "__main__":
    main()

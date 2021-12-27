#!/usr/bin/env python

import argparse
from LhcVaspTools.BasicUtils import readDataFromJson, Vaspdata, EnergyCutCrsSec


def parseArgv() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="This script is used to plot Fermi surface cross section.")
    parser.add_argument("input_file_name", nargs="?", type=str, help="input hdf5 file.")
    parser.add_argument("-o", "--output-file", nargs="?", type=str,
                        dest="output_file_name", required=True, help="output file.")
    parser.add_argument('-e', '--energy-level', nargs='?', type=float,
                        dest='energy_level', help='the energy level where bands cross.')
    parser.add_argument("-cb", "--cross-bands-file", nargs="?", type=str,
                        dest="cross_bands_file_name", required=True, help="cross bands file.")
    parser.add_argument('-Nx', '--num-of-xgrids', nargs='?', type=int,
                        dest="num_of_xgrids", help="number of x grids.")
    parser.add_argument('-Ny', '--num-of-ygrids', nargs='?', type=int,
                        dest="num_of_ygrids", help="number of y grids.")
    options: argparse.Namespace = parser.parse_args()
    return options


def main() -> int:
    options: argparse.Namespace = parseArgv()
    input_file_name: str = options.input_file_name
    output_file_name: str = options.output_file_name
    energy_level: float = options.energy_level
    cross_bands_file_name: str = options.cross_bands_file_name
    num_of_grids: list = [options.num_of_xgrids, options.num_of_ygrids]
    vasp_data: Vaspdata = Vaspdata()
    vasp_data.readFile(input_file_name)
    energy_cut_crs_sec: EnergyCutCrsSec = EnergyCutCrsSec()
    energy_cut_crs_sec.loadVaspdata(vasp_data, num_of_grids)
    energy_cut_crs_sec.energy_level = energy_level
    energy_cut_crs_sec.cross_bands = readDataFromJson(cross_bands_file_name)
    energy_cut_crs_sec.saveFile(output_file_name)
    return 0


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
from LhcVaspTools.BasicUtils import Vaspdata
from LhcVaspTools.OamExts import ElecDnstyCrsSecWithOam


def parseArgv() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="This script is used to plot Fermi surface cross section.")
    parser.add_argument("input_file_name", nargs="?", type=str,
                        help="input hdf5 file.")
    parser.add_argument("-o", "--output-file", nargs="?", type=str,
                        dest="output_file_name", required=True, help="output file.")
    parser.add_argument('-e', '--energy-level', nargs='?', type=float,
                        dest='energy_level', help='the energy level where bands cross.')
    parser.add_argument('-s', '--sigma', nargs='?', type=float,
                        dest="sigma", default=0.1, required=True, help="cross bands file.")
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
    sigma: float = options.sigma
    num_of_grids: list = [options.num_of_xgrids, options.num_of_ygrids]
    vasp_data: Vaspdata = Vaspdata()
    vasp_data.readFile(input_file_name)
    elec_dnsty_crs_sec_with_oam: ElecDnstyCrsSecWithOam = ElecDnstyCrsSecWithOam()
    elec_dnsty_crs_sec_with_oam.loadVaspdata(vasp_data, num_of_grids)
    elec_dnsty_crs_sec_with_oam.energy_level = energy_level
    elec_dnsty_crs_sec_with_oam.sigma = sigma
    elec_dnsty_crs_sec_with_oam.saveFile(output_file_name)
    return 0


if __name__ == "__main__":
    main()

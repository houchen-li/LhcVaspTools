#!/usr/bin/env python

import argparse
from LhcVaspTools.BasicUtils import readDataFromJson, ElecDnstyCrsSec


def parseArgv() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="This script is used to produce Fermi surface cross section.")
    parser.add_argument("input_file_name", nargs="?", type=str, help="input hdf5 file.")
    parser.add_argument("-o", "--output-file", nargs="?", type=str, dest="output_file_name", required=True,
                        help="output file.")
    parser.add_argument('-bf', '--band-indices-file', nargs='?', type=str, dest='band_indices_file_name',
                        help='band indices file.')
    options: argparse.Namespace = parser.parse_args()
    return options


def main() -> int:
    options: argparse.Namespace = parseArgv()
    input_file_name: str = options.input_file_name
    output_file_name: str = options.output_file_name
    band_indices_file_name: str = options.band_indices_file_name
    if band_indices_file_name is None:
        band_indices: list = None
    else:
        band_indices: list = readDataFromJson(band_indices_file_name)
    elec_dnsty_crs_sec: ElecDnstyCrsSec = ElecDnstyCrsSec()
    elec_dnsty_crs_sec.readFile(input_file_name)
    elec_dnsty_crs_sec.plotFigure(output_file_name, band_indices)
    return 0


if __name__ == '__main__':
    main()

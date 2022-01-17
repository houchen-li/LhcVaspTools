#!/usr/bin/env python

import argparse
import json
import numpy as np


def parseArgv() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='This script is used to generate KPOINTS file of a plane.')
    parser.add_argument('input_file_name', metavar='INPUT_FILE_NAME', nargs='?', type=str,
                        help='input config json file.')
    parser.add_argument('-o', '--output-file', nargs='?', type=str, dest='output_file_name',
                        help='output KPOINTS file.')
    options: argparse.Namespace = parser.parse_args()
    return options


def readDataFromJson(file_name: str) -> dict:
    with open(file_name, 'r') as f:
        data: dict = json.load(f)
    return data


def sliceDistance(coord:np.ndarray, num_of_grids):
    return np.linspace(coord[0], coord[1], num_of_grids)


def genKpointsPlane(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                    num_of_grids_AB: int=101, num_of_grids_AD: int=101) -> np.ndarray:
    lines_segment: np.ndarray = np.stack(
        [np.linspace(A, D, num_of_grids_AD), np.linspace(B, C, num_of_grids_AD)], axis=1)
    kpoints_plane: np.ndarray = np.apply_along_axis(sliceDistance, 1, lines_segment, num_of_grids_AB)
    kpoints_plane = kpoints_plane.reshape((-1, 3))
    return kpoints_plane


def saveKPoints2File(kpoints: np.ndarray, file_name: str) -> None:
    data: np.ndarray = np.insert(kpoints, 3, 1., axis=1)
    with open(file_name, 'w') as f:
        f.write("PdGa Fermi Surface (111)\n")
        f.write("{0:d}\n".format(np.shape(data)[0]))
        f.write("Reciprocal\n")
        np.savetxt(f, data, fmt='%1.6f', delimiter='\t')
    return


def main() -> int:
    options: argparse.Namespace = parseArgv()
    input_file_name: str = options.input_file_name
    output_file_name: str = options.output_file_name
    data: dict = readDataFromJson(input_file_name)
    A: np.ndarray = np.asarray(data['A'], dtype=np.float_)
    B: np.ndarray = np.asarray(data['B'], dtype=np.float_)
    C: np.ndarray = np.asarray(data['C'], dtype=np.float_)
    D: np.ndarray = np.asarray(data['D'], dtype=np.float_)
    num_of_grids_AB: int = int(data['num_of_grids_AB'])
    num_of_grids_AC: int = int(data['num_of_grids_AC'])
    kpoints_plane: np.ndarray = genKpointsPlane(A, B, C, D, num_of_grids_AB, num_of_grids_AC)
    saveKPoints2File(kpoints_plane, 'KPOINTS')
    return 0


if __name__ == "__main__":
    main()


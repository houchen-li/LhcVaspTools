#!/usr/bin/env python

import argparse

import numpy as np
from LhcVaspTools.BasicUtils import readDataFromJson, Vaspdata


def parseArgv() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='This script is used to calculate the weights of each band.')
    parser.add_argument('input_file_name', nargs='?', type=str, help='input \"vaspout.h5\" file.')
    parser.add_argument('-o', '--output-file', nargs='?', type=str, dest='output_file_name', help='output hdf5 file')
    parser.add_argument('-cf', '--config-file', nargs='?', type=str, dest='config_file_name', required=True,
                        help='config file')
    options: argparse.Namespace = parser.parse_args()
    return options


def selectWeights(vasp_data: Vaspdata, config: object) -> dict:
    atomic_orbits_indices: dict = {
        's': 0, 'py': 1, 'pz': 2, 'px': 3, 'dxy': 4, 'dyz': 5, 'dz2': 6, 'dxz': 7, 'dx2-y2': 8}
    atoms_indices = genAtomsIndices(vasp_data.types_of_ions, vasp_data.nums_of_each_type)
    selected_weights: dict = {}
    weights: np.ndarray = np.absolute(vasp_data.phase)
    if type(config) == list:
        if config[0] in atoms_indices.keys():
            for atomic_symbol in config:
                key: str = atomic_symbol
                value = np.sum(
                    weights[:, :, atoms_indices[atomic_symbol], :], axis=(1, 2)) / vasp_data.num_of_kpoints
                value = value.tolist()
                selected_weights[key] = value
        elif config[0] in atomic_orbits_indices.keys():
            for orbital_symbol in config:
                key: str = orbital_symbol
                value = np.sum(
                    weights[:, :, :, atomic_orbits_indices[orbital_symbol]], axis=(1, 2)) / vasp_data.num_of_kpoints
                value = value.tolist()
                selected_weights[key] = value
    elif type(config) == dict:
        for atomic_symbol in config.keys():
            if 'all' in config[atomic_symbol]:
                config[atomic_symbol] = ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2']
            for orbital_symbol in config[atomic_symbol]:
                key: str = '{0:s}: {1:s}'.format(atomic_symbol, orbital_symbol)
                value: np.ndarray = np.sum(
                    weights[:, :, atoms_indices[atomic_symbol],atomic_orbits_indices[orbital_symbol]], axis=1) \
                                    / vasp_data.num_of_kpoints
                value = value.tolist()
                selected_weights[key] = value
    return selected_weights


def saveWeights2File(selected_weights: dict, file_name) -> None:
    return


def genAtomsIndices(types_of_ions: list, nums_of_each_type: np.ndarray) -> dict:
    atoms_indices: dict = {}
    count: int = 0
    for ion_type, ion_nums in zip(types_of_ions, nums_of_each_type):
        for i in range(1, ion_nums + 1):
            key: str = '{1:d}{0:s}'.format(ion_type, i)
            value: int = count
            atoms_indices[key] = value
            count += 1
    del count
    return atoms_indices


def main() -> int:
    options: argparse.Namespace = parseArgv()
    input_file_name: str = options.input_file_name
    output_file_name: str = options.output_file_name
    config_file_name: str = options.config_file_name
    config: object = readDataFromJson(config_file_name)
    vasp_data: Vaspdata = Vaspdata()
    vasp_data.readFile(input_file_name)
    selected_weights: dict = selectWeights(vasp_data, config)
    saveData2Json(selected_weights, output_file_name)
    return 0


if __name__ == '__main__':
    main()

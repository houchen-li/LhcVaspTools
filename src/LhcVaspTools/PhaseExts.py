from __future__ import annotations

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection

from LhcVaspTools.BasicUtils import Int, Real, String, Array, List, Dict,\
    initFigure, Vaspdata, EnergyBands


class EnergyBandsWithWeights(EnergyBands):

    atomic_orbits_indices: Dict = {
        's': 0,
        'py': 1,
        'pz': 2,
        'px': 3,
        'dxy': 4,
        'dyz': 5,
        'dz2': 6,
        'dxz': 7,
        'dx2-y2': 8
    }

    def __init__(self, kpath: Array = None, eigenvalues: Array = None, weights: Array = None, efermi: Real = 0.,
                 xticklabels: List = None, xticks: Array = None, discontinued_indices: Array = None,
                 types_of_ions: List = None, nums_of_each_type: Array = None) -> None:
        super(EnergyBandsWithWeights, self).__init__(kpath, eigenvalues, efermi,
                                                     xticklabels, xticks, discontinued_indices)
        self._weights: Array = weights
        self._types_of_ions: List = types_of_ions
        self._nums_of_each_type: Array = nums_of_each_type
        return

    def readFile(self, file_name: String) -> None:
        EnergyBandsWithWeights.readDataFromFile(self, file_name)
        return

    @staticmethod
    def readDataFromFile(energy_bands_with_weights: EnergyBandsWithWeights, file_name: String) -> None:
        with h5.File(file_name, 'r') as f:
            grp: h5.Group = f['results/energy_bands_with_weights']
            EnergyBandsWithWeights.readDataFromH5grp(energy_bands_with_weights, grp)
        return

    @staticmethod
    def readDataFromH5grp(energy_bands_with_weights: EnergyBandsWithWeights, h5grp: h5.Group) -> None:
        super(EnergyBandsWithWeights, EnergyBandsWithWeights).readDataFromH5grp(energy_bands_with_weights, h5grp)
        energy_bands_with_weights._weights = np.asarray(h5grp['weights'], dtype=Real)
        energy_bands_with_weights._types_of_ions = np.char.decode(h5grp["types_of_ions"]).tolist()
        energy_bands_with_weights._nums_of_each_type = np.asarray(h5grp["nums_of_each_type"], dtype=Int)
        return

    def saveFile(self, file_name: String) -> None:
        EnergyBandsWithWeights.saveData2File(self, file_name)
        return

    @staticmethod
    def saveData2File(energy_bands_with_weights: EnergyBandsWithWeights, file_name: String) -> None:
        with h5.File(file_name, 'a') as f:
            grp: h5.Group = f.require_group('results')
            if 'energy_bands_with_weights' in grp:
                del grp['energy_bands_with_weights']
            subgrp: h5.Group = grp.create_group('energy_bands_with_weights')
            EnergyBandsWithWeights.saveData2H5grp(energy_bands_with_weights, subgrp)
        return

    @staticmethod
    def saveData2H5grp(energy_bands_with_weights: EnergyBandsWithWeights, h5grp: h5.Group) -> None:
        super(EnergyBandsWithWeights, EnergyBandsWithWeights).saveData2H5grp(energy_bands_with_weights, h5grp)
        utf8_type = h5.string_dtype('utf-8', 30)
        h5grp.create_dataset('weights', shape=np.shape(energy_bands_with_weights._weights), dtype=Real,
                             data=energy_bands_with_weights._weights, chunks=True, compression='gzip',
                             compression_opts=9)
        h5grp.create_dataset('types_of_ions', shape=np.shape(energy_bands_with_weights._types_of_ions),
                             dtype=utf8_type, data=energy_bands_with_weights._types_of_ions)
        h5grp.create_dataset('nums_of_each_type', shape=np.shape(energy_bands_with_weights._nums_of_each_type),
                             dtype=Int, data=energy_bands_with_weights._nums_of_each_type)
        return

    def loadVaspdata(self, vasp_data: Vaspdata) -> None:
        EnergyBandsWithWeights.loadDataFromVaspdata(self, vasp_data)
        return

    @staticmethod
    def loadDataFromVaspdata(energy_bands_with_weights: EnergyBandsWithWeights, vasp_data: Vaspdata) -> None:
        super(EnergyBandsWithWeights, EnergyBandsWithWeights).loadDataFromVaspdata(energy_bands_with_weights,
                                                                                   vasp_data)
        energy_bands_with_weights._weights = np.absolute(vasp_data.phase)
        energy_bands_with_weights._types_of_ions = vasp_data.types_of_ions
        energy_bands_with_weights._nums_of_each_type = vasp_data.nums_of_each_type
        return

    def plotFigure(self, file_name: String, *,
                   xlim: List = None, ylim: List = None,
                   config: object) -> None:
        EnergyBandsWithWeights.plotFigureOfEnergyBands(
            self, file_name, xlim=xlim, ylim=ylim, config=config)
        return

    @staticmethod
    def plotFigureOfEnergyBands(energy_bands_with_weights: EnergyBandsWithWeights, file_name: String, *,
                                xlim: List = None, ylim: List = None,
                                config: object) -> None:
        fig, ax = initFigure()
        divnorm = mcolors.TwoSlopeNorm(vmin=0., vcenter=0.5, vmax=1.)
        line_segments, selected_weights_segments = EnergyBandsWithWeights.genSegments(
            energy_bands_with_weights, config)
        colors: List = ['Purple', 'Blue', 'Green', 'Orange', 'Red', 'Grey']
        colormaps: List = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greys']
        for i, label in enumerate(selected_weights_segments.keys()):
            line_collection: LineCollection = LineCollection(
                line_segments, array=selected_weights_segments[label], linewidths=selected_weights_segments[label],
                cmap=colormaps[i % 6], label=label, norm=divnorm)
            ax.add_collection(line_collection)
        ax.set_ylabel(r"\(E-E_F\) \(\left[\mathrm{eV}\right]\)")
        leg = ax.legend()
        for i, line in enumerate(leg.get_lines()):
            line.set_linewidth(2.)
            line.set_color(colors[i % 6])
        ax.set_xticks(energy_bands_with_weights._xticks)
        if energy_bands_with_weights.xticklabels is not None:
            ax.set_xticklabels(energy_bands_with_weights.xticklabels)
        if xlim is None:
            xlim = [energy_bands_with_weights._kpath[0], energy_bands_with_weights._kpath[-1]]
        ax.set_xlim(xlim)
        if ylim is None:
            ylim = [-2., 1.]
        ax.set_ylim(ylim)
        ax.grid()
        plt.show()
        fig.savefig(file_name)
        plt.close()
        return

    @staticmethod
    def genSegments(
            energy_bands_with_weights: EnergyBandsWithWeights, config: object) -> (Array, Dict):
        x: Array = np.column_stack([energy_bands_with_weights._kpath[:-1], energy_bands_with_weights._kpath[1:]])
        x = np.delete(x, energy_bands_with_weights._discontinued_indices - 1, axis=0)
        data: Array = energy_bands_with_weights._eigenvalues
        ys = np.stack([np.expand_dims(data[:, :-1] - energy_bands_with_weights.efermi, axis=2),
                       np.expand_dims(data[:, 1:] - energy_bands_with_weights.efermi, axis=2)], axis=2)
        ys = np.delete(ys, energy_bands_with_weights._discontinued_indices - 1, axis=1)
        line_segments: Array = np.insert(ys, 0, x, axis=3)
        line_segments = line_segments.reshape(-1, 2, 2)
        atoms_indices: Dict = EnergyBandsWithWeights.genAtomsIndices(
            energy_bands_with_weights._types_of_ions, energy_bands_with_weights._nums_of_each_type)
        selected_weight: Dict = EnergyBandsWithWeights.selectWeights(
            energy_bands_with_weights._weights, atoms_indices, config)
        selected_weight_segments: Dict = EnergyBandsWithWeights.genSegments4SelectedWeights(
            selected_weight, energy_bands_with_weights._discontinued_indices)
        return (line_segments, selected_weight_segments)

    @staticmethod
    def selectWeights(weights: Array, atoms_indices: Dict, config: object) -> Dict:
        selected_weights: Dict = {}
        if type(config) == List:
            if config[0] in atoms_indices.keys():
                for atomic_symbol in config:
                    key: String = atomic_symbol
                    value: Array = np.sum(
                        weights[:, :, atoms_indices[atomic_symbol], :], axis=2)
                    selected_weights[key] = value
            elif config[0] in EnergyBandsWithWeights.atomic_orbits_indices.keys():
                for orbital_symbol in config:
                    key: String = orbital_symbol
                    value: Array = np.sum(
                        weights[:, :, :, EnergyBandsWithWeights.atomic_orbits_indices[orbital_symbol]], axis=2)
                    selected_weights[key] = value
        elif type(config) == Dict:
            for atomic_symbol in config.keys():
                if 'all' in config[atomic_symbol]:
                    config[atomic_symbol] = ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2']
                for orbital_symbol in config[atomic_symbol]:
                    key: String = '{0:s}: {1:s}'.format(atomic_symbol, orbital_symbol)
                    value: Array = weights[:, :, atoms_indices[atomic_symbol],
                                   EnergyBandsWithWeights.atomic_orbits_indices[orbital_symbol]]
                    selected_weights[key] = value
        return selected_weights

    @staticmethod
    def genAtomsIndices(types_of_ions: List, nums_of_each_type: Array) -> Dict:
        atoms_indices: Dict = {}
        for i, (ion_type, ion_nums) in enumerate(zip(types_of_ions, nums_of_each_type)):
            for k in range(1, ion_nums+1):
                key: String = '{1:d}{0:s}'.format(ion_type, k)
                value: Int = i
                atoms_indices[key] = value
        return atoms_indices

    @staticmethod
    def genSegments4SelectedWeights(selected_weights: Dict, discontinued_indices: Array) -> Dict:
        selected_weights_segments: Dict = {}
        for label in selected_weights.keys():
            value: Array = (selected_weights[label][:, :-1] + selected_weights[label][:, 1:]) / 2.
            value = np.delete(value, discontinued_indices - 1, axis=1)
            value = np.ravel(value)
            selected_weights_segments[label] = value
        return selected_weights_segments

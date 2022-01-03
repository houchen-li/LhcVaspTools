from __future__ import annotations

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection

from LhcVaspTools.BasicUtils import Int, Real, String, Array, List,\
    Vaspdata, EnergyBands, GaussFilter, ElecDnstyCrsSec


class EnergyBandsWithWeights(EnergyBands):

    def __init__(self, kpath: Array = None, eigenvalues: Array = None, weights: Array = None, efermi: Real = 0.,
                 xticklabels: List = None, xticks: Array = None,
                 xlim: Array = None, ylim: Array = None,
                 discontinued_indices: Array = None) -> None:
        super(EnergyBandsWithWeights, self).__init__(kpath, eigenvalues, efermi,
                                                     xticklabels, xticks,
                                                     xlim, ylim,
                                                     discontinued_indices,)
        self._weights: Array = weights
        return

    def readFile(self, file_name: String) -> None:
        EnergyBandsWithWeights.readDataFromFile(self, file_name)
        return

    @staticmethod
    def readDataFromFile(energy_bands_with_weights: EnergyBandsWithWeights, file_name: String) -> None:
        with h5.File(file_name, 'r') as f:
            grp: h5.Group = f['results/energy_bands_with_weight']
            EnergyBandsWithWeights.readDataFromH5grp(energy_bands_with_weights, grp)
        return

    @staticmethod
    def readDataFromH5grp(energy_bands_with_weights: EnergyBandsWithWeights, h5grp: h5.Group) -> None:
        super(EnergyBandsWithWeights, EnergyBandsWithWeights).readDataFromH5grp(energy_bands_with_weights, h5grp)
        energy_bands_with_weights._weights = np.asarray(h5grp['weights'], dtype=Real)
        return

    def saveFile(self, file_name: String) -> None:
        EnergyBandsWithWeights.saveData2File(self, file_name)
        return

    @staticmethod
    def saveData2File(energy_bands_with_weights: EnergyBandsWithWeights, file_name: String) -> None:
        with h5.File(file_name, 'a') as f:
            grp: h5.Group = f.require_group('results')
            if 'energy_bands_with_weights' in grp:
                del grp['enery_bands_with_weights']
            subgrp: h5.Group = grp.create_group('energy_bands_with_weight')
            EnergyBandsWithWeights.saveData2H5grp(energy_bands_with_weights, subgrp)
        return

    @staticmethod
    def saveData2H5grp(energy_bands_with_weights: EnergyBandsWithWeights, h5grp: h5.Group) -> None:
        super(EnergyBandsWithWeights, EnergyBandsWithWeights).saveData2H5grp()
        h5grp.create_dataset('weights', shape=np.shape(energy_bands_with_weights._weights), dtype=Real,
                             data=energy_bands_with_weights._weights, chunks=True, compression='gzip',
                             compression_opts=9)
        return

    def loadVaspdata(self, vasp_data: Vaspdata) -> None:
        EnergyBandsWithWeights.loadDataFromVaspdata(self, vasp_data)
        return

    @staticmethod
    def loadDataFromVaspdata(energy_bands_with_weights: EnergyBandsWithWeights, vasp_data: Vaspdata) -> None:
        super(EnergyBandsWithWeights, EnergyBandsWithWeights).loadDataFromVaspdata(energy_bands_with_weights,
                                                                                   vasp_data)
        energy_bands_with_weights._weights = np.absolute(vasp_data.phase)
        return

    def plotFigure(self, file_name: String, *,
                   xlim: List = None, ylim: List = None,
                   atomic_orbits: List) -> None:
        return

    @staticmethod
    def plotFigureOfEnergyBands(energy_bands_with_weights: EnergyBandsWithWeights, file_name: String, *,
                                xlim: List = None, ylim: List = None,
                                atomic_orbits: List) -> None:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]
        })
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
        })
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.add_subplot()
        divnorm = mcolors.TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=0.15)
        line_segments, weight_segments = EnergyBandsWithWeights.genSegments(energy_bands_with_weights, atomic_orbits)
        colors: List = [mcolors.to_rgba(c) \
                        for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
        for i in np.shape(weights_segments)[0]:
            line_collection: LineCollection = LineCollection(
                line_segments, array=weight_segments, cmap='coolwarm', norm=divnorm)
            ax.add_collection(line_collection)
        # cbar = fig.colorbar(line_collection, ax=ax)
        # cbar.ax.set_ylabel(r"\(\left<L_" + component[1] + r"\right>\)")
        ax.set_ylabel(r"\(E-E_F\) \(\left[\mathrm{eV}\right]\)")
        ax.set_xticks(energy_bands_with_weights._xticks)
        if energy_bands_with_weights.xticklabels is not None:
            ax.set_xticklabels(energy_bands_with_weights.xticklabels)
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
    def genSegments(energy_bands_with_weights: EnergyBandsWithWeights, component: String,
                    band_indices: List) -> (Array, Array):
        x: Array = np.column_stack([energy_bands_with_weights._kpath[:-1], energy_bands_with_weights._kpath[1:]])
        x = np.delete(x, energy_bands_with_weights._discontinued_indices - 1, axis=0)
        if band_indices is None:
            data: Array = energy_bands_with_weights._eigenvalues
        else:
            data: Array = energy_bands_with_weights._eigenvalues[band_indices]
        ys = np.stack([np.expand_dims(data[:, :-1] - energy_bands_with_weights.efermi, axis=2),
                       np.expand_dims(data[:, 1:] - energy_bands_with_weights.efermi, axis=2)], axis=2)
        ys = np.delete(ys, energy_bands_with_weights._discontinued_indices - 1, axis=1)
        line_segments: Array = np.insert(ys, 0, x, axis=3)
        line_segments = line_segments.reshape(-1, 2, 2)
        if component == 'Lx':
            L: Array = np.sum(energy_bands_with_weights._Lx, axis=2)
        elif component == 'Ly':
            L: Array = np.sum(energy_bands_with_weights._Ly, axis=2)
        elif component == 'Lz':
            L: Array = np.sum(energy_bands_with_weights._Lz, axis=2)
        L_segments: Array = (L[:, :-1] + L[:, 1:]) / 2.
        L_segments = np.delete(L_segments, energy_bands_with_weights._discontinued_indices - 1, axis=1)
        L_segments = np.ravel(L_segments)
        return (line_segments, L_segments)
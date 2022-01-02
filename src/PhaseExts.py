from __future__ import annotations

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection

from LhcVaspTools.BasicUtils import Int, Real, String, \
    Array, List, Vaspdata, EnergyBands, GaussFilter, ElecDnstyCrsSec


class EnergyBandsWithWeights(EnergyBands):

    def __init__(self, kpath: Array = None, eigenvalues: Array = None, weights: Array = None, efermi: Real = 0.,
                 xticklabels: List = None, xticks: Array = None,
                 xlim: Array = None, ylim: Array = None,
                 discontinued_indices: Array = None,
                 orbits: List = None) -> None:
        super(EnergyBandsWithWeights, self).__init__(kpath, eigenvalues, efermi,
                                                     xticklabels, xticks,
                                                     xlim, ylim,
                                                     discontinued_indices,
                                                     orbits)
        self._weights: Array = weights
        return

    def readFile(self, file_name: String) -> None:
        return

    @staticmethod
    def readDataFromFile(energy_bands_with_weights: EnergyBandsWithWeights, file_name: String) -> None:
        return

    @staticmethod
    def readDataFromH5grp(energy_bands_with_weights: EnergyBandsWithWeights, h5grp: h5.Group) -> None:
        return

    def saveFile(self, file_name: String) -> None:
        return

    @staticmethod
    def saveData2File(energy_bands_with_weights: EnergyBandsWithWeights, file_name: String) -> None:
        return

    @staticmethod
    def saveData2H5grp(energy_bands_with_weights: EnergyBandsWithWeights, h5grp: h5.Group) -> None:
        return

    def plotFigure(self, file_name: String) -> None:
        return

    @staticmethod
    def plotFigureOfEnergyBands(energy_bands_with_weights: EnergyBandsWithWeights, file_name: String) -> None:
        return
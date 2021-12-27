from __future__ import annotations

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection

from LhcVaspTools.BasicUtils import Int, Real, String, Array, List,\
    initFigure, Vaspdata, EnergyBands, GaussFilter, ElecDnstyCrsSec


class OamCalc(object):

    sqrt_3 = np.sqrt(3.)

    def __init__(self, phase: Array = None) -> None:
        self.phase: Array = phase
        self.Lx: Array = None
        self.Ly: Array = None
        self.Lz: Array = None
        return

    def do_calculation(self) -> None:
        OamCalc.calOam(self)
        return

    @staticmethod
    def calOam(oam_calc: OamCalc) -> None:
        oam_calc.Lx = np.apply_along_axis(OamCalc.genLxAtPhase, 3, oam_calc.phase)
        oam_calc.Ly = np.apply_along_axis(OamCalc.genLyAtPhase, 3, oam_calc.phase)
        oam_calc.Lz = np.apply_along_axis(OamCalc.genLzAtPhase, 3, oam_calc.phase)
        return

    @staticmethod
    def genLxAtPhase(x: Array) -> Real:
        temp: Array = np.asarray([0, x[2] * 1J, -x[1] * 1J, 0, x[7] * 1J,
                                  (x[6] * OamCalc.sqrt_3 + x[8]) * 1J, -x[5] * OamCalc.sqrt_3 * 1J, -x[4] * 1J,
                                  -x[5] * 1J])
        lx: Real = np.real(np.dot(np.conjugate(x), temp))
        return lx

    @staticmethod
    def genLyAtPhase(x: Array) -> Real:
        temp: Array = np.asarray([0, 0, -x[3] * 1J, x[2] * 1J, x[5] * 1J, -x[4] * 1J,
                                  -x[7] * OamCalc.sqrt_3 * 1J, (x[6] * OamCalc.sqrt_3 - x[8]) * 1J, x[7] * 1J])
        ly: Real = np.real(np.dot(np.conjugate(x), temp))
        return ly

    @staticmethod
    def genLzAtPhase(x: Array) -> Real:
        temp: Array = np.asarray([0, -x[3] * 1J, 0, x[1] * 1J,
                                       -x[8] * 2J, -x[7] * 1J, 0, x[5] * 1J, x[4] * 2J])
        lz: Real = np.real(np.dot(np.conjugate(x), temp))
        return lz


class EnergyBandsWithOam(EnergyBands):

    def __init__(self, kpath: Array = None, eigenvalues: Array = None,
                 Lx: Array = None, Ly: Array = None, Lz: Array = None, efermi: Real = 0.,
                 xticklabels: List = None, xticks: Array = None,
                 discontinued_indices: Array = None) -> None:
        super(EnergyBandsWithOam, self).__init__(kpath, eigenvalues, efermi,
                                                 xticklabels, xticks,
                                                 discontinued_indices)
        self._Lx: Array = Lx
        self._Ly: Array = Ly
        self._Lz: Array = Lz
        return

    def readFile(self, file_name: String) -> None:
        EnergyBandsWithOam.readDataFromFile(self, file_name)
        return

    @staticmethod
    def readDataFromFile(energy_bands_with_oam: EnergyBandsWithOam, file_name: String) -> None:
        with h5.File(file_name, 'r') as f:
            grp: h5.Group = f['results/energy_bands_with_oam']
            EnergyBandsWithOam.readDataFromH5grp(energy_bands_with_oam, grp)
        return

    @staticmethod
    def readDataFromH5grp(energy_bands_with_oam: EnergyBands, h5grp: h5.Group) -> None:
        super(EnergyBandsWithOam, EnergyBandsWithOam).readDataFromH5grp(energy_bands_with_oam, h5grp)
        energy_bands_with_oam._Lx = np.asarray(h5grp['Lx'], dtype=Real)
        energy_bands_with_oam._Ly = np.asarray(h5grp['Ly'], dtype=Real)
        energy_bands_with_oam._Lz = np.asarray(h5grp['Lz'], dtype=Real)
        return

    def saveFile(self, file_name: String) -> None:
        EnergyBandsWithOam.saveData2File(self, file_name)
        return

    @staticmethod
    def saveData2File(energy_bands_with_oam: EnergyBandsWithOam, file_name: String) -> None:
        with h5.File(file_name, 'a') as f:
            grp: h5.Group = f.require_group("results")
            if 'energy_bands_with_oam' in grp:
                del grp['energy_bands_with_oam']
            h5grp: h5.Group = grp.create_group('energy_bands_with_oam')
            EnergyBandsWithOam.saveData2H5grp(energy_bands_with_oam, h5grp)
        return

    @staticmethod
    def saveData2H5grp(energy_bands_with_oam: EnergyBandsWithOam, h5grp: h5.Group) -> None:
        super(EnergyBandsWithOam, EnergyBandsWithOam).saveData2H5grp(energy_bands_with_oam, h5grp)
        h5grp.create_dataset('Lx', shape=np.shape(energy_bands_with_oam._Lx), dtype=Real,
                             data=energy_bands_with_oam._Lx, chunks=True, compression='gzip', compression_opts=9)
        h5grp.create_dataset('Ly', shape=np.shape(energy_bands_with_oam._Ly), dtype=Real,
                             data=energy_bands_with_oam._Ly, chunks=True, compression='gzip', compression_opts=9)
        h5grp.create_dataset('Lz', shape=np.shape(energy_bands_with_oam._Lz), dtype=Real,
                             data=energy_bands_with_oam._Lz, chunks=True, compression='gzip', compression_opts=9)
        return

    def loadVaspdata(self, vasp_data: Vaspdata) -> None:
        EnergyBandsWithOam.loadDataFromVaspdata(self, vasp_data)
        return

    @staticmethod
    def loadDataFromVaspdata(energy_bands_with_oam: EnergyBandsWithOam, vasp_data: Vaspdata) -> None:
        super(EnergyBandsWithOam, EnergyBandsWithOam).loadDataFromVaspdata(energy_bands_with_oam, vasp_data)
        oam_calc: OamCalc = OamCalc()
        oam_calc.phase = vasp_data.phase
        oam_calc.do_calculation()
        energy_bands_with_oam._Lx = oam_calc.Lx
        energy_bands_with_oam._Ly = oam_calc.Ly
        energy_bands_with_oam._Lz = oam_calc.Lz
        return

    def plotFigure(self, file_name: String, component: String, *,
                   xlim: List = None, ylim: List = None,
                   band_indices: List = None) -> None:
        EnergyBandsWithOam.plotFigureOfEnergyBands(
            self, file_name, component, xlim=xlim, ylim=ylim, band_indices=band_indices)
        return

    @staticmethod
    def plotFigureOfEnergyBands(energy_bands_with_oam: EnergyBandsWithOam, file_name: String, component: String, *,
                                xlim: List = None, ylim: List = None,
                                band_indices: List = None) -> None:
        fig, ax = initFigure()
        divnorm = mcolors.TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=0.15)
        line_segments, L_segments = EnergyBandsWithOam.genSegments(energy_bands_with_oam, component, band_indices)
        line_collection: LineCollection = LineCollection(
            line_segments, array=L_segments, cmap='coolwarm', norm=divnorm)
        ax.add_collection(line_collection)
        cbar = fig.colorbar(line_collection, ax=ax)
        cbar.ax.set_ylabel(r"\(\left<L_" + component[1] + r"\right>\)")
        ax.set_ylabel(r"\(E-E_F\) \(\left[\mathrm{eV}\right]\)")
        ax.set_xticks(energy_bands_with_oam._xticks)
        if energy_bands_with_oam.xticklabels is not None:
            ax.set_xticklabels(energy_bands_with_oam.xticklabels)
        if xlim is None:
            xlim = [energy_bands_with_oam._kpath[0], energy_bands_with_oam._kpath[-1]]
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
    def genSegments(energy_bands_with_oam: EnergyBandsWithOam, component: String,
                    band_indices: List) -> (Array, Array):
        x: Array = np.column_stack([energy_bands_with_oam._kpath[:-1], energy_bands_with_oam._kpath[1:]])
        x = np.delete(x, energy_bands_with_oam._discontinued_indices - 1, axis=0)
        if band_indices is None:
            data: Array = energy_bands_with_oam._eigenvalues
        else:
            data: Array = energy_bands_with_oam._eigenvalues[band_indices]
        ys = np.stack([np.expand_dims(data[:, :-1] - energy_bands_with_oam.efermi, axis=2),
                       np.expand_dims(data[:, 1:] - energy_bands_with_oam.efermi, axis=2)], axis=2)
        ys = np.delete(ys, energy_bands_with_oam._discontinued_indices - 1, axis=1)
        line_segments: Array = np.insert(ys, 0, x, axis=3)
        line_segments = line_segments.reshape(-1, 2, 2)
        if component == 'Lx':
            L: Array = np.sum(energy_bands_with_oam._Lx, axis=2)
        elif component == 'Ly':
            L: Array = np.sum(energy_bands_with_oam._Ly, axis=2)
        elif component == 'Lz':
            L: Array = np.sum(energy_bands_with_oam._Lz, axis=2)
        L_segments: Array = (L[:, :-1] + L[:, 1:]) / 2.
        L_segments = np.delete(L_segments, energy_bands_with_oam._discontinued_indices - 1, axis=1)
        L_segments = np.ravel(L_segments)
        return (line_segments, L_segments)


class ElecDnstyCrsSecWithOam(ElecDnstyCrsSec):

    def __init__(self, kx: Array = None, ky: Array = None, eigenvalues: Array = None,
                 LX: Array = None, LY: Array = None, LZ: Array = None,
                 energy_level: Real = None, sigma: Real = None) -> None:
        super(ElecDnstyCrsSecWithOam, self).__init__(kx, ky, eigenvalues, energy_level, sigma)
        self._LX: Array = LX
        self._LY: Array = LY
        self._LZ: Array = LZ
        return

    def readFile(self, file_name: String) -> None:
        ElecDnstyCrsSecWithOam.readDataFromFile(self, file_name)
        return

    @staticmethod
    def readDataFromFile(elec_dnsty_crs_sec_with_oam: ElecDnstyCrsSecWithOam, file_name: String) -> None:
        with h5.File(file_name, 'r') as f:
            grp: h5.Group = f['results/elec_dnsty_crs_sec_with_oam']
            ElecDnstyCrsSecWithOam.readDataFromH5grp(elec_dnsty_crs_sec_with_oam, grp)
        return

    @staticmethod
    def readDataFromH5grp(elec_dnsty_crs_sec_with_oam: ElecDnstyCrsSecWithOam, h5grp: h5.Group) -> None:
        super(ElecDnstyCrsSecWithOam, ElecDnstyCrsSecWithOam).readDataFromH5grp(elec_dnsty_crs_sec_with_oam, h5grp)
        elec_dnsty_crs_sec_with_oam._LX = np.asarray(h5grp['LX'], dtype=Real)
        elec_dnsty_crs_sec_with_oam._LY = np.asarray(h5grp['LY'], dtype=Real)
        elec_dnsty_crs_sec_with_oam._LZ = np.asarray(h5grp['LZ'], dtype=Real)
        return

    def saveFile(self, file_name: String) -> None:
        ElecDnstyCrsSecWithOam.saveData2File(self, file_name)
        return

    @staticmethod
    def saveData2File(elec_dnsty_crs_sec_with_oam: ElecDnstyCrsSecWithOam, file_name: String) -> None:
        with h5.File(file_name, 'a') as f:
            grp: h5.Group = f.require_group('results')
            if 'results/elec_dnsty_crs_sec_with_oam' in f:
                del f['results/elec_dnsty_crs_sec_with_oam']
            subgrp: h5.Group = grp.create_group('elec_dnsty_crs_sec_with_oam')
            ElecDnstyCrsSecWithOam.saveData2H5grp(elec_dnsty_crs_sec_with_oam, subgrp)
        return

    @staticmethod
    def saveData2H5grp(elec_dnsty_crs_sec_with_oam: ElecDnstyCrsSecWithOam, h5grp: h5.Group) -> None:
        super(ElecDnstyCrsSecWithOam, ElecDnstyCrsSecWithOam).saveData2H5grp(elec_dnsty_crs_sec_with_oam, h5grp)
        h5grp.create_dataset('LX', shape=np.shape(elec_dnsty_crs_sec_with_oam._LX), dtype=Real,
                             data=elec_dnsty_crs_sec_with_oam._LX, chunks=True, compression='gzip',
                             compression_opts=9)
        h5grp.create_dataset('LY', shape=np.shape(elec_dnsty_crs_sec_with_oam._LY), dtype=Real,
                             data=elec_dnsty_crs_sec_with_oam._LY, chunks=True, compression='gzip',
                             compression_opts=9)
        h5grp.create_dataset('LZ', shape=np.shape(elec_dnsty_crs_sec_with_oam._LZ), dtype=Real,
                             data=elec_dnsty_crs_sec_with_oam._LZ, chunks=True, compression='gzip',
                             compression_opts=9)
        return

    def loadVaspdata(self, vasp_data: Vaspdata, num_of_grids: List) -> None:
        ElecDnstyCrsSecWithOam.loadDataFromVaspdata(self, vasp_data, num_of_grids)
        return

    @staticmethod
    def loadDataFromVaspdata(elec_dnsty_crs_sec_with_oam: ElecDnstyCrsSecWithOam, vasp_data: Vaspdata,
                             num_of_grids: List) -> None:
        super(ElecDnstyCrsSecWithOam, ElecDnstyCrsSecWithOam).loadDataFromVaspdata(elec_dnsty_crs_sec_with_oam,
                                                                                   vasp_data, num_of_grids)
        oam_calc: OamCalc = OamCalc()
        oam_calc.phase = vasp_data.phase
        oam_calc.do_calculation()
        elec_dnsty_crs_sec_with_oam._LX = oam_calc.Lx.reshape((
            vasp_data.num_of_bands, num_of_grids[1], num_of_grids[0], vasp_data.num_of_ions))
        elec_dnsty_crs_sec_with_oam._LY = oam_calc.Ly.reshape((
            vasp_data.num_of_bands, num_of_grids[1], num_of_grids[0], vasp_data.num_of_ions))
        elec_dnsty_crs_sec_with_oam._LZ = oam_calc.Lz.reshape((
            vasp_data.num_of_bands, num_of_grids[1], num_of_grids[0], vasp_data.num_of_ions))
        return

    def plotFigure(self, file_name: String, component: String, band_indices: List = None) -> None:
        ElecDnstyCrsSecWithOam.plotFigureOfElecDnsty(self, file_name, component, band_indices)
        return

    @staticmethod
    def plotFigureOfElecDnsty(elec_dnsty_crs_sec_with_oam: ElecDnstyCrsSecWithOam, file_name: String,
                              component: String, band_indices: List = None) -> None:
        fig, ax = initFigure()
        divnorm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
        L_density: Array = ElecDnstyCrsSecWithOam.genOamDensityForCrsSec(elec_dnsty_crs_sec_with_oam,
                                                                         component, band_indices)
        cont = ax.contourf(elec_dnsty_crs_sec_with_oam._KX, elec_dnsty_crs_sec_with_oam._KY, L_density,
                           cmap='coolwarm', norm=divnorm, levels=50)
        cbar = fig.colorbar(cont, ax=ax)
        cbar.ax.set_ylabel(r"\(\left<L_"+component[1]+r"\right>\)")
        ax.set_aspect('equal', adjustable='box')
        #ax.set_xlabel(r"\(k_x\)")
        #ax.set_ylabel(r"\(k_y\)")
        plt.show()
        fig.savefig(file_name)
        plt.close()
        return

    @staticmethod
    def genOamDensityForCrsSec(elec_dnsty_crs_sec_with_oam: ElecDnstyCrsSecWithOam, component: String,
                               band_indices: List = None) -> Array:
        gauss_filter: GaussFilter = GaussFilter()
        if band_indices is None:
            data: Array = elec_dnsty_crs_sec_with_oam._eigenvalues
            if component == 'Lx':
                L: Array = np.sum(elec_dnsty_crs_sec_with_oam._LX, axis=3)
            elif component == 'Ly':
                L: Array = np.sum(elec_dnsty_crs_sec_with_oam._LY, axis=3)
            elif component == 'Lz':
                L: Array = np.sum(elec_dnsty_crs_sec_with_oam._LZ, axis=3)
        else:
            data: Array = elec_dnsty_crs_sec_with_oam._eigenvalues[band_indices]
            if component == 'Lx':
                L: Array = np.sum(elec_dnsty_crs_sec_with_oam._LX[band_indices], axis=3)
            elif component == 'Ly':
                L: Array = np.sum(elec_dnsty_crs_sec_with_oam._LY[band_indices], axis=3)
            elif component == 'Lz':
                L: Array = np.sum(elec_dnsty_crs_sec_with_oam._LZ[band_indices], axis=3)
        gauss_filter.eigenvalues = data
        gauss_filter.energy_level = elec_dnsty_crs_sec_with_oam.energy_level
        gauss_filter.sigma = elec_dnsty_crs_sec_with_oam.sigma
        gauss_filter.do_calculation()
        L_density: Array = np.sum(L * gauss_filter.f, axis=0)
        return L_density

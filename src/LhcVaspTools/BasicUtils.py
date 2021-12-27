from __future__ import annotations

import sys

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection

Int = np.int_
Real = np.float_
Complex = np.complex_
String = str
Array = np.ndarray
List = list
Dict = dict


def readDataFromJson(file_name: String) -> object:
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data


def saveData2Json(data: object, file_name: String) -> None:
    with open(file_name, 'w') as f:
        json.dump(data, f)
    return


def initFigure() -> (plt.Figure, plt.Axes):
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
    return (fig, ax)


class Vaspdata(object):

    def __init__(self, num_of_bands: Int = None, num_of_kpoints: Int = None,
                 num_of_ions: Int = None, num_of_orbits: Int = None,
                 scale: Real = None, lattice_vectors: Array = None,
                 types_of_ions: Array = None, nums_of_each_type: Array = None, positions_of_ions: Array = None,
                 kpoints: Array = None, eigenvalues: Array = None,
                 energies: Array = None, dos: Array = None, dos_integral: Array = None,
                 efermi: Real = None, phase: Array = None) -> None:
        self.num_of_bands: Int = num_of_bands
        self.num_of_kpoints: Int = num_of_kpoints
        self.num_of_ions: Int = num_of_ions
        self.num_of_orbits: Int = num_of_orbits
        self.scale: Real = scale
        self.lattice_vectors: Array = lattice_vectors
        self.types_of_ions: List = types_of_ions
        self.nums_of_each_type: Array = nums_of_each_type
        self.positions_of_ions: Array = positions_of_ions
        self.kpoints: Array = kpoints
        self.eigenvalues: Array = eigenvalues
        self.energies: Array = energies
        self.dos: Array = dos
        self.dos_integral: Array = dos_integral
        self.efermi: Real = efermi
        self.phase: Array = phase
        return

    def readFile(self, file_name: String) -> None:
        Vaspdata.readDataFromFile(self, file_name)
        return

    @staticmethod
    def readDataFromFile(vasp_data: Vaspdata, file_name: String) -> None:
        with h5.File(file_name, 'r') as f:
            grp: h5.Group = f['results']
            Vaspdata.readDataFromH5grp(vasp_data, grp)
        return

    @staticmethod
    def readDataFromH5grp(vasp_data: Vaspdata, h5grp: h5.Group) -> None:
        vasp_data.num_of_bands = Int(h5grp["electron_eigenvalues/nb_tot"][()])
        vasp_data.num_of_kpoints = Int(h5grp["electron_eigenvalues/kpoints"][()])
        vasp_data.num_of_ions = Int(h5grp["electron_eigenvalues/nions"][()])
        vasp_data.num_of_orbits = 9
        vasp_data.scale = Real(h5grp["positions/scale"])
        vasp_data.lattice_vectors = np.asarray(h5grp["positions/lattice_vectors"], dtype=Real)
        vasp_data.types_of_ions = np.char.decode(h5grp["positions/ion_types"]).tolist()
        vasp_data.nums_of_each_type = np.asarray(h5grp["positions/number_ion_types"], dtype=Int)
        vasp_data.positions_of_ions = np.asarray(h5grp["positions/position_ions"], dtype=Real)
        vasp_data.kpoints = np.asarray(h5grp["electron_eigenvalues/kpoint_coords"], dtype=Real)
        vasp_data.eigenvalues = np.asarray(
            h5grp["electron_eigenvalues/eigenvalues"][0, :, :], dtype=Real)
        vasp_data.eigenvalues = np.transpose(vasp_data.eigenvalues)
        vasp_data.energies = np.asarray(h5grp["electron_dos/energies"], dtype=Real)
        vasp_data.dos = np.asarray(h5grp["electron_dos/dos"], dtype=Real)
        vasp_data.dos_integral = np.asarray(h5grp["electron_dos/dosi"], dtype=Real)
        vasp_data.efermi = Real(h5grp["electron_dos/efermi"][()])
        vasp_data.phase = np.asarray(h5grp["projectors/phase"][0, :, :, :, :, 0], dtype=Real) + \
                          np.asarray(h5grp["projectors/phase"][0, :, :, :, :, 1], dtype=Real) * 1J
        return

    @staticmethod
    def deterCrossBands(vasp_data: Vaspdata, energy_level: Real = None) -> Array:
        if energy_level is None:
            energy_level = vasp_data.efermi
        cross_bands: Array = np.logical_and(
            np.amin(vasp_data.eigenvalues, axis=1) < energy_level,
            np.amax(vasp_data.eigenvalues, axis=1) > energy_level)
        cross_bands = np.ravel(np.argwhere(cross_bands))
        return cross_bands


class EnergyBands(object):

    def __init__(self, kpath: Array = None, eigenvalues: Array = None, efermi: Real = 0.,
                 xticklabels: List = None, xticks: Array = None,
                 discontinued_indices: Array = None) -> None:
        self._kpath: Array = kpath
        self._eigenvalues: Array = eigenvalues
        self.efermi: Real = efermi
        self.xticklabels: List = xticklabels
        self._xticks: Array = xticks
        self._discontinued_indices: Array = discontinued_indices
        return

    def readFile(self, file_name: String) -> None:
        EnergyBands.readDataFromFile(self, file_name)
        return

    @staticmethod
    def readDataFromFile(energy_bands: EnergyBands, file_name: String) -> None:
        with h5.File(file_name, 'r') as f:
            grp: h5.Group = f['results/energy_bands']
            EnergyBands.readDataFromH5grp(energy_bands, grp)
        return

    @staticmethod
    def readDataFromH5grp(energy_bands: EnergyBands, h5grp: h5.Group) -> None:
        energy_bands._kpath = np.asarray(h5grp['kpath'], dtype=Real)
        energy_bands._eigenvalues = np.asarray(h5grp['eigenvalues'], dtype=Real)
        energy_bands.efermi = Real(h5grp['efermi'][0])
        if 'xticklabels' in h5grp:
            energy_bands.xticklabels = np.char.decode(h5grp['xticklabels'])
        energy_bands._xticks = np.asarray(h5grp['xticks'], dtype=Real)
        energy_bands._discontinued_indices = np.asarray(h5grp['discontinued_indices'], dtype=Int)
        return

    def saveFile(self, file_name: String) -> None:
        EnergyBands.saveData2File(self, file_name)
        return

    @staticmethod
    def saveData2File(energy_bands: EnergyBands, file_name: String) -> None:
        with h5.File(file_name, 'a') as f:
            grp: h5.Group = f.require_group('results')
            if 'energy_bands' in grp:
                del grp['energy_bands']
            subgrp: h5.Group = grp.create_group('energy_bands')
            EnergyBands.saveData2H5grp(energy_bands, subgrp)
        return

    @staticmethod
    def saveData2H5grp(energy_bands: EnergyBands, h5grp: h5.Group) -> None:
        utf8_type = h5.string_dtype('utf-8', 30)
        h5grp.create_dataset("kpath", shape=np.shape(energy_bands._kpath), dtype=Real,
                             data=energy_bands._kpath, chunks=True, compression='gzip', compression_opts=9)
        h5grp.create_dataset("eigenvalues", shape=np.shape(energy_bands._eigenvalues), dtype=Real,
                             data=energy_bands._eigenvalues, chunks=True, compression='gzip', compression_opts=9)
        h5grp.create_dataset("efermi", shape=(1), dtype=Real, data=energy_bands.efermi)
        if energy_bands.xticklabels is not None:
            h5grp.create_dataset("xticklabels", shape=np.shape(energy_bands.xticklabels),
                                 dtype=utf8_type, data=energy_bands.xticklabels, chunks=True, compression='gzip',
                                 compression_opts=9)
        h5grp.create_dataset("xticks", shape=np.shape(energy_bands._xticks), dtype=Real,
                             data=energy_bands._xticks, chunks=True, compression='gzip', compression_opts=9)
        h5grp.create_dataset("discontinued_indices", shape=np.shape(energy_bands._discontinued_indices),
                             dtype=Int, data=energy_bands._discontinued_indices, chunks=True, compression='gzip',
                             compression_opts=9)
        return

    def loadVaspdata(self, vasp_data: Vaspdata) -> None:
        EnergyBands.loadDataFromVaspdata(self, vasp_data)
        return

    @staticmethod
    def loadDataFromVaspdata(energy_bands: EnergyBands, vasp_data: Vaspdata) -> None:
        reciprocal_lattice_vectors: Array = np.transpose(
            np.linalg.inv(vasp_data.lattice_vectors * vasp_data.scale))
        kdiff: Array = np.diff(np.dot(vasp_data.kpoints, reciprocal_lattice_vectors), axis=0)
        dk: Array = np.linalg.norm(kdiff, axis=1)
        dk = np.where(dk < 1e-2, dk, 0.)
        energy_bands._discontinued_indices = np.ravel(np.argwhere(dk < 1e-6)) + 1
        energy_bands._kpath = np.cumsum(np.insert(dk, 0, 0.), dtype=Real)
        energy_bands._xticks = np.asarray([energy_bands._kpath[0], energy_bands._kpath[-1]], dtype=Real)
        energy_bands._xticks = np.insert(
            energy_bands._xticks, 1, energy_bands._kpath[energy_bands._discontinued_indices])
        energy_bands._eigenvalues = vasp_data.eigenvalues
        return

    def plotFigure(self, file_name: String, *,
                   xlim: List = None, ylim: List = None,
                   band_indices: List = None) -> None:
        EnergyBands.plotFigureOfEnergyBands(self, file_name, xlim=xlim, ylim=ylim, band_indices=band_indices)
        return

    @staticmethod
    def plotFigureOfEnergyBands(energy_bands: EnergyBands, file_name: String, *,
                                xlim: List = None, ylim: List = None,
                                band_indices: List = None) -> None:
        fig, ax = initFigure()
        x: Array = np.insert(energy_bands._kpath, energy_bands._discontinued_indices, np.nan)
        if band_indices is None:
            data: Array = energy_bands._eigenvalues
        else:
            data: Array = energy_bands._eigenvalues[band_indices]
        ys: Array = np.insert(data - energy_bands.efermi,
                              energy_bands._discontinued_indices, np.nan, axis=1)
        line_segments: Array = np.insert(np.expand_dims(ys, axis=2), 0, x, axis=2)
        colors: List = [mcolors.to_rgba(c)
                        for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
        line_collection: LineCollection = LineCollection(line_segments, colors=colors)
        ax.add_collection(line_collection)
        ax.set_ylabel(r"\(E-E_F\) \(\left[\mathrm{eV}\right]\)")
        ax.set_xticks(energy_bands._xticks)
        if energy_bands.xticklabels is not None:
            ax.set_xticklabels(energy_bands.xticklabels)
        if xlim is None:
            xlim = [energy_bands._kpath[0], energy_bands._kpath[-1]]
        ax.set_xlim(xlim)
        if ylim is None:
            ylim = [-2., 1.]
        ax.set_ylim(ylim)
        ax.grid()
        plt.show()
        fig.savefig(file_name)
        plt.close()
        return


class GaussFilter(object):

    def __init__(self, eigenvalues: Array = None, energy_level: Real = None, sigma: Real = None) -> None:
        self.eigenvalues: Array = eigenvalues
        self.energy_level: Array = energy_level
        self.sigma: Real = sigma
        self.f: Array = None
        return

    def do_calculation(self) -> None:
        self.f = np.exp(((self.eigenvalues - self.energy_level) / self.sigma) ** 2 * (-0.5)) \
                 / (self.sigma * np.sqrt(2. * np.pi))
        self.f = np.where(np.logical_and(self.eigenvalues < self.energy_level, self.eigenvalues > self.sigma * 3.), self.f, 0.)
        return


class CrsSec(object):

    def __init__(self, KX: Array = None, KY: Array = None) -> None:
        self._KX: Array = KX
        self._KY: Array = KY
        return

    def readFile(self, file_name: String) -> None:
        CrsSec.readDataFromFile(self, file_name)
        return

    @staticmethod
    def readDataFromFile(crs_sec: CrsSec, file_name: String) -> None:
        with h5.File(file_name, 'r') as f:
            grp: h5.Group = f['results/crs_sec']
            CrsSec.readDataFromH5grp(crs_sec, grp)
        return

    @staticmethod
    def readDataFromH5grp(crs_sec: CrsSec, h5grp: h5.Group) -> None:
        crs_sec._KX = np.asarray(h5grp['KX'], dtype=Real)
        crs_sec._KY = np.asarray(h5grp['KY'], dtype=Real)
        return

    def saveFile(self, file_name: String) -> None:
        CrsSec.saveData2File(self, file_name)
        return

    @staticmethod
    def saveData2File(crs_sec: CrsSec, file_name: String) -> None:
        with h5.File(file_name, 'a') as f:
            grp: h5.Group = f.require_group('results')
            if 'crs_sec' in grp:
                del grp['crs_sec']
            subgrp: h5.Group = grp.create_group('crs_sec')
            CrsSec.saveData2H5grp(crs_sec, subgrp)
        return

    @staticmethod
    def saveData2H5grp(crs_sec: CrsSec, h5grp: h5.Group) -> None:
        h5grp.create_dataset('KX', shape=np.shape(crs_sec._KX), dtype=Real,
                             data=crs_sec._KX, chunks=True, compression='gzip', compression_opts=9)
        h5grp.create_dataset('KY', shape=np.shape(crs_sec._KY), dtype=Real,
                             data=crs_sec._KY, chunks=True, compression='gzip', compression_opts=9)
        return

    def loadVaspdata(self, vasp_data: Vaspdata, num_of_grids: List) -> None:
        CrsSec.loadDataFromVaspdata(self, vasp_data, num_of_grids)
        return

    @staticmethod
    def loadDataFromVaspdata(crs_sec: CrsSec, vasp_data: Vaspdata, num_of_grids: List) -> None:
        CrsSec.verityNumOfGrids(vasp_data.num_of_kpoints, num_of_grids)
        crs_sec._KX, crs_sec._KY = CrsSec.mapKPoints2Plane(vasp_data.kpoints, num_of_grids)
        return

    @staticmethod
    def verityNumOfGrids(num_of_kpoints: Int, num_of_grids: List) -> None:
        try:
            if num_of_kpoints is None:
                raise ValueError('number of kpoints has not been set.')
            else:
                if num_of_grids[0] is None and num_of_grids[1] is None:
                    num_of_grids[0] = int(np.sqrt(num_of_kpoints))
                    num_of_grids[1] = num_of_grids[0]
                    if num_of_kpoints != int(num_of_grids[0] * num_of_grids[1]):
                        raise ValueError('number of kpoints is not a perfect square integer.')
                elif num_of_grids[0] is not None and num_of_grids[1] is None:
                    num_of_grids[1] = num_of_kpoints // num_of_grids[0]
                    if num_of_kpoints != int(num_of_grids[0] * num_of_grids[1]):
                        raise ValueError('number of kpoints can not be divided by number of x grids.')
                elif num_of_grids[0] is None and num_of_grids[1] is not None:
                    num_of_grids[0] = num_of_kpoints // num_of_grids[1]
                    if num_of_kpoints != int(num_of_grids[0] * num_of_grids[1]):
                        raise ValueError('number of kpoints can not be divided by number of y grids.')
                elif num_of_grids[0] is not None and num_of_grids[1] is not None:
                    if num_of_kpoints != int(num_of_grids[0] * num_of_grids[1]):
                        raise ValueError('number of kpoints does not equal to the multiplication of \
                                          number of x and y grids.')
        except ValueError as err:
            print(err)
            sys.exit(1)
        return

    @staticmethod
    def mapKPoints2Plane(kpoints: Array, num_of_grids: List) -> (Array, Array):
        num_of_kpoints: Int = np.shape(kpoints)[1]
        m: Array = np.stack([kpoints[0, :], kpoints[num_of_grids[0]-1, :], kpoints[-1, :]]) + 10.1
        s: Array = np.linalg.solve(m, np.asarray([1., 1., 1.], dtype=Real))
        tmp_0: Real = np.linalg.norm(s[1:])
        tmp_1: Real = np.linalg.norm(s)
        sin_tx: Real = s[1] / tmp_0
        cos_tx: Real = s[2] / tmp_0
        sin_ty: Real = s[0] / tmp_1
        cos_ty: Real = tmp_0 / tmp_1
        rotate_matrix: Array = np.asarray([[cos_ty, 0, sin_ty],
                                           [-sin_tx * sin_ty, cos_tx, sin_tx * cos_ty],
                                           [-cos_tx * sin_ty, -sin_tx, cos_tx * cos_ty]],
                                          dtype=Real)
        data: Array = np.dot(kpoints, rotate_matrix)
        # v: Array = data[num_of_grids[0]-1]-data[0]
        # sin_tz = -v[1]/np.linalg.norm(v)
        # cos_tz = v[0]/np.linalg.norm(v)
        # rotate_matrix = np.asarray([[cos_tz, sin_tz, 0],
                                    # [-sin_tz, cos_tz, 0],
                                    # [0, 0, 1.]],
                                   # dtype=Real)
        # data = np.dot(data, rotate_matrix)
        KX: Array = data[:, 0].reshape(num_of_grids[1], num_of_grids[0])
        KY: Array = data[:, 1].reshape(num_of_grids[1], num_of_grids[0])
        return (KX, KY)


class EnergyCutCrsSec(CrsSec):

    def __init__(self, KX: Array = None, KY: Array = None, eigenvalues: Array = None, energy_level: Real = None,
                 cross_bands: Array = None) -> None:
        super(EnergyCutCrsSec, self).__init__(KX, KY)
        self._eigenvalues: Array = eigenvalues
        self.energy_level: Real = energy_level
        self.cross_bands: Array = cross_bands
        return

    def readFile(self, file_name: String) -> None:
        EnergyCutCrsSec.readDataFromFile(self, file_name)
        return

    @staticmethod
    def readDataFromFile(energy_cut_crs_sec: EnergyCutCrsSec, file_name: String) -> None:
        with h5.File(file_name, 'r') as f:
            grp = f['results/energy_cut_crs_sec']
            EnergyCutCrsSec.readDataFromH5grp(energy_cut_crs_sec, grp)
        return

    @staticmethod
    def readDataFromH5grp(energy_cut_crs_sec: CrsSec, h5grp: h5.Group) -> None:
        super(EnergyCutCrsSec, EnergyCutCrsSec).readDataFromH5grp(energy_cut_crs_sec, h5grp)
        energy_cut_crs_sec._eigenvalues = np.asarray(h5grp['eigenvalues'], dtype=Real)
        energy_cut_crs_sec.energy_level = Real(h5grp['energy_level'][0])
        energy_cut_crs_sec.cross_bands = np.asarray(h5grp['cross_bands'], dtype=Int)
        return

    def saveFile(self, file_name: String) -> None:
        EnergyCutCrsSec.saveData2File(self, file_name)
        return

    @staticmethod
    def saveData2File(energy_cut_crs_sec: EnergyCutCrsSec, file_name: String) -> None:
        with h5.File(file_name, 'a') as f:
            grp: h5.Group = f.require_group('results')
            if 'energy_cut_crs_sec' in grp:
                del grp['energy_cut_crs_sec']
            subgrp: h5.Group = grp.create_group('energy_cut_crs_sec')
            EnergyCutCrsSec.saveData2H5grp(energy_cut_crs_sec, subgrp)
        return

    @staticmethod
    def saveData2H5grp(energy_cut_crs_sec: EnergyCutCrsSec, h5grp: h5.Group) -> None:
        super(EnergyCutCrsSec, EnergyCutCrsSec).saveData2H5grp(energy_cut_crs_sec, h5grp)
        h5grp.create_dataset('eigenvalues', shape=np.shape(energy_cut_crs_sec._eigenvalues),
                             dtype=Real, data=energy_cut_crs_sec._eigenvalues, chunks=True, compression='gzip',
                             compression_opts=9)
        h5grp.create_dataset('energy_level', shape=(1), dtype=Real, data=energy_cut_crs_sec.energy_level)
        h5grp.create_dataset('cross_bands', shape=np.shape(energy_cut_crs_sec.cross_bands),
                             dtype=Int, data=energy_cut_crs_sec.cross_bands, chunks=True, compression='gzip',
                             compression_opts=9)
        return

    def loadVaspdata(self, vasp_data: Vaspdata, num_of_grids: List) -> None:
        EnergyCutCrsSec.loadDataFromVaspdata(self, vasp_data, num_of_grids)
        return

    @staticmethod
    def loadDataFromVaspdata(energy_cut_crs_sec: EnergyCutCrsSec, vasp_data: Vaspdata, num_of_grids: List) -> None:
        super(EnergyCutCrsSec, EnergyCutCrsSec).loadDataFromVaspdata(energy_cut_crs_sec, vasp_data, num_of_grids)
        energy_cut_crs_sec._eigenvalues = vasp_data.eigenvalues.reshape(vasp_data.num_of_bands,
                                                                        num_of_grids[1], num_of_grids[0])
        return

    def plotFigure(self, file_name: String) -> None:
        EnergyCutCrsSec.plotFigureForCrsSec(self, file_name)
        return

    @staticmethod
    def plotFigureForCrsSec(energy_cut_crs_sec: EnergyCutCrsSec, file_name: String) -> None:
        fig, ax = initFigure()
        for i in energy_cut_crs_sec.cross_bands:
            ax.contour(energy_cut_crs_sec._KX, energy_cut_crs_sec._KY, energy_cut_crs_sec._eigenvalues[i, :, :],
                       levels=[energy_cut_crs_sec.energy_level])
        ax.set_aspect('equal', adjustable='box')
        #ax.set_xlabel(r"\(k_x\)")
        #ax.set_ylabel(r"\(k_y\)")
        plt.show()
        fig.savefig(file_name)
        plt.close()
        return


class ElecDnstyCrsSec(CrsSec):

    def __init__(self, KX: Array = None, KY: Array = None,
                 eigenvalues: Array = None, energy_level: Real = None,
                 sigma: Array = None) -> None:
        super(ElecDnstyCrsSec, self).__init__(KX, KY)
        self._eigenvalues: Array = eigenvalues
        self.energy_level: Real = energy_level
        self.sigma: Real = sigma
        return

    def readFile(self, file_name: String) -> None:
        ElecDnstyCrsSec.readDataFromFile(self, file_name)
        return

    @staticmethod
    def readDataFromFile(elec_dnsty_crs_sec: ElecDnstyCrsSec, file_name: String) -> None:
        with h5.File(file_name, 'r') as f:
            grp: h5.Group = f['results/elec_dnsty_crs_sec']
            ElecDnstyCrsSec.readDataFromH5grp(elec_dnsty_crs_sec, grp)
        return

    @staticmethod
    def readDataFromH5grp(elec_dnsty_crs_sec: ElecDnstyCrsSec, h5grp: h5.Group) -> None:
        super(ElecDnstyCrsSec, ElecDnstyCrsSec).readDataFromH5grp(elec_dnsty_crs_sec, h5grp)
        elec_dnsty_crs_sec._eigenvalues = np.asarray(h5grp['eigenvalues'], dtype=Real)
        elec_dnsty_crs_sec.energy_level = Real(h5grp['energy_level'][0])
        elec_dnsty_crs_sec.sigma = Real(h5grp['sigma'][0])
        return

    def saveFile(self, file_name: String) -> None:
        ElecDnstyCrsSec.saveData2File(self, file_name)
        return

    @staticmethod
    def saveData2File(elec_dnsty_crs_sec: ElecDnstyCrsSec, file_name: String) -> None:
        with h5.File(file_name, 'a') as f:
            grp: h5.Group = f.require_group('results')
            if 'results/elec_dnsty_crs_sec' in f:
                del f['results/elec_dnsty_crs_sec']
            subgrp: h5.Group = grp.create_group('elec_dnsty_crs_sec')
            ElecDnstyCrsSec.saveData2H5grp(elec_dnsty_crs_sec, subgrp)
        return

    @staticmethod
    def saveData2H5grp(elec_dnsty_crs_sec: ElecDnstyCrsSec, h5grp: h5.Group) -> None:
        super(ElecDnstyCrsSec, ElecDnstyCrsSec).saveData2H5grp(elec_dnsty_crs_sec, h5grp)
        h5grp.create_dataset('eigenvalues', shape=np.shape(elec_dnsty_crs_sec._eigenvalues), dtype=Real,
                             data=elec_dnsty_crs_sec._eigenvalues, chunks=True, compression='gzip',
                             compression_opts=9)
        h5grp.create_dataset('energy_level', shape=(1), dtype=Real, data=elec_dnsty_crs_sec.energy_level)
        h5grp.create_dataset('sigma', shape=(1), dtype=Real, data=elec_dnsty_crs_sec.sigma)
        return

    def loadVaspdata(self, vasp_data: Vaspdata, num_of_grids: List) -> None:
        ElecDnstyCrsSec.loadDataFromVaspdata(self, vasp_data, num_of_grids)
        return

    @staticmethod
    def loadDataFromVaspdata(elec_dnsty_crs_sec: CrsSec, vasp_data: Vaspdata, num_of_grids: List) -> None:
        super(ElecDnstyCrsSec, ElecDnstyCrsSec).loadDataFromVaspdata(elec_dnsty_crs_sec, vasp_data, num_of_grids)
        elec_dnsty_crs_sec._eigenvalues = vasp_data.eigenvalues.reshape(vasp_data.num_of_bands,
                                                                        num_of_grids[1], num_of_grids[0])
        return

    def plotFigure(self, file_name: String, band_indices: List = None) -> None:
        ElecDnstyCrsSec.plotFigureForCrsSec(self, file_name, band_indices)
        return

    @staticmethod
    def plotFigureForCrsSec(elec_dnsty_crs_sec: ElecDnstyCrsSec, file_name: String, band_indices: List = None) -> None:
        fig, ax = initFigure()
        divnorm = mcolors.TwoSlopeNorm(vmin=0., vcenter=7.5, vmax=15.)
        elec_dnsty: Array = ElecDnstyCrsSec.genElecDnstyForCrsSec(elec_dnsty_crs_sec, band_indices)
        cont = ax.contourf(elec_dnsty_crs_sec._KX, elec_dnsty_crs_sec._KY, elec_dnsty,
                           cmap='viridis', norm=divnorm, levels=50)
        cbar = fig.colorbar(cont, ax=ax)
        cbar.ax.set_ylabel(r"\(\rho\)")
        ax.set_aspect('equal', adjustable='box')
        #ax.set_xlabel(r"\(k_x\)")
        #ax.set_ylabel(r"\(k_y\)")
        plt.show()
        fig.savefig(file_name)
        plt.close()
        return

    @staticmethod
    def genElecDnstyForCrsSec(elec_dnsty_crs_sec: ElecDnstyCrsSec, band_indices: List = None) -> Array:
        if band_indices is None:
            data: Array = elec_dnsty_crs_sec._eigenvalues
        else:
            data: Array = elec_dnsty_crs_sec._eigenvalues[band_indices]
        gauss_filter: GaussFilter = GaussFilter()
        gauss_filter.eigenvalues = data
        gauss_filter.energy_level = elec_dnsty_crs_sec.energy_level
        gauss_filter.sigma = elec_dnsty_crs_sec.sigma
        gauss_filter.do_calculation()
        elec_dnsty: Array = np.sum(gauss_filter.f, axis=0)
        return elec_dnsty

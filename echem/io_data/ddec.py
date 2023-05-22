from matplotlib import colors
import numpy as np
from monty.re import regrep
from echem.core.structure import Structure
from nptyping import NDArray, Shape, Number
from dataclasses import dataclass
from echem.core.constants import ElemNum2Name, Bohr2Angstrom


@dataclass()
class LocalMultipoleMoments:
    net_charges: NDArray[Shape['Natoms'], Number]
    dipoles: NDArray[Shape['Natoms, 4'], Number]
    quadrupoles: NDArray[Shape['Natoms, 8'], Number]


class Output_DDEC:
    def __init__(self,
                 structure: Structure,
                 lmm_hirshfeld: LocalMultipoleMoments,
                 lmm_ddec: LocalMultipoleMoments,
                 charges_cm5: NDArray[Shape['Natons'], Number]):

        self.structure = structure
        self.lmm_hirshfeld = lmm_hirshfeld
        self.lmm_ddec = lmm_ddec
        self.charges_cm5 = charges_cm5

    @staticmethod
    def _process_lmm_(data, line_number, natoms):
        charges_ddec = np.zeros(natoms)
        dipoles_ddec = np.zeros((natoms, 4))
        quadrupoles_ddec = np.zeros((natoms, 8))

        idx = 0
        while len(line := data[line_number].split()) != 0:
            charges_ddec[idx] = float(line[5])
            dipoles_ddec[idx] = list(map(float, line[6: 10]))
            quadrupoles_ddec[idx] = list(map(float, line[10:]))

            line_number += 1
            idx += 1

        return LocalMultipoleMoments(charges_ddec, dipoles_ddec, quadrupoles_ddec)

    @staticmethod
    def from_file(filepath):
        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'lattice': r' parameters',
                    'lmm': r'Multipole analysis for each of the expansion sites.',
                    'cm5': r'The computed CM5 net atomic charges are:'}
        matches = regrep(filepath, patterns)

        lattice = np.zeros((3, 3))

        i = matches['lattice'][0][1]
        natoms = int((data[i + 1].split()[0]).split('.')[0])

        line = data[i + 2].split()
        NX = int(line[0].split('.')[0])
        lattice[0] = np.array([float(line[1]), float(line[2]), float(line[3])])
        line = data[i + 3].split()
        NY = int(line[0].split('.')[0])
        lattice[1] = np.array([float(line[1]), float(line[2]), float(line[3])])
        line = data[i + 4].split()
        NZ = int(line[0].split('.')[0])
        lattice[2] = np.array([float(line[1]), float(line[2]), float(line[3])])

        if NX > 0 and NY > 0 and NZ > 0:
            units = 'Bohr'
        elif NX < 0 and NY < 0 and NZ < 0:
            units = 'Angstrom'
        else:
            raise ValueError('The sign of the number of all voxels should be > 0 or < 0')

        if units == 'Angstrom':
            NX, NY, NZ = -NX, -NY, -NZ

        lattice = lattice * np.array([NX, NY, NZ]).reshape((-1, 1)) * Bohr2Angstrom

        coords = np.zeros((natoms, 3))
        species = []
        line_number = matches['lmm'][0][1] + 3
        idx = 0
        while len(line := data[line_number].split()) != 0:
            species.append(ElemNum2Name[int(line[1])])
            coords[idx] = list(map(float, line[2:5]))
            line_number += 1

        structure = Structure(lattice, species, coords)

        line_number = matches['lmm'][0][1] + 3
        lmm_hirshfeld = Output_DDEC._process_lmm_(data, line_number, natoms)
        line_number = matches['lmm'][1][1] + 3
        lmm_ddec = Output_DDEC._process_lmm_(data, line_number, natoms)

        line_number = matches['cm5'][0][1] + 1
        charges_cm5 = []
        i = 0
        while i < natoms:
            charges = list(map(float, data[line_number].split()))
            charges_cm5 += charges
            line_number += 1
            i += len(charges)

        return Output_DDEC(structure, lmm_hirshfeld, lmm_ddec, np.array(charges_cm5))


class AtomicNetCharges:
    """Class that operates with DDEC output file DDEC6_even_tempered_net_atomic_charges.xyz"""
    def __init__(self, structure: Structure, net_charges, dipoles_xyz=None,
                 dipoles_mag=None, Qs=None, quadrupole_tensor_eigs=None, date=None):
        """
        Create a DDEC class object.
        Args:
            structure (Structure class): a base class that contains lattice, coords and species information
            net_charges:
            dipoles_xyz:
            dipoles_mag:
            Qs:
            quadrupole_tensor_eigs:
        """
        self.structure = structure
        self.net_charges = net_charges
        self.dipoles_xyz = dipoles_xyz
        self.dipoles_mag = dipoles_mag
        self.Qs = Qs
        self.quadrupole_tensor_eigs = quadrupole_tensor_eigs
        self.date = date

    @staticmethod
    def from_file(filepath):
        """
        Read the positions of atoms and theirs charges
        from file "DDEC6_even_tempered_net_atomic_charges.xyz"

        Parameters:
        ----------
        filepath: str
            Path to file with atomic charges

        Returns:
        -------
        DDEC class instance
        """
        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'date': r'\s+(\d\d\d\d/\d\d/\d\d\s+\d\d:\d\d:\d\d)'}
        matches = regrep(filepath, patterns)
        date = matches['date'][0][0][0]

        natoms = int(data[0])
        x_axis = data[1].split()[10:13]
        y_axis = data[1].split()[15:18]
        z_axis = data[1].split()[20:23]
        lattice = np.array([x_axis, y_axis, z_axis], dtype=np.float32)

        for start_line, string in enumerate(data):
            if 'The following XYZ coordinates are in angstroms' in string:
                break

        coords = np.zeros((natoms, 3))
        species = []
        net_charges = np.zeros(natoms)
        dipoles_xyz = np.zeros((natoms, 3))
        dipoles_mag = np.zeros(natoms)
        Qs = np.zeros((natoms, 5))
        quadrupole_tensor_eigs = np.zeros((natoms, 3))

        for i, j in enumerate(range(start_line + 2, start_line + 2 + natoms)):
            line_splitted = data[j].split()
            species.append(line_splitted[1])
            coords[i] = line_splitted[2:5]
            net_charges[i] = line_splitted[5]
            dipoles_xyz[i] = line_splitted[6:9]
            dipoles_mag[i] = line_splitted[9]
            Qs[i] = line_splitted[10:15]
            quadrupole_tensor_eigs[i] = line_splitted[15:18]

        structure = Structure(lattice, species, coords, coords_are_cartesian=True)
        return AtomicNetCharges(structure, net_charges, dipoles_xyz, dipoles_mag, Qs, quadrupole_tensor_eigs, date)


class AtomicSpinMoments:
    """Class that operates with DDEC output file DDEC6_even_tempered_atomic_spin_moments.xyz"""
    def __init__(self, structure: Structure, spin_moments, date):
        self.structure = structure
        self.spin_moments = spin_moments
        self.date = date

    @staticmethod
    def from_file(filepath):
        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'date': r'\s+(\d\d\d\d/\d\d/\d\d\s+\d\d:\d\d:\d\d)'}
        matches = regrep(filepath, patterns)
        date = matches['date'][0][0][0]

        natoms = int(data[0])
        x_axis = data[1].split()[10:13]
        y_axis = data[1].split()[15:18]
        z_axis = data[1].split()[20:23]
        lattice = np.array([x_axis, y_axis, z_axis], dtype=np.float32)

        coords = np.zeros((natoms, 3))
        species = []
        spin_moments = np.zeros(natoms)

        for i, j in enumerate(range(2, 2 + natoms)):
            line_splitted = data[j].split()
            species += [line_splitted[0]]
            coords[i] = line_splitted[1:4]
            spin_moments[i] = line_splitted[4]

        structure = Structure(lattice, species, coords, coords_are_cartesian=True)
        return AtomicSpinMoments(structure, spin_moments, date)


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

import numpy as np
from monty.re import regrep
from ..core.structure import Structure
from ..core.constants import Bohr2Angstrom


class Ionpos:
    def __init__(self, structure: Structure):
        pass

    @staticmethod
    def from_file(filepath: str):
        pass


class Output:
    def __init__(self, fft_box_size: np.ndarray,
                 energy_hist: np.ndarray,
                 energy_ionic_hist: np.ndarray,
                 coords_hist: np.ndarray,
                 forces_hist: np.ndarray,
                 structure: Structure):

        self.fft_box_size = fft_box_size
        self.energy_hist = energy_hist
        self.energy_ionic_hist = energy_ionic_hist
        self.coords_hist = coords_hist
        self.forces_hist = forces_hist
        self.structure = structure

    @property
    def forces(self):
        return self.forces_hist[-1]

    @property
    def energy(self):
        return self.energy_ionic_hist[-1]

    @property
    def nisteps(self):
        return len(self.energy_ionic_hist)

    @staticmethod
    def from_file(filepath: str):
        # \TODO Non-Cartesin coods case is not imptemented
        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'natoms': r'Initialized \d+ species with (\d+) total atoms.',
                    'energy': r'ElecMinimize:\s+Iter:\s+\d+\s+\w:\s+([-+]?\d*\.\d*)',
                    'energy_ionic': r'IonicMinimize: Iter:\s+\d+\s+\w:\s+([-+]?\d*\.\d*)',
                    'coords': r'# Ionic positions in cartesian coordinates:',
                    'forces': r'# Forces in Cartesian coordinates:',
                    'fft_box_size': r'Chosen fftbox size, S = \[(\s+\d+\s+\d+\s+\d+\s+)\]',
                    'lattice': r'---------- Initializing the Grid ----------'}
        matches = regrep(filepath, patterns)

        energy_hist = np.array([float(i[0][0]) for i in matches['energy']])
        energy_ionic_hist = np.array([float(i[0][0]) for i in matches['energy_ionic']])

        nisteps = len(energy_ionic_hist)
        natoms = int(matches['natoms'][0][0][0])
        fft_box_size = np.array([int(i) for i in matches['fft_box_size'][0][0][0].split()])

        lattice = np.zeros((3, 3))
        lattice[0] = [float(i) for i in data[matches['lattice'][0][1] + 2].split()[1:4]]
        lattice[1] = [float(i) for i in data[matches['lattice'][0][1] + 3].split()[1:4]]
        lattice[2] = [float(i) for i in data[matches['lattice'][0][1] + 4].split()[1:4]]
        lattice = lattice.T * Bohr2Angstrom

        coords_hist = np.zeros((nisteps, natoms, 3))
        line_numbers_coords = [int(i[1]) + 1 for i in matches['coords']]
        species = []
        atom_number = 0
        while len(line := data[line_numbers_coords[0] + atom_number].split()) > 0:
            species += [line[1]]
            atom_number += 1
        for i, line_number in enumerate(line_numbers_coords):
            atom_number = 0
            while len(line := data[line_number + atom_number].split()) > 0:
                coords_hist[i, atom_number] = [float(line[2]), float(line[3]), float(line[4])]
                atom_number += 1

        forces_hist = np.zeros((nisteps, natoms, 3))
        line_numbers_forces = [int(i[1]) + 1 for i in matches['forces']]
        for i, line_number in enumerate(line_numbers_forces):
            atom_number = 0
            while len(line := data[line_number + atom_number].split()) > 0:
                forces_hist[i, atom_number] = [float(line[2]), float(line[3]), float(line[4])]
                atom_number += 1

        structure = Structure(lattice, species, coords_hist[-1] * Bohr2Angstrom, coords_are_cartesian=True)

        return Output(fft_box_size, energy_hist, energy_ionic_hist, coords_hist, forces_hist, structure)

    def get_forces(self,
                   mod: str = 'mean',
                   diff: bool = False):
        """
        Args:
            mod (str, optional):
                norm - returns the norm of forces along the ionic trajectory
                mean - returns the mean value of forces' norm in simulation cell along the ionic trajectory
                max - returns the max value of forces' norm in simulation cell along the ionic trajectory
            diff:

        Returns:

        """
        if mod == 'norm':
            forces = np.linalg.norm(self.forces_hist, axis=2)
        elif mod == 'mean':
            forces = np.mean(np.linalg.norm(self.forces_hist, axis=2), axis=1)
        elif mod == 'max':
            forces = np.max(np.linalg.norm(self.forces_hist, axis=2), axis=1)
        else:
            raise ValueError(f'mod should be norm/mean/max. You set {mod}')

        if diff:
            return np.abs(forces[1:] - forces[:-1])
        else:
            return forces


class Eigenvals:
    def __init__(self):
        pass

    @staticmethod
    def from_file(filepath):
        data = np.fromfile(filepath, dtype=np.float64)
        return data

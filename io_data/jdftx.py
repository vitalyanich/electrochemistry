import numpy as np
from monty.re import regrep
from ..core.structure import Structure
from ..core.constants import Bohr2Angstrom, Hartree2eV, eV2Hartree
from ..core.ionic_dynamics import IonicDynamics
from . import vasp
from .universal import Cube
from typing import Union, List
import warnings


class Lattice:
    def __init__(self,
                 lattice: np.ndarray):
        self.lattice = lattice

    @staticmethod
    def from_file(filepath: str):
        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'lattice': r'^\s*lattice\s+'}
        matches = regrep(filepath, patterns)

        lattice = []
        i = 0
        while len(lattice) < 9:
            line = data[matches['lattice'][0][1] + i].split()
            for word in line:
                try:
                    word = float(word)
                    lattice.append(word)
                except:
                    pass
            i += 1

        lattice = np.array(lattice).reshape((3, 3))

        return Lattice(lattice)

    def to_file(self, filepath: str):
        file = open(filepath, 'w')

        file.write('lattice \\\n')
        width_coords_float = max(len(str(int(np.max(self.lattice)))), len(str(int(np.min(self.lattice))))) + 16
        for i, vector in enumerate(self.lattice):
            file.write('\t')
            for vector_i in vector:
                file.write(f'{vector_i:{width_coords_float}.15f}  ')
            if i < 2:
                file.write('\\')
            file.write('\n')

        file.close()


class Ionpos:
    def __init__(self,
                 species: List[str],
                 coords: np.ndarray,
                 move_scale: Union[list, np.ndarray] = None):
        self.species = species
        self.coords = coords
        if move_scale is None:
            move_scale = np.ones(len(coords), dtype=int)
        if isinstance(move_scale, list):
            move_scale = np.array(move_scale, dtype=int)
        self.move_scale = move_scale

    @staticmethod
    def from_file(filepath: str):
        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'coords': r'^\s*ion\s+'}
        matches = regrep(filepath, patterns)

        natoms = len(matches['coords'])
        species = []
        coords = np.zeros((natoms, 3))
        move_scale = np.zeros(natoms, dtype=int)
        for i, ion in enumerate(matches['coords']):
            line = data[ion[1]].split()
            species.append(line[1])
            coords[i] = [line[2], line[3], line[4]]
            move_scale[i] = line[5]

        return Ionpos(species, coords, move_scale)

    def to_file(self, filepath: str):
        file = open(filepath, 'w')

        width_species = max([len(sp) for sp in self.species])
        width_coords_float = max(len(str(int(np.max(self.coords)))), len(str(int(np.min(self.coords))))) + 16
        for sp, coord, ms in zip(self.species, self.coords, self.move_scale):
            file.write(f'ion {sp:{width_species}}  ')
            for coord_i in coord:
                file.write(f'{coord_i:{width_coords_float}.15f}  ')
            file.write(f'{ms}\n')

        file.close()

    def convert(self, format, *args):
        if format == 'vasp':
            lattice = np.transpose(args[0].lattice) * Bohr2Angstrom
            return vasp.Poscar(Structure(lattice, self.species, self.coords * Bohr2Angstrom))

    def get_structure(self,
                      lattice: Lattice):
        return Structure(lattice.lattice * Bohr2Angstrom, self.species, self.coords * Bohr2Angstrom)


class Input:
    def __init__(self, structure):
        self.structure = structure

    @staticmethod
    def from_file(filepath: str):
        # \TODO Non-Cartesin coods case is not imptemented
        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'lattice': r'^\s*lattice\s+',
                    'coords': r'^\s*ion\s+'}
        matches = regrep(filepath, patterns)

        species = []
        coords = np.zeros((len(matches['coords']), 3))
        for i, ion in enumerate(matches['coords']):
            line = data[ion[1]].split()
            species.append(line[1])
            coords[i] = [line[2], line[3], line[4]]

        lattice = []
        i = 0
        while len(lattice) < 9:
            line = data[matches['lattice'][0][1] + i].split()
            for word in line:
                try:
                    word = float(word)
                    lattice.append(word)
                except:
                    pass
            i += 1

        lattice = np.array(lattice).reshape((3, 3)).T * Bohr2Angstrom
        structure = Structure(lattice, species, coords * Bohr2Angstrom, coords_are_cartesian=True)
        return Input(structure)


class Output(IonicDynamics):
    def __init__(self,
                 fft_box_size: np.ndarray,
                 #energy_hist: np.ndarray,
                 energy_ionic_hist: dict,
                 coords_hist: np.ndarray,
                 forces_hist: np.ndarray,
                 nelec_hist: np.ndarray,
                 structure: Structure,
                 nbands: int,
                 nkpts: int,
                 mu: float,
                 HOMO: float,
                 LUMO: float):
        super(Output, self).__init__(forces_hist)
        self.fft_box_size = fft_box_size
        #self.energy_hist = energy_hist
        self.energy_ionic_hist = energy_ionic_hist
        self.coords_hist = coords_hist
        self.nelec_hist = nelec_hist
        self.structure = structure
        self.nbands = nbands
        self.nkpts = nkpts
        self.mu = mu
        self.HOMO = HOMO
        self.LUMO = LUMO

    @property
    def energy(self):
        if 'G' in self.energy_ionic_hist.keys():
            return self.energy_ionic_hist['G'][-1]
        else:
            return self.energy_ionic_hist['F'][-1]

    @property
    def nisteps(self):
        return len(self.energy_ionic_hist['F'])

    @staticmethod
    def from_file(filepath: str):
        # \TODO Non-Cartesin coods case is not imptemented
        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'natoms': r'Initialized \d+ species with (\d+) total atoms.',
                    #'energy': r'ElecMinimize:\s+Iter:\s+\d+\s+\w:\s+([-+]?\d*\.\d*)',
                    #'energy_ionic': r'IonicMinimize: Iter:\s+\d+\s+\w:\s+([-+]?\d*\.\d*)',
                    'coords': r'# Ionic positions in cartesian coordinates:',
                    'forces': r'# Forces in Cartesian coordinates:',
                    'fft_box_size': r'Chosen fftbox size, S = \[(\s+\d+\s+\d+\s+\d+\s+)\]',
                    'lattice': r'---------- Initializing the Grid ----------',
                    'nbands': r'nBands:\s+(\d+)',
                    'nkpts': r'Reduced to (\d+) k-points under symmetry',
                    'nelec': r'nElectrons:\s+(\d+.\d+)',
                    'mu': r'\s+mu\s+:\s+([-+]?\d*\.\d*)',
                    'HOMO': r'\s+HOMO\s*:\s+([-+]?\d*\.\d*)',
                    'LUMO': r'\s+LUMO\s*:\s+([-+]?\d*\.\d*)',
                    'F': r'\s+F\s+=\s+([-+]?\d*\.\d*)',
                    'muN': r'\s+muN\s+=\s+([-+]?\d*\.\d*)',
                    'G': r'\s+G\s+=\s+([-+]?\d*\.\d*)'}

        matches = regrep(filepath, patterns)

        #energy_hist = np.array([float(i[0][0]) for i in matches['energy']])
        #energy_ionic_hist = np.array([float(i[0][0]) for i in matches['energy_ionic']])
        energy_ionic_hist = {}
        F = np.array([float(i[0][0]) for i in matches['F']])
        energy_ionic_hist['F'] = F
        if 'muN' in matches.keys():
            energy_ionic_hist['muN'] = np.array([float(i[0][0]) for i in matches['muN']])
        if 'G' in matches.keys():
            energy_ionic_hist['G'] = np.array([float(i[0][0]) for i in matches['G']])

        nelec_hist = np.array([float(i[0][0]) for i in matches['nelec']])

        #nisteps = len(energy_ionic_hist['F'])
        natoms = int(matches['natoms'][0][0][0])
        nbands = int(matches['nbands'][0][0][0])
        nkpts = int(matches['nkpts'][0][0][0])
        mu = float(matches['mu'][0][0][0])
        HOMO = float(matches['HOMO'][0][0][0])
        LUMO = float(matches['LUMO'][0][0][0])
        fft_box_size = np.array([int(i) for i in matches['fft_box_size'][0][0][0].split()])

        lattice = np.zeros((3, 3))
        lattice[0] = [float(i) for i in data[matches['lattice'][0][1] + 2].split()[1:4]]
        lattice[1] = [float(i) for i in data[matches['lattice'][0][1] + 3].split()[1:4]]
        lattice[2] = [float(i) for i in data[matches['lattice'][0][1] + 4].split()[1:4]]
        lattice = lattice.T * Bohr2Angstrom

        line_numbers_coords = [int(i[1]) + 1 for i in matches['coords']]
        coords_hist = np.zeros((len(line_numbers_coords), natoms, 3))
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

        line_numbers_forces = [int(i[1]) + 1 for i in matches['forces']]
        forces_hist = np.zeros((len(line_numbers_forces), natoms, 3))
        for i, line_number in enumerate(line_numbers_forces):
            atom_number = 0
            while len(line := data[line_number + atom_number].split()) > 0:
                forces_hist[i, atom_number] = [float(line[2]), float(line[3]), float(line[4])]
                atom_number += 1

        structure = Structure(lattice, species, coords_hist[-1] * Bohr2Angstrom, coords_are_cartesian=True)

        return Output(fft_box_size, energy_ionic_hist, coords_hist, forces_hist, nelec_hist,
                      structure, nbands, nkpts, mu, HOMO, LUMO)


class EBS_data:

    @staticmethod
    def from_file(filepath: str,
                  output: Output):
        data = np.fromfile(filepath, dtype=np.float64)
        if len(data) % (output.nkpts * output.nbands) != 0:
            raise ValueError(
                f'Number of eigenvalues should be equal to nspin * nkpts * nbands, but now {output.nkpts=},'
                f'{output.nbands=}, and data has {len(data)} values')
        nspin = len(data) // (output.nkpts * output.nbands)
        data = data.reshape(nspin, output.nkpts, output.nbands)

        return data


class Eigenvals(EBS_data):
    def __init__(self,
                 eigenvalues: np.ndarray,
                 units: str):
        self.eigenvalues = eigenvalues
        self.units = units

    @staticmethod
    def from_file(filepath: str,
                  output: Output):
        eigenvalues = super(Eigenvals, Eigenvals).from_file(filepath, output)
        return Eigenvals(eigenvalues, 'Hartree')

    def mod_to_eV(self):
        if self.units == 'eV':
            print('Units are already eV')
        else:
            self.eigenvalues *= Hartree2eV
            self.units = 'eV'

    def mod_to_Ha(self):
        if self.units == 'Hartree':
            print('Units are already Hartree')
        else:
            self.eigenvalues *= eV2Hartree
            self.units = 'Hartree'


class Fillings(EBS_data):
    def __init__(self,
                 occupations: np.ndarray):
        self.occupations = occupations

    @staticmethod
    def from_file(filepath: str,
                  output: Output):
        occupations = super(Fillings, Fillings).from_file(filepath, output)
        return Fillings(occupations)


class VolumetricData:
    def __init__(self,
                 data: np.ndarray,
                 structure: Structure):
        self.data = data
        self.structure = structure

    def __add__(self, other):
        assert isinstance(other, VolumetricData), 'Other object must belong to VolumetricData class'
        assert self.data.shape == other.data.shape, f'Shapes of two data arrays must be the same but they are ' \
                                                    f'{self.data.shape} and {other.data.shape}'
        if self.structure != other.structure:
            warnings.warn('Two VolumetricData instances contain different Staructures. '
                          'The Structure will be taken from the 2nd (other) instance. '
                          'Hope you know, what you are doing')
        return VolumetricData(self.data + other.data, other.structure)

    def __sub__(self, other):
        assert isinstance(other, VolumetricData), 'Other object must belong to VolumetricData class'
        assert self.data.shape == other.data.shape, f'Shapes of two data arrays must be the same but they are ' \
                                                    f'{self.data.shape} and {other.data.shape}'
        if self.structure != other.structure:
            warnings.warn('Two VolumetricData instances contain different Staructures. '
                          'The Structure will be taken from the 2nd (other) instance. '
                          'Hope you know, what you are doing')
        return VolumetricData(self.data - other.data, other.structure)

    @staticmethod
    def from_file(filepath: str,
                  fft_box_size: np.ndarray,
                  structure: Structure):
        data = np.fromfile(filepath, dtype=np.float64)
        data = data.reshape(fft_box_size)
        return VolumetricData(data, structure)

    def convert_to_cube(self):
        return Cube(self.data, self.structure, np.zeros(3))


class kPts:
    def __init__(self, weights):
        self.weights = weights

    @staticmethod
    def from_file(filepath):
        file = open(filepath, 'r')
        data = file.readlines()
        file.close()
        weights = []
        spin = data[0].split()[8]
        for line in data:
            line_split = line.split()
            if spin != line_split[8]:
                break
            weights.append(float(line.split()[6]))
        weights = np.array(weights)
        return kPts(weights)

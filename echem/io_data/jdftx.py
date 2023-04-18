import numpy as np
from monty.re import regrep
from echem.core.structure import Structure
from echem.core.constants import Bohr2Angstrom, Hartree2eV, eV2Hartree
from echem.core.ionic_dynamics import IonicDynamics
from echem.core.electronic_structure import EBS
from echem.core.thermal_properties import Thermal_properties
from echem.io_data import vasp
from echem.io_data.universal import Cube
from typing import Union, Literal, TypedDict
from typing_extensions import NotRequired
from pathlib import Path
import warnings
import copy
from nptyping import NDArray, Shape, Number


class Lattice:
    def __init__(self,
                 lattice: NDArray[Shape['3, 3'], Number]):
        self.lattice = lattice

    @staticmethod
    def from_file(filepath: str | Path):

        if isinstance(filepath, str):
            filepath = Path(filepath)

        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'lattice': r'^\s*lattice\s+'}
        matches = regrep(str(filepath), patterns)

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
                 species: list[str],
                 coords: NDArray[Shape['Natoms, 3'], Number],
                 move_scale: list[int] | NDArray[Shape['Natoms'], Number] = None,
                 constraint_type: list[Literal['HyperPlane', 'Linear', 'None', 'Planar'] | None] = None,
                 constraint_params: list[list[float] | None] = None):

        self.species = species
        self.coords = coords
        if move_scale is None:
            move_scale = np.ones(len(coords), dtype=int)
        elif isinstance(move_scale, list):
            move_scale = np.array(move_scale, dtype=int)
        self.move_scale = move_scale
        self.constraint_type = constraint_type
        self.constraint_params = constraint_params

    @staticmethod
    def from_file(filepath: str | Path):
        if isinstance(filepath, str):
            filepath = Path(filepath)

        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'coords': r'^\s*ion\s+'}
        matches = regrep(str(filepath), patterns)

        natoms = len(matches['coords'])
        species = []
        coords = np.zeros((natoms, 3))
        move_scale = np.zeros(natoms, dtype=int)
        constraint_type = []
        constraint_params = []

        for i, ion in enumerate(matches['coords']):
            line = data[ion[1]].split()
            species.append(line[1])
            coords[i] = [line[2], line[3], line[4]]
            move_scale[i] = line[5]
            if len(line) > 6:
                constraint_type.append(line[6])
                constraint_params.append([float(line[7]), float(line[8]), float(line[9])])
            else:
                constraint_type.append(None)
                constraint_params.append(None)

        return Ionpos(species, coords, move_scale, constraint_type, constraint_params)

    def to_file(self,
                filepath: str | Path) -> None:
        if isinstance(filepath, str):
            filepath = Path(filepath)

        file = open(filepath, 'w')

        width_species = max([len(sp) for sp in self.species])
        width_coords_float = max(len(str(int(np.max(self.coords)))), len(str(int(np.min(self.coords))))) + 16

        if self.constraint_params is None and self.constraint_type is None:
            for sp, coord, ms in zip(self.species, self.coords, self.move_scale):
                file.write(f'ion {sp:{width_species}}  ')
                for coord_i in coord:
                    file.write(f'{coord_i:{width_coords_float}.15f}  ')
                file.write(f'{ms}\n')
        elif self.constraint_params is not None and self.constraint_type is not None:
            for sp, coord, ms, ctype, cparams in zip(self.species, self.coords, self.move_scale,
                                                     self.constraint_type, self.constraint_params):
                file.write(f'ion {sp:{width_species}}  ')
                for coord_i in coord:
                    file.write(f'{coord_i:{width_coords_float}.15f}  ')
                if ctype is None:
                    file.write(f'{ms}\n')
                else:
                    file.write(f'{ms}  ')
                    file.write(f'{ctype}  ')
                    file.write(f'{cparams[0]}  {cparams[1]}  {cparams[2]}\n')
        else:
            raise ValueError('constraint_type and constraint_params must be both specified or both be None')

        file.close()

    def convert(self,
                format: Literal['vasp'], *args):
        if format == 'vasp':
            lattice = np.transpose(args[0].lattice) * Bohr2Angstrom
            return vasp.Poscar(Structure(lattice, self.species, self.coords * Bohr2Angstrom))
        else:
            raise NotImplemented('Currently only format=vasp is supported')

    def get_structure(self,
                      lattice: Lattice) -> Structure:
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


class EnergyIonicHist(TypedDict):
    F: NDArray[Shape['Nsteps'], Number]
    G: NotRequired[NDArray[Shape['Nsteps'], Number]]
    muN: NotRequired[NDArray[Shape['Nsteps'], Number]]


class Output(IonicDynamics):
    def __init__(self,
                 fft_box_size: NDArray[Shape['3'], Number],
                 energy_ionic_hist: EnergyIonicHist,
                 coords_hist: NDArray[Shape['Nsteps, Natoms, 3'], Number] | None,
                 forces_hist: NDArray[Shape['Nsteps, Natoms, 3'], Number] | None,
                 nelec_hist: np.ndarray,
                 magnetization_hist: NDArray[Shape['Nesteps, 2'], Number] | None,
                 structure: Structure,
                 nbands: int,
                 nkpts: int,
                 mu: float,
                 HOMO: float,
                 LUMO: float,
                 phonons: dict,
                 pseudopots: dict):
        super(Output, self).__init__(forces_hist)
        self.fft_box_size = fft_box_size
        self.energy_ionic_hist = energy_ionic_hist
        self.coords_hist = coords_hist
        self.nelec_hist = nelec_hist
        self.magnetization_hist = magnetization_hist
        self.structure = structure
        self.nbands = nbands
        self.nkpts = nkpts
        self.mu = mu
        self.HOMO = HOMO
        self.LUMO = LUMO
        self.phonons = phonons
        self.pseudopots = pseudopots
        if phonons['real'] is not None and len(phonons['real']) > 0:
            self.thermal_props = Thermal_properties(np.array([phonons['real']]) * Hartree2eV)

    @property
    def energy(self):
        if 'G' in self.energy_ionic_hist.keys():
            return self.energy_ionic_hist['G'][-1]
        else:
            return self.energy_ionic_hist['F'][-1]

    @property
    def nisteps(self):
        return len(self.energy_ionic_hist['F'])

    @property
    def nelec(self):
        return self.nelec_hist[-1]

    @property
    def nelec_pzc(self):
        return np.sum([self.structure.natoms_by_type[key] * self.pseudopots[key] for key in self.pseudopots.keys()])

    @property
    def magnetization_abs(self):
        if self.magnetization_hist is None:
            raise ValueError('It is non-spin-polarized calculation')
        else:
            return self.magnetization_hist[-1, 0]

    @property
    def magnetization_tot(self):
        if self.magnetization_hist is None:
            raise ValueError('It is non-spin-polarized calculation')
        else:
            return self.magnetization_hist[-1, 1]

    @property
    def nspin(self):
        if self.magnetization_hist is None:
            return 1
        else:
            return 2

    @staticmethod
    def from_file(filepath: str | Path):
        if isinstance(filepath, str):
            filepath = Path(filepath)

        # \TODO Non-Cartesin coods case is not implemented

        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'natoms': r'Initialized \d+ species with (\d+) total atoms.',
                    'coords': r'# Ionic positions in cartesian coordinates:',
                    'forces': r'# Forces in Cartesian coordinates:',
                    'fft_box_size': r'Chosen fftbox size, S = \[(\s+\d+\s+\d+\s+\d+\s+)\]',
                    'lattice': r'---------- Initializing the Grid ----------',
                    'nbands': r'nBands:\s+(\d+)',
                    'nkpts': r'Reduced to (\d+) k-points under symmetry',
                    'nkpts_folded': r'Folded \d+ k-points by \d+x\d+x\d+ to (\d+) k-points.',
                    'is_kpts_irreducable': r'No reducable k-points',
                    'nelec': r'nElectrons:\s+(\d+.\d+)',
                    'magnetization': r'magneticMoment:\s+\[\s+Abs:\s+(\d+.\d+)\s+Tot:\s+([-+]?\d*\.\d*)',
                    'mu': r'\s+mu\s+:\s+([-+]?\d*\.\d*)',
                    'mu_hist': r'mu:\s+([-+]?\d*\.\d*)',
                    'HOMO': r'\s+HOMO\s*:\s+([-+]?\d*\.\d*)',
                    'LUMO': r'\s+LUMO\s*:\s+([-+]?\d*\.\d*)',
                    'F': r'^\s*F\s+=\s+([-+]?\d*\.\d*)',
                    'muN': r'\s+muN\s+=\s+([-+]?\d*\.\d*)',
                    'G': r'\s+G\s+=\s+([-+]?\d*\.\d*)',
                    'phonon report': r'(\d+) imaginary modes, (\d+) modes within cutoff, (\d+) real modes',
                    'zero mode': r'Zero mode \d+:',
                    'imaginary mode': r'Imaginary mode \d+:',
                    'real mode': r'Real mode \d+:',
                    'ionic convergence': r'IonicMinimize: Converged',
                    'pseudopots': r'\s*Title:\s+([a-zA-Z0-9]*).',
                    'valence_elecs': r'(\d+) valence electrons in orbitals'}

        matches = regrep(str(filepath), patterns)

        F = np.array([float(i[0][0]) for i in matches['F']])
        energy_ionic_hist: EnergyIonicHist = {'F': F}
        if 'muN' in matches.keys():
            energy_ionic_hist['muN'] = np.array([float(i[0][0]) for i in matches['muN']])
        if 'G' in matches.keys():
            energy_ionic_hist['G'] = np.array([float(i[0][0]) for i in matches['G']])

        nelec_hist = np.array([float(i[0][0]) for i in matches['nelec']])

        natoms = int(matches['natoms'][0][0][0])
        nbands = int(matches['nbands'][0][0][0])
        if bool(matches['is_kpts_irreducable']):
            nkpts = int(matches['nkpts_folded'][0][0][0])
        else:
            nkpts = int(matches['nkpts'][0][0][0])
        if bool(matches['mu']):
            mu = float(matches['mu'][0][0][0])
        else:
            mu = float(matches['mu_hist'][-1][0][0])
        if bool(matches['HOMO']):
            HOMO = float(matches['HOMO'][0][0][0])
        else:
            HOMO = None
        if bool(matches['LUMO']):
            LUMO = float(matches['LUMO'][0][0][0])
        else:
            LUMO = None
        if bool(matches['magnetization']):
            magnetization_hist = np.zeros((len(matches['magnetization']), 2))
            for i, mag in enumerate(matches['magnetization']):
                magnetization_hist[i] = [float(mag[0][0]), float(mag[0][1])]
        else:
            magnetization_hist = None
        fft_box_size = np.array([int(i) for i in matches['fft_box_size'][0][0][0].split()])

        lattice = np.zeros((3, 3))
        lattice[0] = [float(i) for i in data[matches['lattice'][0][1] + 2].split()[1:4]]
        lattice[1] = [float(i) for i in data[matches['lattice'][0][1] + 3].split()[1:4]]
        lattice[2] = [float(i) for i in data[matches['lattice'][0][1] + 4].split()[1:4]]
        lattice = lattice.T * Bohr2Angstrom

        if matches['coords']:
            line_numbers = [int(i[1]) + 1 for i in matches['coords']]
            coords_hist = np.zeros((len(line_numbers), natoms, 3))
            species = []
            atom_number = 0
            while len(line := data[line_numbers[0] + atom_number].split()) > 0:
                species += [line[1]]
                atom_number += 1
            for i, line_number in enumerate(line_numbers):
                atom_number = 0
                while len(line := data[line_number + atom_number].split()) > 0:
                    coords_hist[i, atom_number] = [float(line[2]), float(line[3]), float(line[4])]
                    atom_number += 1

        if matches['forces']:
            line_numbers = [int(i[1]) + 1 for i in matches['forces']]
            forces_hist = np.zeros((len(line_numbers), natoms, 3))
            for i, line_number in enumerate(line_numbers):
                atom_number = 0
                while len(line := data[line_number + atom_number].split()) > 0:
                    forces_hist[i, atom_number] = [float(line[2]), float(line[3]), float(line[4])]
                    atom_number += 1
        else:
            forces_hist = None

        if not matches['ionic convergence'] and not matches['phonon report']:
            warnings.warn(f'Ionic Minimization has not been converged! {filepath}')

        if matches['phonon report']:
            freq_report = {key: int(i) for key, i in zip(['imaginary modes', 'modes within cutoff', 'real modes'],
                                                         matches['phonon report'][0][0])}
            if freq_report['modes within cutoff']:
                line_numbers = [int(i[1]) + 1 for i in matches['zero mode']]
                zero_mode_freq = np.zeros(freq_report['modes within cutoff'], dtype=complex)
                for i, line_number in enumerate(line_numbers):
                    zero_mode_freq[i] = complex(data[line_number].split()[1].replace('i', 'j'))
            else:
                zero_mode_freq = []

            if freq_report['imaginary modes']:
                line_numbers = [int(i[1]) + 1 for i in matches['imaginary mode']]
                imag_mode_freq = np.zeros(freq_report['imaginary modes'], dtype=complex)
                for i, line_number in enumerate(line_numbers):
                    imag_mode_freq[i] = complex(data[line_number].split()[1].replace('i', 'j'))
            else:
                imag_mode_freq = []

            if freq_report['real modes']:
                line_numbers = [int(i[1]) + 1 for i in matches['real mode']]
                real_mode_freq = np.zeros(freq_report['real modes'])
                for i, line_number in enumerate(line_numbers):
                    real_mode_freq[i] = float(data[line_number].split()[1])
            else:
                real_mode_freq = []

            phonons = {'zero': zero_mode_freq, 'imag': imag_mode_freq, 'real': real_mode_freq}

            matches = regrep(str(filepath), {'ions': r'ion\s+([a-zA-Z]+)\s+[-+]?\d*\.\d*',
                                             'coords': r'ion\s+[a-zA-Z]+\s+([-+]?\d*\.\d*)\s+([-+]?\d*\.\d*)\s+([-+]?\d*\.\d*)'})
            species = [i[0][0] for i in matches['ions']]

            coords_hist = [[[float(i) for i in coord[0]] for coord in matches['coords']]]
            coords_hist = np.array(coords_hist)

        else:
            phonons = {'zero': None, 'imag': None, 'real': None}

        structure = Structure(lattice, species, coords_hist[-1] * Bohr2Angstrom, coords_are_cartesian=True)

        pseudopots = {i[0][0]: int(j[0][0]) for i, j in zip(matches['pseudopots'], matches['valence_elecs'])}

        return Output(fft_box_size, energy_ionic_hist, coords_hist, forces_hist, nelec_hist, magnetization_hist,
                      structure, nbands, nkpts, mu, HOMO, LUMO, phonons, pseudopots)

    def get_xdatcar(self):
        transform = np.linalg.inv(self.structure.lattice)
        return vasp.Xdatcar(structure=self.structure,
                            trajectory=np.matmul(self.coords_hist * Bohr2Angstrom, transform))

    def get_poscar(self):
        structure = copy.copy(self.structure)
        structure.coords = self.coords_hist[0] * Bohr2Angstrom
        return vasp.Poscar(structure=structure)

    def get_contcar(self):
        return vasp.Poscar(structure=self.structure)


class EBS_data:

    @staticmethod
    def from_file(filepath: str | Path,
                  output: Output) -> NDArray[Shape['Nspin, Nkpts, Nbands'], Number]:

        if isinstance(filepath, Path):
            filepath = Path(filepath)

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
                 eigenvalues: NDArray[Shape['Nspin, Nkpts, Nbands'], Number],
                 units: Literal['eV', 'Hartree']):
        self.eigenvalues = eigenvalues
        self.units = units

    @staticmethod
    def from_file(filepath: str | Path,
                  output: Output):

        if isinstance(filepath, Path):
            filepath = Path(filepath)

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
    def from_file(filepath: str | Path,
                  output: Output):

        if isinstance(filepath, Path):
            filepath = Path(filepath)

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
            warnings.warn('Two VolumetricData instances contain different Structures. '
                          'The Structure will be taken from the 2nd (other) instance. '
                          'Hope you know, what you are doing')
        return VolumetricData(self.data + other.data, other.structure)

    def __sub__(self, other):
        assert isinstance(other, VolumetricData), 'Other object must belong to VolumetricData class'
        assert self.data.shape == other.data.shape, f'Shapes of two data arrays must be the same but they are ' \
                                                    f'{self.data.shape} and {other.data.shape}'
        if self.structure != other.structure:
            warnings.warn('Two VolumetricData instances contain different Structures. '
                          'The Structure will be taken from the 2nd (other) instance. '
                          'Hope you know, what you are doing')
        return VolumetricData(self.data - other.data, other.structure)

    @staticmethod
    def from_file(filepath: str | Path,
                  fft_box_size: NDArray[Shape['3'], Number],
                  structure: Structure):

        if isinstance(filepath, Path):
            filepath = Path(filepath)

        data = np.fromfile(filepath, dtype=np.float64)
        data = data.reshape(fft_box_size)
        return VolumetricData(data, structure)

    def convert_to_cube(self) -> Cube:
        return Cube(self.data, self.structure, np.zeros(3))


class kPts:
    def __init__(self,
                 weights: NDArray[Shape['Nkpts'], Number]):
        self.weights = weights

    @staticmethod
    def from_file(filepath: str | Path):

        if isinstance(filepath, Path):
            filepath = Path(filepath)

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


class BandProjections:
    def __init__(self,
                 proj_coeffs: NDArray[Shape['Nspin, Nkpts, Nbands, Norbs'], Number],
                 weights: NDArray[Shape['Nkpts'], Number],
                 species: list[str],
                 norbs_per_atomtype: dict,
                 orbs_names: list[str],
                 orbs_data: list[dict]):
        self.proj_coeffs = proj_coeffs
        self.weights = weights
        self.species = species
        self.norbs_per_atomtype = norbs_per_atomtype
        self.orbs_names = orbs_names
        self.orbs_data = orbs_data

        self.eigenvalues = None

    @property
    def nspin(self):
        return self.proj_coeffs.shape[0]

    @property
    def nkpts(self):
        return self.proj_coeffs.shape[1]

    @property
    def nbands(self):
        return self.proj_coeffs.shape[3]

    @property
    def norbs(self):
        return self.proj_coeffs.shape[4]

    @staticmethod
    def from_file(filepath: str | Path):

        if isinstance(filepath, str):
            filepath = Path(filepath)

        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'x': r'#\s+\d+'}
        matches = regrep(str(filepath), patterns)

        nstates = int(data[0].split()[0])
        nbands = int(data[0].split()[2])
        norbs = int(data[0].split()[4])

        if 'spin' in data[int(matches['x'][0][1])]:
            nspin = 2
        else:
            nspin = 1

        nkpts = int(nstates / nspin)

        proj_coeffs = np.zeros((nspin, nkpts, nbands, norbs))
        weights = np.zeros(nstates)

        start_lines = []
        for i, match in enumerate(matches['x']):
            start_lines.append(int(match[1]))
            weights[i] = float(data[int(match[1])].split()[7])

        if not np.array_equal(weights[:len(weights) // 2], weights[len(weights) // 2:]):
            raise ValueError(f'Kpts weights can not be correctly split {weights=}')

        weights = weights[:len(weights) // 2]

        species = []
        norbs_per_atomtype = {}
        orbs_names = []
        orbs_data = []

        idx_atom = -1
        for iline in range(2, start_lines[0]):
            line = data[iline].split()
            atomtype = line[0]
            natoms_per_atomtype = int(line[1])
            species += [atomtype] * natoms_per_atomtype
            norbs_per_atomtype[line[0]] = int(line[2])

            l_max = int(line[3])
            nshalls_per_l = []
            for i in range(l_max + 1):
                nshalls_per_l.append(int(line[4 + i]))

            for i in range(natoms_per_atomtype):
                idx_atom += 1
                for l, n_max in zip(range(l_max + 1), nshalls_per_l):
                    for n in range(n_max):
                        if l == 0:
                            orbs_names.append(f'{idx_atom} {atomtype} s {n + 1}({n_max})')
                            orbs_data.append({'atom_type': atomtype,
                                              'atom_index': idx_atom,
                                              'l': l,
                                              'm': 0,
                                              'orb_name': 's'})
                        elif l == 1:
                            for m, m_name in zip([-1, 0, 1], ['p_x', 'p_y', 'p_z']):
                                orbs_names.append(f'{idx_atom} {atomtype} {m_name} {n + 1}({n_max})')
                                orbs_data.append({'atom_type': atomtype,
                                                  'atom_index': idx_atom,
                                                  'l': l,
                                                  'm': m,
                                                  'orb_name': m_name})
                        elif l == 2:
                            for m, m_name in zip([-2, -1, 0, 1, 2], ['d_xy', 'd_yz', 'd_z^2', 'd_xz', 'd_x^2-y^2']):
                                orbs_names.append(f'{idx_atom} {atomtype} {m_name} {n + 1}({n_max})')
                                orbs_data.append({'atom_type': atomtype,
                                                  'atom_index': idx_atom,
                                                  'l': l,
                                                  'm': m,
                                                  'orb_name': m_name})
                        elif l > 2:
                            raise NotImplementedError('Only s, p snd d orbitals are currently supported')

        ikpt_major = -1
        ikpt_minor = -1
        for istate, (start, stop) in enumerate(zip(start_lines[:-1], start_lines[1:])):
            if nspin == 2:
                if data[start].split()[9] == '+1;':
                    ispin = 0
                    ikpt_major += 1
                    ikpt = ikpt_major
                elif data[start].split()[9] == '-1;':
                    ispin = 1
                    ikpt_minor += 1
                    ikpt = ikpt_minor
                else:
                    raise ValueError(f'Can\'t determine spin in string {data[start].split()}')
            else:
                ispin = 0
                ikpt = istate

            for iband, line in enumerate(range(start + 1, stop)):
                proj_coeffs[ispin, ikpt, iband] = [float(k) for k in data[line].split()]

        return BandProjections(proj_coeffs, weights, species, norbs_per_atomtype, orbs_names, orbs_data)

    def get_PDOS(self,
                 atom_numbers: list[int] | int,
                 eigenvals: Eigenvals,
                 get_orbs_names: bool = False,
                 specific_l: int = None,
                 dE: float = 0.01,
                 emin: float = None,
                 emax: float = None,
                 zero_at_fermi: bool = False,
                 sigma: float = 0.02,
                 efermi: float = None) -> Union[tuple[NDArray[Shape['Ngrid'], Number],
                                                      NDArray[Shape['Nspin, Norbs_selected, Ngrid'], Number]],
                                                tuple[NDArray[Shape['Ngrid'], Number],
                                                      NDArray[Shape['Nspin, Norbs_selected, Ngrid'], Number],
                                                      list[str]]]:
        self.eigenvalues = eigenvals.eigenvalues
        if isinstance(atom_numbers, int):
            atom_numbers = [atom_numbers]

        if zero_at_fermi is True and efermi is None:
            raise ValueError('You can not set zero_at_fermi=True if you did not specify efermi value')

        if emin is None:
            emin = np.min(self.eigenvalues) - 1
        if emax is None:
            emax = np.max(self.eigenvalues) + 1

        E_arr = np.arange(emin, emax, dE)
        ngrid = E_arr.shape[0]

        idxs = []
        for atom in atom_numbers:
            start = sum([self.norbs_per_atomtype[i] for i in self.species[:atom]])
            for i in range(self.norbs_per_atomtype[self.species[atom]]):
                idxs.append(start + i)

        if specific_l is not None:
            idxs = [idx for idx in idxs if self.orbs_data[idx]['l'] == specific_l]

        proj_coeffs_weighted = self.proj_coeffs[:, :, :, idxs]

        for spin in range(self.nspin):
            for i, weight_kpt in enumerate(self.weights):
                proj_coeffs_weighted[spin, i] *= weight_kpt

        W_arr = np.moveaxis(proj_coeffs_weighted, [1, 2, 3], [2, 3, 1])
        G_arr = EBS.gaussian_smearing(E_arr, self.eigenvalues, sigma)

        PDOS_arr = np.zeros((self.nspin, len(idxs), ngrid))
        for spin in range(self.nspin):
            for idx in range(len(idxs)):
                PDOS_arr[spin, idx] = np.sum(G_arr[spin, :, :, :] * W_arr[spin, idx, :, :, None],
                                             axis=(0, 1))

        if self.nspin == 1:
            PDOS_arr *= 2

        if get_orbs_names:
            return E_arr, PDOS_arr, [self.orbs_names[i] for i in idxs]
        else:
            return E_arr, PDOS_arr

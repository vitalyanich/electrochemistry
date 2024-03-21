from __future__ import annotations
import numpy as np
from typing import Union, List, Iterable
from monty.re import regrep
from echem.core.structure import Structure
from ..io_data.universal import Cube
from echem.core.electronic_structure import EBS
from echem.core.ionic_dynamics import IonicDynamics
from echem.core.constants import Angstrom2Bohr
from . import jdftx
from pymatgen.io.vasp import Procar as Procar_pmg
from nptyping import NDArray, Shape, Number
from pathlib import Path
import warnings


class Poscar:
    """Class that reads VASP POSCAR files"""
    def __init__(self,
                 structure: Structure,
                 comment: str = None,
                 sdynamics_data: list = None):
        """
        Create a Poscar instance
        Args:
            structure (Structure class): a base class that contains lattice, coords and species information
            comment (str): a VASP comment
            sdynamics_data (list, 2D np.array): data about selective dynamics for each atom. [['T', 'T', 'F'],
            ['F', 'F', 'F'],...]
        """
        self.structure = structure
        self.comment = comment
        self.sdynamics_data = sdynamics_data

    def __repr__(self):
        return f'{self.comment}\n' + repr(self.structure)

    @staticmethod
    def from_file(filepath: str | Path):
        """
        Static method to read a POSCAR file
        Args:
            filepath: path to the POSCAR file

        Returns:
            Poscar class object
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)

        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        comment = data[0].strip()
        scale = float(data[1])
        lattice = np.array([[float(i) for i in line.split()] for line in data[2:5]])
        if scale < 0:
            # In VASP, a negative scale factor is treated as a volume.
            # We need to translate this to a proper lattice vector scaling.
            vol = abs(np.linalg.det(lattice))
            lattice *= (-scale / vol) ** (1 / 3)
        else:
            lattice *= scale

        name_species = data[5].split()
        num_species = [int(i) for i in data[6].split()]
        species = []
        for name, num in zip(name_species, num_species):
            species += [name]*num

        sdynamics_is_used = False
        start_atoms = 8
        if data[7][0] in 'sS':
            sdynamics_is_used = True
            start_atoms = 9

        coords_are_cartesian = False
        if sdynamics_is_used:
            if data[8][0] in 'cCkK':
                coords_are_cartesian = True
        else:
            if data[7][0] in 'cCkK':
                coords_are_cartesian = True

        coords = []
        coords_scale = scale if coords_are_cartesian else 1
        sdynamics_data = list() if sdynamics_is_used else None
        for i in range(start_atoms, start_atoms + np.sum(num_species), 1):
            line = data[i].split()
            coords.append([float(j) * coords_scale for j in line[:3]])
        if sdynamics_is_used:
            for i in range(start_atoms, start_atoms + np.sum(num_species), 1):
                line = data[i].split()
                sdynamics_data.append([j for j in line[3:6]])

        struct = Structure(lattice, species, coords, coords_are_cartesian)

        if sdynamics_is_used:
            return Poscar(struct, comment, sdynamics_data)
        else:
            return Poscar(struct, comment)

    def to_file(self, filepath: str | Path):
        if isinstance(filepath, str):
            filepath = Path(filepath)

        file = open(filepath, 'w')
        file.write(f'{self.comment}\n')
        file.write('1\n')
        for vector in self.structure.lattice:
            file.write(f'  {vector[0]}  {vector[1]}  {vector[2]}\n')

        species = np.array(self.structure.species)
        sorted_order = np.argsort(species, kind='stable')
        unique, counts = np.unique(species, return_counts=True)
        line = '   '
        for u in unique:
            line += u + '  '
        file.write(line + '\n')
        line = '   '
        for c in counts:
            line += str(c) + '  '
        file.write(line + '\n')

        if self.sdynamics_data is not None:
            file.write('Selective dynamics\n')

        if self.structure.coords_are_cartesian:
            file.write('Cartesian\n')
        else:
            file.write('Direct\n')

        if self.sdynamics_data is None:
            for i in sorted_order:
                atom = self.structure.coords[i]
                file.write(f'  {atom[0]}  {atom[1]}  {atom[2]}\n')
        else:
            for i in sorted_order:
                atom = self.structure.coords[i]
                sd_atom = self.sdynamics_data[i]
                file.write(f'  {atom[0]}  {atom[1]}  {atom[2]}  {sd_atom[0]}  {sd_atom[1]}  {sd_atom[2]}\n')

        file.close()

    def convert(self, format):
        if format == 'jdftx':
            self.mod_coords_to_cartesian()
            return jdftx.Ionpos(self.structure.species, self.structure.coords * Angstrom2Bohr), \
                   jdftx.Lattice(np.transpose(self.structure.lattice) * Angstrom2Bohr)
        else:
            raise ValueError('Only format = jdftx is supported')

    def mod_add_atoms(self, coords, species, sdynamics_data=None):
        self.structure.mod_add_atoms(coords, species)
        if sdynamics_data is not None:
            if any(isinstance(el, list) for el in sdynamics_data):
                for sd_atom in sdynamics_data:
                    self.sdynamics_data.append(sd_atom)
            else:
                self.sdynamics_data.append(sdynamics_data)

    def mod_change_atoms(self, ids: Union[int, Iterable],
                         new_coords: Union[Iterable[float], Iterable[Iterable[float]]] = None,
                         new_species: Union[str, List[str]] = None,
                         new_sdynamics_data: Union[Iterable[str], Iterable[Iterable[str]]] = None):
        self.structure.mod_change_atoms(ids, new_coords, new_species)
        if new_sdynamics_data is not None:
            if self.sdynamics_data is None:
                self.sdynamics_data = [['T', 'T', 'T'] for _ in range(self.structure.natoms)]
            if isinstance(ids, Iterable):
                for i, new_sdata in zip(ids, new_sdynamics_data):
                    self.sdynamics_data[i] = new_sdata
            else:
                self.sdynamics_data[ids] = new_sdynamics_data

    def mod_coords_to_box(self):
        assert self.structure.coords_are_cartesian is False, 'This operation allowed only for NON-cartesian coords'
        self.structure.coords %= 1

    def mod_coords_to_direct(self):
        self.structure.mod_coords_to_direct()

    def mod_coords_to_cartesian(self):
        self.structure.mod_coords_to_cartesian()


class Outcar(EBS, IonicDynamics):
    """Class that reads VASP OUTCAR files"""

    def __init__(self,
                 weights: NDArray[Shape['Nkpts'], Number],
                 efermi_hist: NDArray[Shape['Nisteps'], Number],
                 eigenvalues_hist: NDArray[Shape['Nisteps, Nspin, Nkpts, Nbands'], Number],
                 occupations_hist: NDArray[Shape['Nisteps, Nspin, Nkpts, Nbands'], Number],
                 energy_hist: NDArray[Shape['Nallsteps'], Number],
                 energy_ionic_hist: NDArray[Shape['Nisteps'], Number],
                 forces_hist: NDArray[Shape['Nispeps, Natoms, 3'], Number]):
        EBS.__init__(self, eigenvalues_hist[-1], weights, efermi_hist[-1], occupations_hist[-1])
        IonicDynamics.__init__(self, forces_hist, None, None, None)

        self.efermi_hist = efermi_hist
        self.energy_hist = energy_hist
        self.energy_ionic_hist = energy_ionic_hist
        self.eigenvalues_hist = eigenvalues_hist
        self.occupations_hist = occupations_hist

    def __add__(self, other):
        """
        Concatenates Outcar files (all histories). It is useful for ionic optimization.
        If k-point meshes from two Outcars are different, weights, eigenvalues and occupations will be taken
        from the 2nd (other) Outcar instance
        Args:
            other (Outcar class): Outcar that should be added to the current Outcar

        Returns (Outcar class):
            New Outcar with concatenated histories
        """
        assert isinstance(other, Outcar), 'Other object must belong to Outcar class'
        assert self.natoms == other.natoms, 'Number of atoms of two files must be equal'
        if not np.array_equal(self.weights, other.weights):
            warnings.warn('Two Outcar instances have been calculated with different k-point folding. '
                          'Weights, eigenvalues and occupations will be taken from the 2nd (other) instance. '
                          'Hope you know, what you are doing')
            return Outcar(other.weights,
                          np.concatenate((self.efermi_hist, other.efermi_hist)),
                          other.eigenvalues_hist,
                          other.occupations_hist,
                          np.concatenate((self.energy_hist, other.energy_hist)),
                          np.concatenate((self.energy_ionic_hist, other.energy_ionic_hist)),
                          np.concatenate((self.forces_hist, other.forces_hist)))

        return Outcar(other.weights,
                      np.concatenate((self.efermi_hist, other.efermi_hist)),
                      np.concatenate((self.eigenvalues_hist, other.eigenvalues_hist)),
                      np.concatenate((self.occupations_hist, other.occupations_hist)),
                      np.concatenate((self.energy_hist, other.energy_hist)),
                      np.concatenate((self.energy_ionic_hist, other.energy_ionic_hist)),
                      np.concatenate((self.forces_hist, other.forces_hist)))

    @property
    def natoms(self):
        return self.forces.shape[0]

    @property
    def nisteps(self):
        return self.energy_ionic_hist.shape[0]

    @property
    def forces(self):
        return self.forces_hist[-1]

    @property
    def energy(self):
        return self.energy_ionic_hist[-1]

    @staticmethod
    def from_file(filepath: str | Path):
        if isinstance(filepath, str):
            filepath = Path(filepath)

        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'nkpts': r'k-points\s+NKPTS\s+=\s+(\d+)',
                    'nbands': r'number of bands\s+NBANDS=\s+(\d+)',
                    'natoms': r'NIONS\s+=\s+(\d+)',
                    'weights': 'Following reciprocal coordinates:',
                    'efermi': r'E-fermi\s:\s+([-.\d]+)',
                    'energy': r'free energy\s+TOTEN\s+=\s+(.\d+\.\d+)\s+eV',
                    'energy_ionic': r'free  energy\s+TOTEN\s+=\s+(.\d+\.\d+)\s+eV',
                    'kpoints': r'k-point\s+(\d+)\s:\s+[-.\d]+\s+[-.\d]+\s+[-.\d]+\n',
                    'forces': r'\s+POSITION\s+TOTAL-FORCE',
                    'spin': r'spin component \d+\n'}
        matches = regrep(str(filepath), patterns)

        nbands = int(matches['nbands'][0][0][0])
        nkpts = int(matches['nkpts'][0][0][0])
        natoms = int(matches['natoms'][0][0][0])
        energy_hist = np.array([float(i[0][0]) for i in matches['energy']])
        energy_ionic_hist = np.array([float(i[0][0]) for i in matches['energy_ionic']])

        if matches['spin']:
            nspin = 2
        else:
            nspin = 1

        if nkpts == 1:
            weights = np.array([float(data[matches['weights'][0][1] + 2].split()[3])])
        else:
            weights = np.zeros(nkpts)
            for i in range(nkpts):
                weights[i] = float(data[matches['weights'][0][1] + 2 + i].split()[3])
            weights /= np.sum(weights)

        arr = matches['efermi']
        efermi_hist = np.zeros(len(arr))
        for i in range(len(arr)):
            efermi_hist[i] = float(arr[i][0][0])

        nisteps = len(energy_ionic_hist)
        eigenvalues_hist = np.zeros((nisteps, nspin, nkpts, nbands))
        occupations_hist = np.zeros((nisteps, nspin, nkpts, nbands))

        each_kpoint_list = np.array([[int(j[0][0]), int(j[1])] for j in matches['kpoints']])
        for step in range(nisteps):
            for spin in range(nspin):
                for kpoint in range(nkpts):
                    arr = data[each_kpoint_list[nkpts * nspin * step + nkpts * spin + kpoint, 1] + 2:
                               each_kpoint_list[nkpts * nspin * step + nkpts * spin + kpoint, 1] + 2 + nbands]
                    eigenvalues_hist[step, spin, kpoint] = [float(i.split()[1]) for i in arr]
                    occupations_hist[step, spin, kpoint] = [float(i.split()[2]) for i in arr]

        arr = matches['forces']
        forces_hist = np.zeros((nisteps, natoms, 3))
        for step in range(nisteps):
            for atom in range(natoms):
                line = data[arr[step][1] + atom + 2:arr[step][1] + atom + 3]
                line = line[0].split()
                forces_hist[step, atom] = [float(line[3]), float(line[4]), float(line[5])]

        return Outcar(weights, efermi_hist, eigenvalues_hist, occupations_hist,
                      energy_hist, energy_ionic_hist, forces_hist)


class Wavecar:
    """Class that reads VASP WAVECAR files"""
    # TODO: add useful functions for Wavecar class: plot charge density, plot real and imag parts etc.

    def __init__(self, kb_array, wavefunctions, ngrid_factor):
        self.kb_array = kb_array
        self.wavefunctions = wavefunctions
        self.ngrid_factor = ngrid_factor

    @staticmethod
    def from_file(filepath, kb_array, ngrid_factor=1.5):
        from echem.core.vaspwfc_p3 import vaspwfc
        wfc = vaspwfc(filepath)
        wavefunctions = []
        for kb in kb_array:
            kpoint = kb[0]
            band = kb[1]
            wf = wfc.wfc_r(ikpt=kpoint, iband=band, ngrid=wfc._ngrid * ngrid_factor)
            wavefunctions.append(wf)
        return Wavecar(kb_array, wavefunctions, ngrid_factor)


class Procar:
    def __init__(self, proj_koeffs, orbital_names):
        self.proj_koeffs = proj_koeffs
        self.eigenvalues = None
        self.weights = None
        self.nspin = None
        self.nkpts = None
        self.nbands = None
        self.efermi = None
        self.natoms = None
        self.norbs = proj_koeffs.shape[4]
        self.orbital_names = orbital_names

    @staticmethod
    def from_file(filepath):
        procar = Procar_pmg(filepath)
        spin_keys = list(procar.data.keys())
        proj_koeffs = np.zeros((len(spin_keys),) + procar.data[spin_keys[0]].shape)

        for i, spin_key in enumerate(spin_keys):
            proj_koeffs[i] = procar.data[spin_key]

        return Procar(proj_koeffs, procar.orbitals)

    def get_PDOS(self, outcar: Outcar, atom_numbers, **kwargs):
        self.eigenvalues = outcar.eigenvalues
        self.weights = outcar.weights
        self.nspin = outcar.nspin
        self.nkpts = outcar.nkpts
        self.nbands = outcar.nbands
        self.efermi = outcar.efermi
        self.natoms = outcar.natoms

        if 'zero_at_fermi' in kwargs:
            zero_at_fermi = kwargs['zero_at_fermi']
        else:
            zero_at_fermi = False

        if 'dE' in kwargs:
            dE = kwargs['dE']
        else:
            dE = 0.01

        if 'smearing' in kwargs:
            smearing = kwargs['smearing']
        else:
            smearing = 'Gaussian'

        if smearing == 'Gaussian':
            if 'sigma' in kwargs:
                sigma = kwargs['sigma']
            else:
                sigma = 0.02
            if 'emin' in kwargs:
                E_min = kwargs['emin']
            else:
                E_min = np.min(self.eigenvalues)
            if 'emax' in kwargs:
                E_max = kwargs['emax']
            else:
                E_max = np.max(self.eigenvalues)
        else:
            raise ValueError(f'Only Gaussian smearing is supported but you used {smearing} instead')

        E_arr = np.arange(E_min, E_max, dE)
        ngrid = E_arr.shape[0]

        proj_coeffs_weighted = self.proj_koeffs[:, :, :, atom_numbers, :]

        for spin in range(self.nspin):
            for i, weight_kpt in enumerate(self.weights):
                proj_coeffs_weighted[spin, i] *= weight_kpt

        W_arr = np.moveaxis(proj_coeffs_weighted, [2, 3, 4], [4, 2, 3])
        G_arr = EBS.gaussian_smearing(E_arr, self.eigenvalues, sigma)

        PDOS_arr = np.zeros((self.nspin, len(atom_numbers), self.norbs, ngrid))
        for spin in range(self.nspin):
            for atom in range(len(atom_numbers)):
                PDOS_arr[spin, atom] = np.sum(G_arr[spin, :, None, :, :] * W_arr[spin, :, atom, :, :, None],
                                              axis=(0, 2))

        if self.nspin == 1:
            PDOS_arr *= 2

        if zero_at_fermi:
            return E_arr - self.efermi, PDOS_arr
        else:
            return E_arr, PDOS_arr


class Chgcar:
    """
    Class for reading CHG and CHGCAR files from vasp
    For now, we ignore augmentation occupancies data
    """
    def __init__(self, structure, charge_density, spin_density=None):
        self.structure = structure
        self.charge_density = charge_density
        self.spin_density = spin_density

    @staticmethod
    def from_file(filepath):
        poscar = Poscar.from_file(filepath)
        structure = poscar.structure

        volumetric_data = []
        read_data = False

        with open(filepath, 'r') as file:
            for i in range(8 + structure.natoms):
                file.readline()

            for line in file:
                line_data = line.strip().split()
                if read_data:
                    for value in line_data:
                        if i < length - 1:
                            data[indexes_1[i], indexes_2[i], indexes_3[i]] = float(value)
                            i += 1
                        else:
                            data[indexes_1[i], indexes_2[i], indexes_3[i]] = float(value)
                            read_data = False
                            volumetric_data.append(data)
                else:
                    if len(line_data) == 3:
                        try:
                            shape = np.array(list(map(int, line_data)))
                        except:
                            pass
                        else:
                            read_data = True
                            nx, ny, nz = shape
                            data = np.zeros(shape)
                            length = np.prod(shape)
                            i = 0
                            indexes = np.arange(0, length)
                            indexes_1 = indexes % nx
                            indexes_2 = (indexes // nx) % ny
                            indexes_3 = indexes // (nx * ny)

        if len(volumetric_data) == 1:
            return Chgcar(structure, volumetric_data[0])
        elif len(volumetric_data) == 2:
            return Chgcar(structure, volumetric_data[0], volumetric_data[1])
        else:
            raise ValueError(f'The file contains more than 2 volumetric data, len = {len(volumetric_data)}')

    def convert_to_cube(self, volumetric_data='charge_density'):
        comment = '  Cube file was created using Electrochemistry package\n'
        if volumetric_data == 'charge_density':
            return Cube(data=self.charge_density,
                        structure=self.structure,
                        comment=comment+'  Charge Density\n',
                        origin=np.zeros(3))
        elif volumetric_data == 'spin_density':
            return Cube(data=self.spin_density,
                        structure=self.structure,
                        comment=comment + '  Spin Density\n',
                        origin=np.zeros(3))
        elif volumetric_data == 'spin_major':
            return Cube(data=(self.charge_density + self.spin_density)/2,
                        structure=self.structure,
                        comment=comment+'  Major Spin\n',
                        origin=np.zeros(3))
        elif volumetric_data == 'spin_minor':
            return Cube(data=(self.charge_density - self.spin_density)/2,
                        structure=self.structure,
                        comment=comment+'  Minor Spin\n',
                        origin=np.zeros(3))

    def to_file(self, filepath):
        #TODO write to_file func
        pass


class Xdatcar:
    """Class that reads VASP XDATCAR files"""

    def __init__(self,
                 structure,
                 comment: str = None,
                 trajectory=None):
        """
        Create an Xdatcar instance
        Args:
            structure (Structure class): a base class that contains lattice, coords and species information
            comment (str): a VASP comment
            trajectory (3D np.array): contains coordinates of all atoms along with trajectory. It has the shape
             n_steps x n_atoms x 3
        """
        self.structure = structure
        self.comment = comment
        self.trajectory = trajectory

    def __add__(self, other):
        """
        Concatenates Xdatcar files (theirs trajectory)
        Args:
            other (Xdatcar class): Xdatcar that should be added to the current Xdatcar

        Returns (Xdatcar class):
            New Xdatcar with concatenated trajectory
        """
        assert isinstance(other, Xdatcar), 'Other object must belong to Xdatcar class'
        assert np.array_equal(self.structure.lattice, other.structure.lattice), 'Lattices of two files must be equal'
        assert self.structure.species == other.structure.species, 'Species in two files must be identical'
        assert self.structure.coords_are_cartesian == other.structure.coords_are_cartesian, \
            'Coords must be in the same coordinate system'
        trajectory = np.vstack((self.trajectory, other.trajectory))

        return Xdatcar(self.structure, self.comment + ' + ' + other.comment, trajectory)

    def add(self, other):
        """
        Concatenates Xdatcar files (theirs trajectory)
        Args:
            other (Xdatcar class): Xdatcar that should be added to the current Xdatcar

        Returns (Xdatcar class):
            New Xdatcar with concatenated trajectory
        """
        return self.__add__(other)

    def add_(self, other):
        """
        Concatenates Xdatcar files (theirs trajectory). It's inplace operation, current Xdatcar will be modified
        Args:
            other (Xdatcar class): Xdatcar that should be added to the current Xdatcar
        """
        assert isinstance(other, Xdatcar), 'Other object must belong to Xdatcar class'
        assert np.array_equal(self.structure.lattice, other.structure.lattice), 'Lattices of two files mist be equal'
        assert self.structure.species == other.structure.species, 'Species in two files must be identical'
        assert self.structure.coords_are_cartesian == other.structure.coords_are_cartesian, \
            'Coords must be in the same coordinate system'
        self.trajectory = np.vstack((self.trajectory, other.trajectory))

    @property
    def nsteps(self):
        return len(self.trajectory)

    @staticmethod
    def from_file(filepath):
        """
        Static method to read a XDATCAR file
        Args:
            filepath: path to the XDATCAR file

        Returns:
            Xdatcar class object
        """
        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        comment = data[0].strip()
        scale = float(data[1])
        lattice = np.array([[float(i) for i in line.split()] for line in data[2:5]])
        if scale < 0:
            # In VASP, a negative scale factor is treated as a volume.
            # We need to translate this to a proper lattice vector scaling.
            vol = abs(np.linalg.det(lattice))
            lattice *= (-scale / vol) ** (1 / 3)
        else:
            lattice *= scale

        name_species = data[5].split()
        num_species = [int(i) for i in data[6].split()]
        species = []
        for name, num in zip(name_species, num_species):
            species += [name] * num

        n_atoms = np.sum(num_species)
        n_steps = int((len(data) - 7) / (n_atoms + 1))
        trajectory = np.zeros((n_steps, n_atoms, 3))

        for i in range(n_steps):
            atom_start = 8 + i * (n_atoms + 1)
            atom_stop = 7 + (i + 1) * (n_atoms + 1)
            data_step = [line.split() for line in data[atom_start:atom_stop]]
            for j in range(n_atoms):
                trajectory[i, j] = [float(k) for k in data_step[j]]

        struct = Structure(lattice, species, trajectory[0], coords_are_cartesian=False)

        return Xdatcar(struct, comment, trajectory)

    def to_file(self, filepath):
        file = open(filepath, 'w')
        file.write(f'{self.comment}\n')
        file.write('1\n')
        for vector in self.structure.lattice:
            file.write(f'  {vector[0]}  {vector[1]}  {vector[2]}\n')

        species = np.array(self.structure.species)
        sorted_order = np.argsort(species, kind='stable')
        sorted_trajectory = self.trajectory[:, sorted_order, :]
        unique, counts = np.unique(species, return_counts=True)
        line = '   '
        for u in unique:
            line += u + '  '
        file.write(line + '\n')
        line = '   '
        for c in counts:
            line += str(c) + '  '
        file.write(line + '\n')

        for i in range(self.nsteps):
            file.write(f'Direct configuration=     {i + 1}\n')
            for j in range(self.structure.natoms):
                file.write(f'  {sorted_trajectory[i, j, 0]}  '
                           f'{sorted_trajectory[i, j, 1]}  '
                           f'{sorted_trajectory[i, j, 2]}\n')

        file.close()

    def mod_coords_to_cartesian(self):
        if self.structure.coords_are_cartesian is True:
            return 'Coords are already cartesian'
        else:
            self.trajectory = np.matmul(self.trajectory, self.structure.lattice)
            self.structure.mod_coords_to_cartesian()

    def mod_coords_to_box(self):
        assert self.structure.coords_are_cartesian is False, 'This operation allowed only for NON-cartesian coords'
        self.trajectory %= 1
        self.structure.coords %= 1

import numpy as np
from typing import Union, List, Iterable
from monty.re import regrep
from electrochemistry.core.structure import Structure
from electrochemistry.io.universal import Cube


class Poscar:
    """Class that reads VASP POSCAR files"""
    def __init__(self,
                 structure,
                 comment: str = None,
                 sdynamics_data=None):
        """
        Create an Poscar instance
        Args:
            structure (Structure class): a base class that contains lattice, coords and species information
            comment (str): a VASP comment
            sdynamics_data (list, 2D np.array): data about selective dynamics for each atom. [['T', 'T', 'F'],
            ['F', 'F', 'F'],...]
        """
        self._structure = structure
        self.comment = comment
        self._sdynamics_data = sdynamics_data

    def __repr__(self):
        return f'{self.comment}\n' + repr(self._structure)

    @staticmethod
    def from_file(filepath):
        """
        Static method to read a POSCAR file
        Args:
            filepath: path to the POSCAR file

        Returns:
            Poscar class object
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

    def to_file(self, filepath):
        file = open(filepath, 'w')
        file.write(f'{self.comment}\n')
        file.write('1\n')
        for vector in self._structure.lattice:
            file.write(f'  {vector[0]}  {vector[1]}  {vector[2]}\n')

        species = np.array(self._structure.species)
        sorted_order = np.argsort(species)
        unique, counts = np.unique(species, return_counts=True)
        line = '   '
        for u in unique:
            line += u + '  '
        file.write(line + '\n')
        line = '   '
        for c in counts:
            line += str(c) + '  '
        file.write(line + '\n')

        if self._sdynamics_data is not None:
            file.write('Selective dynamics\n')

        if self._structure.coords_are_cartesian:
            file.write('Cartesian\n')
        else:
            file.write('Direct\n')

        if self._sdynamics_data is None:
            for i in sorted_order:
                atom = self._structure.coords[i]
                file.write(f'  {atom[0]}  {atom[1]}  {atom[2]}\n')
        else:
            for i in sorted_order:
                atom = self._structure.coords[i]
                sd_atom = self._sdynamics_data[i]
                file.write(f'  {atom[0]}  {atom[1]}  {atom[2]}  {sd_atom[0]}  {sd_atom[1]}  {sd_atom[2]}\n')

        file.close()

    def add_atoms(self, coords, species, sdynamics_data=None):
        self._structure.mod_add_atoms(coords, species)
        if sdynamics_data is not None:
            if any(isinstance(el, list) for el in sdynamics_data):
                for sd_atom in sdynamics_data:
                    self._sdynamics_data.append(sd_atom)
            else:
                self._sdynamics_data.append(sdynamics_data)

    def change_atoms(self, ids: Union[int, Iterable],
                     new_coords: Union[Iterable[float], Iterable[Iterable[float]]] = None,
                     new_species: Union[str, List[str]] = None,
                     new_sdynamics_data: Union[Iterable[str], Iterable[Iterable[str]]] = None):
        self._structure.mod_change_atoms(ids, new_coords, new_species)
        if new_sdynamics_data is not None:
            if self._sdynamics_data is None:
                self._sdynamics_data = [['T', 'T', 'T'] for _ in range(self._structure.natoms)]
            if isinstance(ids, Iterable):
                for i, new_sdata in zip(ids, new_sdynamics_data):
                    self._sdynamics_data[i] = new_sdata
            else:
                self._sdynamics_data[ids] = new_sdynamics_data

    def coords_to_cartesian(self):
        if self._structure.coords_are_cartesian is True:
            return 'Coords are already cartesian'
        else:
            self._structure._coords = np.matmul(self._structure.coords, self._structure.lattice)
            self._structure.coords_are_cartesian = True

    def coords_to_direct(self):
        if self._structure.coords_are_cartesian is False:
            return 'Coords are alresdy direct'
        else:
            transform = np.linalg.inv(self._structure.lattice)
            self._structure._coords = np.matmul(self._structure.coords, transform)
            self._structure.coords_are_cartesian = False

    def convert(self, frmt):
        pass


class Outcar:
    """Class that reads VASP OUTCAR files"""

    def __init__(self, nkpts, nbands, natoms, weights, nisteps, spin_restricted, efermi_hist=None, eigenvalues_hist=None,
                 occupations_hist=None, energy_hist=None, energy_ionic_hist=None, forces_hist=None):
        self.nkpts = nkpts
        self.nbands = nbands
        self.natoms = natoms
        self.weights = weights
        self.nisteps = nisteps
        self.spin_restricted = spin_restricted

        self.efermi = efermi_hist[-1]
        self.forces = forces_hist[-1]

        if not spin_restricted:
            self.eigenvalues = eigenvalues_hist[-1][0]
            self.occupations = occupations_hist[-1][0]
        else:
            self.eigenvalues = eigenvalues_hist[-1]
            self.occupations = occupations_hist[-1]

        self.forces_hist = forces_hist
        self.efermi_hist = efermi_hist
        if not spin_restricted:
            self.eigenvalues_hist = eigenvalues_hist[:, 0, :, :]
            self.occupations_hist = occupations_hist[:, 0, :, :]
        else:
            self.eigenvalues_hist = eigenvalues_hist
            self.occupations_hist = occupations_hist
        self.energy_hist = energy_hist
        self.energy_ionic_hist = energy_ionic_hist

    @staticmethod
    def _GaussianSmearing(x, x0, sigma):
        """Simulate the Delta function by a Gaussian shape function"""

        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    @staticmethod
    def from_file(filepath):
        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'nkpts': 'k-points\s+NKPTS\s+=\s+(\d+)',
                    'nbands': 'number of bands\s+NBANDS=\s+(\d+)',
                    'natoms': 'NIONS\s+=\s+(\d+)',
                    'weights': 'Following reciprocal coordinates:',
                    'efermi': 'E-fermi\s:\s+([-.\d]+)',
                    'energy': 'free energy\s+TOTEN\s+=\s+(.\d+\.\d+)\s+eV',
                    'energy_ionic': 'free  energy\s+TOTEN\s+=\s+(.\d+\.\d+)\s+eV',
                    'kpoints': r'k-point\s+(\d+)\s:\s+[-.\d]+\s+[-.\d]+\s+[-.\d]+\n',
                    'forces': '\s+POSITION\s+TOTAL-FORCE',
                    'spin': 'spin component \d+\n'}
        matches = regrep(filepath, patterns)

        nbands = int(matches['nbands'][0][0][0])
        nkpts = int(matches['nkpts'][0][0][0])
        natoms = int(matches['natoms'][0][0][0])
        energy_hist = ([float(i[0][0]) for i in matches['energy']])
        energy_ionic_hist = ([float(i[0][0]) for i in matches['energy_ionic']])

        if matches['spin'] != []:
            spin_restricted = True
            nspin = 2
        else:
            spin_restricted = False
            nspin = 1

        if nkpts == 1:
            weights = [float(data[matches['weights'][0][1] + 2].split()[3])]
        else:
            weights = np.zeros(nkpts)
            for i in range(nkpts):
                weights[i] = float(data[matches['weights'][0][1] + 2 + i].split()[3])
            weights /= np.sum(weights)

        arr = matches['efermi']
        efermi_hist = np.zeros(len(arr))
        for i in range(len(arr)):
            efermi_hist[i] = float(arr[i][0][0])

        nisteps = len(efermi_hist)
        eigenvalues_hist = np.zeros((nisteps, nspin, nkpts, nbands))
        occupations_hist = np.zeros((nisteps, nspin, nkpts, nbands))

        each_kpoint_list = np.array([[int(j[0][0]), int(j[1])] for j in matches['kpoints']])
        for step in range(nisteps):
            for spin in range(nspin):
                for kpoint in range(nkpts):
                    arr = data[each_kpoint_list[nkpts * nspin * step + nkpts * spin + kpoint, 1] + 2:each_kpoint_list[
                                                nkpts * nspin * step + nkpts * spin + kpoint, 1] + 2 + nbands]
                    eigenvalues_hist[step, spin, kpoint] = [float(i.split()[1]) for i in arr]
                    occupations_hist[step, spin, kpoint] = [float(i.split()[2]) for i in arr]

        arr = matches['forces']
        forces_hist = np.zeros((nisteps, natoms, 3))
        for step in range(nisteps):
            for atom in range(natoms):
                line = data[arr[step][1] + atom + 2:arr[step][1] + atom + 3]
                line = line[0].split()
                forces_hist[step, atom] = [float(line[3]), float(line[4]), float(line[5])]

        return Outcar(nkpts, nbands, natoms, weights, nisteps, spin_restricted, efermi_hist, eigenvalues_hist, occupations_hist,
                      energy_hist, energy_ionic_hist, forces_hist)


    def get_band_eigs(self, bands):
        if type(bands) is int:
            return np.array([eig for eig in self.eigenvalues[:, bands]])
        elif isinstance(bands, Iterable):
            return np.array([[eig for eig in self.eigenvalues[:, band]] for band in bands])
        else:
            raise ValueError('Variable bands should be int or iterable')

    def get_band_occ(self, bands):
        if type(bands) is int:
            return [occ for occ in self.occupations[:, bands]]
        elif isinstance(bands, Iterable):
            return np.array([[occ for occ in self.occupations[:, band]] for band in bands])
        else:
            raise ValueError('Variable bands should be int or iterable')

    def get_DOS(self, **kwargs):

        # TODO: Add if smearing == False, add if smearing == "Lorenz", Check *2 electrons/states?
        """Calculate Density of States based on eigenvalues and its weights

        Args:
            dE (float, optional): step of energy array in function's output. Default value is 0.01
            zero_at_fermi (bool, optional): if True Fermi energy will be equal to zero
            emin (float, optional): minimum value in DOS calculation.
            emax (float, optional): maximum value in DOS calculation.
            smearing (str, optional): define whether will be used smearing or not. Possible options: 'Gaussian'
            sigma (float, optional): define the sigma parameter in Gaussian smearing. Default value is 0.02

        Returns:
            E, DOS - Two 1D np.arrays that contain energy and according DOS values
        """
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
            smearing = False

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
            E_arr = np.arange(E_min, E_max, dE)

            if not self.spin_restricted:
                DOS_arr = np.zeros_like(E_arr)
                for energy_kpt, weight in zip(self.eigenvalues, self.weights):
                    for energy in energy_kpt:
                        DOS_arr += 2 * weight * self._GaussianSmearing(E_arr, energy, sigma)
                        # 2 above means occupancy for spin unrestricted calculation
            else:
                DOS_arr = np.zeros((2,) + np.shape(E_arr))
                for spin in range(2):
                    for energy_kpt, weight in zip(self.eigenvalues[spin], self.weights):
                        for energy in energy_kpt:
                            DOS_arr[spin] += weight * self._GaussianSmearing(E_arr, energy, sigma)

            if zero_at_fermi:
                return E_arr - self.efermi, DOS_arr
            else:
                return E_arr, DOS_arr


class Wavecar:
    """Class that reads VASP WAVECAR files"""
    # TODO: add useful functions for Wavecar class: plot charge density, plot real and imag parts etc.

    def __init__(self, kb_array, wavefunctions, ngrid_factor):
        self.kb_array = kb_array
        self.wavefunctions = wavefunctions
        self.ngrid_factor = ngrid_factor

    @staticmethod
    def from_file(filepath, kb_array, ngrid_factor=1.5):
        from electrochemistry.core.vaspwfc_p3 import vaspwfc
        wfc = vaspwfc(filepath)
        wavefunctions = []
        for kb in kb_array:
            kpoint = kb[0]
            band = kb[1]
            wf = wfc.wfc_r(ikpt=kpoint, iband=band, ngrid=wfc._ngrid * ngrid_factor)
            wavefunctions.append(wf)
        return Wavecar(kb_array, wavefunctions, ngrid_factor)


class Procar:
    # TODO create Procar class
    pass

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
        structure = poscar._structure

        volumetric_data = []
        read_data = False

        with open(filepath, 'r') as file:
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
            return Cube(self.structure, comment + '  Charge Density\n', np.array(list(self.charge_density.shape)),
                        np.zeros(self.structure.natoms), self.charge_density)
        elif volumetric_data == 'spin_density':
            return Cube(self.structure, comment + '  Spin Density\n', np.array(list(self.spin_density.shape)),
                        np.zeros(self.structure.natoms), self.spin_density)
        elif volumetric_data == 'spin_major':
            return Cube(self.structure, comment + '  Major Spin\n', np.array(list(self.spin_density.shape)),
                        np.zeros(self.structure.natoms), (self.charge_density + self.spin_density) / 2)
        elif volumetric_data == 'spin_minor':
            return Cube(self.structure, comment + '  Minor Spin\n', np.array(list(self.spin_density.shape)),
                        np.zeros(self.structure.natoms), (self.charge_density - self.spin_density) / 2)

    def to_file(self, filepath):
        #TODO write to_file func
        pass

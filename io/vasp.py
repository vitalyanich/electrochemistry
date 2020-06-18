import numpy as np
from typing import Union, List, Iterable
from monty.re import regrep
from seaborn.distributions import stats
from scipy.integrate import simps
from electrochemistry.core.structure import Structure


class Poscar:
    # TODO Description
    def __init__(self,
                 structure,
                 comment: str = None,
                 sdynamics_data=None):
        self._structure = structure
        self.comment = comment
        self._sdynamics_data = sdynamics_data

    def __repr__(self):
        return f'{self.comment}\n' + repr(self._structure)

    @staticmethod
    def from_file(filepath):
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
        self._structure.add_atoms(coords, species)
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
        self._structure.change_atoms(ids, new_coords, new_species)
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
    # TODO Description
    def __init__(self, nkpts, nbands, weights, nisteps, efermi, eigenvalues, occupations,
                 efermi_hist=None, eigenvalues_hist=None, occupations_hist=None, energy_hist=None):
        self.nkpts = nkpts
        self.nbands = nbands
        self.weights = weights
        self.nisteps = nisteps
        self.efermi = efermi
        self.eigenvalues = eigenvalues
        self.occupations = occupations

        self.efermi_hist = efermi_hist
        self.eigenvalues_hist = eigenvalues_hist
        self.occupations_hist = occupations_hist
        self.energy_hist = energy_hist

    @staticmethod
    def from_file(filepath):
        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'nkpts': 'k-points\s+NKPTS\s+=\s+(\d+)',
                    'nbands': 'number of bands\s+NBANDS=\s+(\d+)',
                    'weights': 'Following reciprocal coordinates:',
                    'efermi': 'E-fermi\s:\s+([-.\d]+)',
                    'energy': 'free energy\s+TOTEN\s+=\s+(.\d+\.\d+)\s+eV',
                    'kpoints': r'k-point\s+(\d+)\s:\s+[-.\d]+\s+[-.\d]+\s+[-.\d]+\n'}
        matches = regrep(filepath, patterns)

        nbands = int(matches['nbands'][0][0][0])
        nkpts = int(matches['nkpts'][0][0][0])
        energy_hist = ([float(i[0][0]) for i in matches['energy']])

        if nkpts == 1:
            weights = data[matches['weights'][0][1] + 2].split()[3]
        else:
            weights = np.zeros(nkpts)
            for i in range(nkpts):
                weights[i] = data[matches['weights'][0][1] + 2 + i].split()[3]
            weights /= np.sum(weights)

        arr = matches['efermi']
        efermi_hist = np.zeros(len(arr))
        for i in range(len(arr)):
            efermi_hist[i] = float(arr[i][0][0])
        efermi = efermi_hist[-1]

        nisteps = len(efermi_hist)
        eigenvalues_hist = np.zeros((nisteps, nkpts, nbands))
        occupations_hist = np.zeros((nisteps, nkpts, nbands))

        each_kpoint_list = np.array([[int(j[0][0]), int(j[1])] for j in matches['kpoints']])
        for step in range(nisteps):
            for kpoint in range(nkpts):
                arr = data[each_kpoint_list[nkpts * step + kpoint, 1] + 2:each_kpoint_list[
                                                                              nkpts * step + kpoint, 1] + 2 + nbands]
                eigenvalues_hist[step, kpoint] = [float(i.split()[1]) for i in arr]
                occupations_hist[step, kpoint] = [float(i.split()[2]) for i in arr]
        eigenvalues = eigenvalues_hist[-1]
        occupations = occupations_hist[-1]

        return Outcar(nkpts, nbands, weights, nisteps, efermi, eigenvalues, occupations,
                      efermi_hist, eigenvalues_hist, occupations_hist, energy_hist)

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

    def get_DOS(self, dE, zero_at_fermi=False, sm_param=None):
        """Calculate Density of States based on eigenvalues and its weights

        Args:
            dE (float): step of energy array in function output
            zero_at_fermi (bool, optional): if True Fermi energy will be equal to zero
            sm_param (dict, optional): parameters for smooth DOS.
                E_min (float, str): minimum value in DOS calculation. If E_min == 'min' left border of energy
                is equal to the minimum eigenvalue
                E_max (float, str): maximum value in DOS calculation. If E_max == 'max' right border of energy
                is equal to the maximum eigenvalue
                bw_method (float): The method used to calculate the estimator bandwidth. This can be 'scott',
                'silverman', a scalar constant or a callable. If a scalar, this will be used directly as `kde.factor`.
                If a callable, it should take a `gaussian_kde` instance as only parameter and return a scalar.
                If None (default), 'scott' is used.
                nelec (int): Number of electrons in the system. DOS integral to efermi should be equal to the nelec

        Returns:
            E, DOS - Two 1D np.arrays that contain energy and according DOS values
        """
        if sm_param is None:
            E_min = np.min(self.eigenvalues.flatten())
            E_max = np.max(self.eigenvalues.flatten())
            E_arr = np.arange(E_min, E_max, dE)
            energies_number = len(E_arr)
            DOS_arr = np.zeros(energies_number)
            for energy_band, weight in zip(self.eigenvalues, self.weights):
                for energy in energy_band:
                    place = int((energy - E_arr[0]) / dE)
                    DOS_arr[place] += weight / dE
            if zero_at_fermi is False:
                return E_arr, DOS_arr
            else:
                return E_arr - self.efermi, DOS_arr
        else:
            weights_flatten = []
            for energy_band, weight in zip(self.eigenvalues, self.weights):
                weights_flatten.append(np.ones(len(energy_band)) * weight)
            weights_flatten = np.array(weights_flatten).flatten()
            a = stats.kde.gaussian_kde(self.eigenvalues.flatten(), bw_method=sm_param['bw_method'],
                                       weights=weights_flatten)
            if sm_param['E_min'] == 'min':
                sm_param['E_min'] = np.min(self.eigenvalues.flatten())
            if sm_param['E_max'] == 'max':
                sm_param['E_max'] = np.max(self.eigenvalues.flatten())
            x = np.arange(sm_param['E_min'], sm_param['E_max'], dE)
            y = a.evaluate(x)

            i = 0
            while self.efermi > x[i]:
                i += 1

            integral = simps(y[:i], x[:i])
            y = (sm_param['nelec'] / integral) * y

            if zero_at_fermi is False:
                return x, y
            else:
                return x - self.efermi, y
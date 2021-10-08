import numpy as np
from typing import Union, List, Sequence, Iterable


class Structure:
    """
    Basic class for structure of unit/super cell.
    Args:
        lattice: 2D array that contains lattice vectors. Each row should corresponds to a lattice vector.
            E.g., [[5, 5, 0], [7, 4, 0], [0, 0, 25]].
        species: List of species on each site. Usually list of elements, e.g., ['Al', 'Al', 'O', 'H'].
        coords: List of lists or np.ndarray (Nx3 dimension) that contains coords of each species.
        coords_are_cartesian: True if coords are cartesian, False if coords are fractional
    """
    def __init__(self,
                 lattice: Union[List, np.ndarray],
                 species: Union[List[str], np.ndarray],
                 coords: Sequence[Sequence[float]],
                 coords_are_cartesian: bool = True):

        if len(species) != len(coords):
            raise StructureError('Number of species and its coords must be the same')

        self.species = species

        if isinstance(lattice, np.ndarray):
            self.lattice = lattice
        else:
            self.lattice = np.array(lattice)

        if isinstance(coords, np.ndarray):
            self.coords = coords
        else:
            self.coords = np.array(coords)

        self.coords_are_cartesian = coords_are_cartesian

    def __repr__(self):
        lines = ['\nLattice:']

        width = len(str(int(np.max(self.lattice)))) + 6
        for axis in self.lattice:
            lines.append('  '.join([f'{axis_coord:{width}.5f}' for axis_coord in axis]))

        width = len(str(int(np.max(self.coords)))) + 6

        lines.append('\nSpecies:')

        unique, counts = np.unique(self.species, return_counts=True)

        if len(self.species) < 10:
            lines.append(' '.join([s for s in self.species]))
            for u, c in zip(unique, counts):
                lines.append(f'{u}:\t{c}')
            lines.append(f'\nCoords are cartesian: {self.coords_are_cartesian}')
            lines.append('\nCoords:')
            for coord in self.coords:
                lines.append('  '.join([f'{c:{width}.5f}' for c in coord]))
        else:
            part_1 = ' '.join([s for s in self.species[:5]])
            part_2 = ' '.join([s for s in self.species[-5:]])
            lines.append(part_1 + ' ... ' + part_2)
            for u, c in zip(unique, counts):
                lines.append(f'{u}:\t{c}')
            lines.append(f'\nCoords are cartesian: {self.coords_are_cartesian}')
            lines.append('\nCoords:')
            for coord in self.coords[:5]:
                lines.append('  '.join([f'{c:{width}.5f}' for c in coord]))
            lines.append('...')
            for coord in self.coords[-5:]:
                lines.append('  '.join([f'{c:{width}.5f}' for c in coord]))

        return '\n'.join(lines)

    @property
    def natoms(self) -> int:
        return len(self.species)

    def mod_add_atoms(self, coords, species) -> None:
        """
        Adds atoms in the Structure
        Args:
            coords: List or np.ndarray (Nx3 dimension) that contains coords of each species
            species: List of species on each site. Usually list of elements, e.g., ['Al', 'Al', 'O', 'H']
        """
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
        self.coords = np.vstack((self.coords, coords))
        if isinstance(species, str):
            self.species += [species]
        else:
            self.species += species

    def mod_delete_atoms(self, ids) -> None:
        """
        Deletes selected atoms by ids
        Args:
            ids: sequence of atoms ids
        """
        self.coords = np.delete(self.coords, ids, axis=0)
        self.species = np.delete(self.species, ids)

    def mod_change_atoms(self, ids, coords, species) -> None:
        """
        Change selected atom by id
        Args:
            ids: List or int. First id is zero.
            coords: None or np.array with new coords, e.g. np.array([1, 2, 4.6])
            species: None or str or List[str]. New types of changed atoms

        Returns:

        """
        if coords is not None:
            self.coords[ids] = coords
        if species is not None:
            if isinstance(ids, Iterable):
                for i, sp in zip(ids, species):
                    self.species[i] = sp
            else:
                self.species[ids] = species

    def mod_coords_to_cartesian(self) -> Union[None, str]:
        """
        Converts species coordinates to Cartesian coordination system.
        """
        if self.coords_are_cartesian is True:
            return 'Coords are already cartesian'
        else:
            self.coords = np.matmul(self.coords, self.lattice)
            self.coords_are_cartesian = True

    def mod_coords_to_direct(self) -> Union[None, str]:
        """
        Converts species coordinates to Direct coordination system.
        """
        if self.coords_are_cartesian is False:
            return 'Coords are already direct'
        else:
            transform = np.linalg.inv(self.lattice)
            self.coords = np.matmul(self.coords, transform)
            self.coords_are_cartesian = False

    def mod_add_vector(self, vector, cartesian=True) -> None:
        """
        Adds a vector to all atoms.
        Args:
            vector (np.ndarray): a vector which will be added to coordinates of all species
            cartesian (bool): determine in which coordination system the vector defined. Cartesian=True means that
            the vector is defined in Cartesian coordination system. Cartesian=False means that the vector is defined
            in Direct coordination system.
        """
        self.mod_coords_to_direct()
        if cartesian:
            transform = np.linalg.inv(self.lattice)
            vector = np.matmul(vector, transform)
        self.coords += vector
        self.coords %= np.array([1, 1, 1])

    def get_vector(self, id_1, id_2, unit=True) -> np.ndarray:
        """
        Returns a vector (unit vector by default) which starts in the atom with id_1 and points to the atom with id_2
        Args:
            id_1: id of the first atom (vector origin)
            id_2: id of the second atom
            unit: Defines whether the vector will be normed or not. Unit=True means that the vector norm will be equal 1
        Returns (np.ndarray): vector from atom with id_1 to atom with id_2
        """
        vector = self.coords[id_2] - self.coords[id_1]
        if unit:
            return vector / np.linalg.norm(vector)
        else:
            return vector

    def get_distance_matrix(self) -> np.ndarray:
        """
        Returns distance matrix R, where R[i,j] is the vector from atom i to atom j.

        Returns:
            np.ndarray (NxNx3 dimensions) which is a distance matrix in Cartesian coordination system
        """
        '''
        Old version (only for parallelepiped cells)
        if self.coords_are_cartesian is False:
            assert StructureError('Now only cartesian coords are supported')
        r1 = np.broadcast_to(self.coords.reshape((self.natoms, 1, 3)), (self.natoms, self.natoms, 3))
        r2 = np.broadcast_to(self.coords.reshape((1, self.natoms, 3)), (self.natoms, self.natoms, 3))
        R = r2 - r1
        L = np.broadcast_to(np.linalg.norm(self._lattice, axis=1).reshape((1, 1, 3)), (self.natoms, self.natoms, 3))
        R = (R + L / 2) % L - L / 2.
        assert np.all(R >= -L / 2.) and np.all(R <= L / 2.)
        return R
        '''

        self.mod_coords_to_direct()
        r1 = np.broadcast_to(self.coords.reshape((self.natoms, 1, 3)), (self.natoms, self.natoms, 3))
        r2 = np.broadcast_to(self.coords.reshape((1, self.natoms, 3)), (self.natoms, self.natoms, 3))
        R = r2 - r1
        R = (R + 0.5) % 1 - 0.5
        assert np.all(R >= - 0.5) and np.all(R <= 0.5)
        return np.matmul(R, self.lattice)

    def get_distance_matrix_scalar(self) -> np.ndarray:
        """
        Returns distance matrix R, where R[i, j] is the Euclidean norm of a vector from atom i to atom j.

        Returns:
            np.ndarray (NxN dimensions) which is a distance matrix containing scalars
        """
        R = self.get_distance_matrix()
        return np.sqrt(np.sum(R * R, axis=2))

    def get_filtered_ids(self, **kwargs):
        """
        Returns np.ndarray that contains atom ids according to filter rules
        Args:
            species (str): Define which atom type will be selected. E.g. 'C H N' means select all C, H, and N atoms.
            '!C' means select all atoms except C.
        Returns:
            np.ndarray contains ids of atoms according to selecting rules
        """
        filter_mask = np.array([True for _ in range(self.natoms)], dtype=np.bool_)

        if 'species' in kwargs:
            species = kwargs['species'].split()
            species_select = []
            species_not_select = []
            for specie in species:
                if '!' in specie:
                    species_not_select.append(specie.replace('!', ''))
                else:
                    species_select.append(specie)
            if len(species_select):
                fm_local = np.array([False for _ in range(self.natoms)], dtype=np.bool_)
                for specie in species_select:
                    fm_local += np.array([True if atom_name == specie else False for atom_name in self.species])
                filter_mask *= fm_local
            if len(species_not_select):
                fm_local = np.array([True for _ in range(self.natoms)], dtype=np.bool_)
                for specie in species_not_select:
                    fm_local *= np.array([False if atom_name == specie else True for atom_name in self.species])
                filter_mask *= fm_local

        if 'x' in kwargs:
            left, right = kwargs['x']
            fm_local = np.array([False for _ in range(self.natoms)], dtype=np.bool_)
            for i, atom_coord in enumerate(self.coords):
                if left < atom_coord[0] < right:
                    fm_local[i] = True
            filter_mask *= fm_local

        if 'y' in kwargs:
            left, right = kwargs['y']
            fm_local = np.array([False for _ in range(self.natoms)], dtype=np.bool_)
            for i, atom_coord in enumerate(self.coords):
                if left < atom_coord[1] < right:
                    fm_local[i] = True
            filter_mask *= fm_local

        if 'z' in kwargs:
            left, right = kwargs['z']
            fm_local = np.array([False for _ in range(self.natoms)], dtype=np.bool_)
            for i, atom_coord in enumerate(self.coords):
                if left < atom_coord[2] < right:
                    fm_local[i] = True
            filter_mask *= fm_local

        return np.array(range(self.natoms))[filter_mask]


class StructureError(Exception):
    """
    Exception class for Structure.
    Raised when the structure has problems, e.g., atoms that are too close.
    """
    pass

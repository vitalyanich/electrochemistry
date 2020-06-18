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
        coords_are_cartesian: True if coords is cartesian, False if coords is fractional
    """
    def __init__(self,
                 lattice: Union[List, np.ndarray],
                 species: List[str],
                 coords: Sequence[Sequence[float]],
                 coords_are_cartesian: bool = True):

        if len(species) != len(coords):
            raise StructureError('Number of species and its coords must be the same')

        self._species = species

        if isinstance(lattice, np.ndarray):
            self._lattice = lattice
        else:
            self._lattice = np.array(lattice)

        if isinstance(coords, np.ndarray):
            self._coords = coords
        else:
            self._coords = np.array(coords)

        self.coords_are_cartesian = coords_are_cartesian

    def __repr__(self):
        lines = ['\nLattice:']

        width = len(str(int(np.max(self._lattice)))) + 6
        for axis in self._lattice:
            lines.append('  '.join([f'{axis_coord:{width}.5f}' for axis_coord in axis]))

        width = len(str(int(np.max(self._coords)))) + 6

        lines.append('\nSpecies:')

        unique, counts = np.unique(self._species, return_counts=True)

        if len(self._species) < 10:
            lines.append(' '.join([s for s in self._species]))
            for u, c in zip(unique, counts):
                lines.append(f'{u}: {c}')
            lines.append('\nCoords:')
            for coord in self._coords:
                lines.append('  '.join([f'{c:{width}.5f}' for c in coord]))
        else:
            part_1 = ' '.join([s for s in self._species[:5]])
            part_2 = ' '.join([s for s in self._species[-5:]])
            lines.append(part_1 + ' ... ' + part_2)
            for u, c in zip(unique, counts):
                lines.append(f'{u}: {c}')
            lines.append('\nCoords:')
            for coord in self._coords[:5]:
                lines.append('  '.join([f'{c:{width}.5f}' for c in coord]))
            lines.append('...')
            for coord in self._coords[-5:]:
                lines.append('  '.join([f'{c:{width}.5f}' for c in coord]))

        return '\n'.join(lines)

    @property
    def lattice(self):
        return self._lattice

    @property
    def coords(self):
        return self._coords

    @property
    def species(self):
        return self._species

    @property
    def natoms(self):
        return len(self._species)

    def add_atoms(self, coords, species) -> None:
        """
        Add atoms in the Structure
        Args:
            coords: List or np.ndarray (Nx3 dimension) that contains coords of each species
            species: List of species on each site. Usually list of elements, e.g., ['Al', 'Al', 'O', 'H']
        """
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
        self._coords = np.vstack((self._coords, coords))
        if isinstance(species, str):
            self._species += [species]
        else:
            self._species += species

    def change_atoms(self, ids, coords, species):
        """
        Change selected atom by id .
        Args:
            ids: List or int. first id is zero.
            coords: None os np.array with new coords, e.g. np.array([1, 2, 4.6])
            species: None of str or List[str]. New types of changed atoms

        Returns:

        """
        if coords is not None:
            self._coords[ids] = coords
        if species is not None:
            if isinstance(ids, Iterable):
                for i, sp in zip(ids, species):
                    self._species[i] = sp
            else:
                self._species[ids] = species

    def coords_to_cartesian(self):
        if self.coords_are_cartesian is True:
            return 'Coords are already cartesian'
        else:
            self._coords = np.matmul(self.coords, self.lattice)
            self.coords_are_cartesian = True

    def coords_to_direct(self):
        if self.coords_are_cartesian is False:
            return 'Coords are alresdy direct'
        else:
            transform = np.linalg.inv(self.lattice)
            self._coords = np.matmul(self.coords, transform)
            self.coords_are_cartesian = False

    def get_vector(self, id_1, id_2):
        vector = self._coords[id_1] - self._coords[id_2]
        return vector / np.linalg.norm(vector)

    def rotate(self, vector, angle):
        pass


class StructureError(Exception):
    """
    Exception class for Structure.
    Raised when the structure has problems, e.g., atoms that are too close.
    """
    pass

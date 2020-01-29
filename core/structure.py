import numpy as np
from typing import Union, List, Iterable


class Structure:
    """
    Basic class for structure of unit/super cell.
    Args:
        lattice: 2D array that contains lattice vectors. Each row should corresponds to a lattice vector.
            E.g., [[5, 5, 0], [7, 4, 0], [0, 0, 25]].
        species: List of species on each site. Usually list of elements, e.g., ['Al', 'Al', 'O', 'H'].
        coords: List or np.ndarray (Nx3 dimension) that contains coords of each species.
        coords_are_cartesian: True if coords is cartesian, False if coords is fractional
    """
    def __init__(self,
                 lattice: Union[List, np.ndarray],
                 species: List[str],
                 coords: Iterable[Iterable[float]],
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
        if coords is not None:
            self._coords[ids] = coords
        if species is not None:
            if isinstance(ids, Iterable):
                for i, sp in zip(ids, species):
                    self._species[i] = sp
            else:
                self._species[ids] = species


class StructureError(Exception):
    """
    Exception class for Structure.
    Raised when the structure has problems, e.g., atoms that are too close.
    """
    pass
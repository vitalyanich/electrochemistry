import numpy as np
from typing import Union, List, Iterable
from core.structure import Structure


class Poscar:
    # TODO Description
    def __init__(self,
                 structure,
                 comment: str = None,
                 sdynamics_data=None):
        self._structure = structure
        self.comment = comment
        self._sdynamics_data = sdynamics_data

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
        if data[7] in 'sS':
            sdynamics_is_used = True
            start_atoms = 9

        coords_are_cartesian = False
        if sdynamics_is_used:
            if data[8] in 'cCkK':
                coords_are_cartesian = True
        else:
            if data[7] in 'cCkK':
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
                self._sdynamics_data = [['T', 'T', 'T'] for i in range(self._structure.natoms)]
            if isinstance(ids, Iterable):
                for i, new_sdata in zip(ids, new_sdynamics_data):
                    self._sdynamics_data[i] = new_sdata
            else:
                self._sdynamics_data[ids] = new_sdynamics_data


    def convert(self, frmt):
        pass

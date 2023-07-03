import numpy as np
from echem.core.constants import ElemNum2Name, ElemName2Num, Bohr2Angstrom, Angstrom2Bohr
from echem.core.structure import Structure
import warnings


class Cube:
    def __init__(self,
                 data: np.ndarray,
                 structure: Structure,
                 origin: np.ndarray,
                 units_data: str = 'Bohr',
                 comment: str = None,
                 charges=None,
                 dset_ids=None):
        self.volumetric_data = data
        self.structure = structure
        self.origin = origin
        self.units_data = units_data

        if comment is None:
            self.comment = 'Comment is not defined\nGood luck!\n'
        else:
            self.comment = comment

        if charges is None:
            self.charges = np.zeros(structure.natoms)
        else:
            self.charges = charges

        self.dset_ids = dset_ids

    def __repr__(self):
        shape = self.volumetric_data.shape
        return f'{self.comment}\n' + f'NX: {shape[0]}\nNY: {shape[1]}\nNZ: {shape[2]}\n' + \
               f'Origin:\n{self.origin[0]:.5f}  {self.origin[1]:.5f}  {self.origin[2]:.5f}\n' + \
               repr(self.structure)

    def __add__(self, other):
        assert isinstance(other, Cube), 'Other object must belong to Cube class'
        assert np.array_equal(self.origin, other.origin), 'Two Cube instances must have the same origin'
        assert self.volumetric_data.shape == other.volumetric_data.shape, 'Two Cube instances must have ' \
                                                                          'the same shape of volumetric_data'
        if self.structure != other.structure:
            warnings.warn('Two Cube instances have different structures. '
                          'The structure will be taken from the 1st (self) instance. '
                          'Hope you know, what you are doing')

        return Cube(self.volumetric_data + other.volumetric_data, self.structure, self.origin)

    def __sub__(self, other):
        assert isinstance(other, Cube), 'Other object must belong to Cube class'
        assert np.array_equal(self.origin, other.origin), 'Two Cube instances must have the same origin'
        assert self.volumetric_data.shape == other.volumetric_data.shape, 'Two Cube instances must have ' \
                                                                          'the same shape of volumetric_data'
        if self.structure != other.structure:
            warnings.warn('\nTwo Cube instances have different structures. '
                          'The structure will be taken from the 1st (self) instance. '
                          'Hope you know, what you are doing')

        return Cube(self.volumetric_data - other.volumetric_data, self.structure, self.origin)

    def __neg__(self):
        return Cube(-self.volumetric_data, self.structure, self.origin)

    def __mul__(self, other):
        assert isinstance(other, Cube), 'Other object must belong to Cube class'
        assert np.array_equal(self.origin, other.origin), 'Two Cube instances must have the same origin'
        assert self.volumetric_data.shape == other.volumetric_data.shape, 'Two Cube instances must have ' \
                                                                          'the same shape of volumetric_data'
        if self.structure != other.structure:
            warnings.warn('\nTwo Cube instances have different structures. '
                          'The structure will be taken from the 1st (self) instance. '
                          'Hope you know, what you are doing')

        return Cube(self.volumetric_data * other.volumetric_data, self.structure, self.origin)

    @staticmethod
    def from_file(filepath):
        with open(filepath, 'rt') as file:
            comment_1 = file.readline()
            comment_2 = file.readline()
            comment = comment_1 + comment_2

            line = file.readline().split()
            natoms = int(line[0])

            if natoms < 0:
                dset_ids_flag = True
                natoms = abs(natoms)
            else:
                dset_ids_flag = False

            origin = np.array([float(line[1]), float(line[2]), float(line[3])])

            if len(line) == 4:
                n_data = 1
            elif len(line) == 5:
                n_data = int(line[4])

            line = file.readline().split()
            NX = int(line[0])
            xaxis = np.array([float(line[1]), float(line[2]), float(line[3])])
            line = file.readline().split()
            NY = int(line[0])
            yaxis = np.array([float(line[1]), float(line[2]), float(line[3])])
            line = file.readline().split()
            NZ = int(line[0])
            zaxis = np.array([float(line[1]), float(line[2]), float(line[3])])

            if NX > 0 and NY > 0 and NZ > 0:
                units = 'Bohr'
            elif NX < 0 and NY < 0 and NZ < 0:
                units = 'Angstrom'
            else:
                raise ValueError('The sign of the number of all voxels should be > 0 or < 0')

            if units == 'Angstrom':
                NX, NY, NZ = -NX, -NY, -NZ

            lattice = np.array([xaxis * NX, yaxis * NY, zaxis * NZ])

            species = []
            charges = np.zeros(natoms)
            coords = np.zeros((natoms, 3))

            for atom in range(natoms):
                line = file.readline().split()
                species += [ElemNum2Name[int(line[0])]]
                charges[atom] = float(line[1])
                coords[atom, :] = line[2:]

            if units == 'Bohr':
                lattice = Bohr2Angstrom * lattice
                coords = Bohr2Angstrom * coords
                origin = Bohr2Angstrom * origin

            structure = Structure(lattice, species, coords, coords_are_cartesian=True)

            dset_ids = None
            dset_ids_processed = -1
            if dset_ids_flag is True:
                dset_ids = []
                line = file.readline().split()
                n_data = int(line[0])
                if n_data < 1:
                    raise ValueError(f'Bad value of n_data: {n_data}')
                dset_ids_processed += len(line)
                dset_ids += [int(i) for i in line[1:]]
                while dset_ids_processed < n_data:
                    line = file.readline().split()
                    dset_ids_processed += len(line)
                    dset_ids += [int(i) for i in line]
                dset_ids = np.array(dset_ids)

            if n_data != 1:
                raise NotImplemented(f'The processing of cube files with more than 1 data values is not implemented.'
                                     f' n_data = {n_data}')

            data = np.zeros((NX, NY, NZ))
            indexes = np.arange(0, NX * NY * NZ)
            indexes_1 = indexes // (NY * NZ)
            indexes_2 = (indexes // NZ) % NY
            indexes_3 = indexes % NZ

            i = 0
            for line in file:
                for value in line.split():
                    data[indexes_1[i], indexes_2[i], indexes_3[i]] = float(value)
                    i += 1

            return Cube(data, structure, origin, units, comment, charges, dset_ids)

    #def reduce(self, factor):
    #    from skimage.measure import block_reduce
    #    try:
    #        volumetric_data_reduced = block_reduce(self.volumetric_data, block_size=(factor, factor, factor), func=np.mean)
    #        Ns_reduced = np.shape(volumetric_data_reduced)
    #    except:
    #        raise ValueError('Try another factor value')
    #    return Cube(volumetric_data_reduced, self.structure, self.comment, Ns_reduced, self.charges)

    def to_file(self, filepath, units='Bohr'):
        if not self.structure.coords_are_cartesian:
            self.structure.mod_coords_to_cartesian()

        Ns = np.array(self.volumetric_data.shape)
        width_Ni = len(str(np.max(Ns)))
        if units == 'Angstrom':
            Ns = - Ns
            width_Ni += 1
            width_lattice = len(str(int(np.max(self.structure.lattice)))) + 7
            width_coord = len(str(int(np.max(self.structure.coords)))) + 7
        elif units == 'Bohr':
            lattice = self.get_voxel() * Angstrom2Bohr
            coords = self.structure.coords * Angstrom2Bohr
            origin = self.origin * Angstrom2Bohr
            width_lattice = len(str(int(np.max(lattice)))) + 7
            width_coord = len(str(int(np.max(coords)))) + 7
        else:
            raise ValueError(f'Irregular units flag: {units}. Units must be \'Bohr\' or \'Angstrom\'')

        if np.sum(self.structure.lattice < 0):
            width_lattice += 1
        if np.sum(self.structure.coords < 0):
            width_coord += 1
        width = np.max([width_lattice, width_coord])

        if self.dset_ids is not None:
            natoms = - self.structure.natoms
        else:
            natoms = self.structure.natoms
        width_natoms = len(str(natoms))
        width_1_column = max(width_Ni, width_natoms)

        with open(filepath, 'w') as file:
            file.write(self.comment)

            if units == 'Angstrom':
                file.write(f'  {natoms:{width_1_column}}   {self.origin[0]:{width}.6f} '
                           f'  {self.origin[1]:{width}.6f}   {self.origin[2]:{width}.6f}\n')
                for N_i, lattice_vector in zip(Ns, self.get_voxel()):
                    file.write(f'  {N_i:{width_1_column}}   {lattice_vector[0]:{width}.6f} '
                               f'  {lattice_vector[1]:{width}.6f}   {lattice_vector[2]:{width}.6f}\n')
                for atom_name, charge, coord in zip(self.structure.species, self.charges, self.structure.coords):
                    file.write(
                        f'  {ElemName2Num[atom_name]:{width_1_column}}   {charge:{width}.6f} '
                        f'  {coord[0]:{width}.6f}   {coord[1]:{width}.6f}   {coord[2]:{width}.6f}\n')

            elif units == 'Bohr':
                file.write(f'  {natoms:{width_1_column}}   {origin[0]:{width}.6f} '
                           f'  {origin[1]:{width}.6f}   {origin[2]:{width}.6f}\n')
                for N_i, lattice_vector in zip(Ns, lattice):
                    file.write(f'  {N_i:{width_1_column}}   {lattice_vector[0]:{width}.6f} '
                               f'  {lattice_vector[1]:{width}.6f}   {lattice_vector[2]:{width}.6f}\n')
                for atom_name, charge, coord in zip(self.structure.species, self.charges, coords):
                    file.write(
                        f'  {ElemName2Num[atom_name]:{width_1_column}}   {charge:{width}.6f} '
                        f'  {coord[0]:{width}.6f}   {coord[1]:{width}.6f}   {coord[2]:{width}.6f}\n')

            else:
                raise ValueError(f'Irregular units flag: {units}. Units must be \'Bohr\' or \'Angstrom\'')

            if self.dset_ids is not None:
                m = len(self.dset_ids)
                file.write(f'  {m:{width_1_column}}' + '   ')
                for dset_id in self.dset_ids:
                    file.write(str(dset_id) + '   ')
                file.write('\n')

            for i in range(abs(Ns[0])):
                for j in range(abs(Ns[1])):
                    for k in range(abs(Ns[2])):
                        file.write(str('  %.5E' % self.volumetric_data[i][j][k]))
                        if k % 6 == 5:
                            file.write('\n')
                    file.write('\n')

    def mod_to_zero_origin(self):
        self.structure.coords -= self.origin
        self.origin = np.zeros(3)

    def get_average_along_axis(self, axis):
        """
        Gets average value along axis
        Args:
            axis (int):
                if 0 than average along x wil be calculated
                if 1 along y
                if 2 along z

        Returns:
            np.array of average value along selected axis
        """
        if axis == 2:
            return np.mean(self.volumetric_data, (0, 1))
        elif axis == 1:
            return np.mean(self.volumetric_data, (0, 2))
        elif axis == 0:
            return np.mean(self.volumetric_data, (1, 2))
        else:
            raise ValueError('axis can be only 0, 1 or 2')

    def get_average_along_axis_max(self, axis: int, scale=None):
        """Calculate the vacuum level (the maximum planar average value along selected axis)

        Args:
            axis (int): The axis number along which the planar average is calculated. The first axis is 0
            scale (float): The value that is multiplying by the result. It's used for converting between
                different units

        Returns:
            (float): The vacuum level multiplied by scale factor

        """
        avr = self.get_average_along_axis(axis)
        if scale is None:
            return np.max(avr)
        else:
            return scale * np.max(avr)

    def get_voxel(self, units='Angstrom'):
        NX, NY, NZ = self.volumetric_data.shape
        voxel = self.structure.lattice.copy()
        voxel[0] /= NX
        voxel[1] /= NY
        voxel[2] /= NZ
        if units == 'Angstrom':
            return voxel
        elif units == 'Bohr':
            return voxel * Angstrom2Bohr
        else:
            raise ValueError('units can be \'Angstrom\' or \'Bohr\'')

    def get_integrated_number(self):
        if self.units_data == 'Bohr':
            voxel_volume = np.linalg.det(self.get_voxel(units='Bohr'))
            return voxel_volume * np.sum(self.volumetric_data)
        else:
            raise NotImplemented()

    def assign_top_n_data_to_atoms(self, n_top, r):
        """Assign top n abs of volumetric data to atoms. Might be used to assign electron density to atoms.

        Args:
            n_top (int): Number of voxels that will be analysed
            r (float): Radius. A voxel is considered belonging to atom is the distance between the voxel center and ]
            atom is less than r.

        Returns:
            (np.ndarray): Array of boolean values. I-th raw represents i-th atom, j-th column represents j-th voxel
        """
        sorted_indices = np.array(np.unravel_index(np.argsort(-np.abs(self.volumetric_data), axis=None),
                                                   self.volumetric_data.shape)).T
        translation_vector = np.sum(self.structure.lattice, axis=0)
        voxels_centres = sorted_indices[:n_top, :] * translation_vector + translation_vector / 2 + self.origin

        atom_indices = list(range(self.structure.natoms))

        if self.structure.natoms == 1:
            return np.linalg.norm(voxels_centres - self.structure.coords[0], axis=-1) < r
        else:
            return np.linalg.norm(np.broadcast_to(voxels_centres, (self.structure.natoms,) + voxels_centres.shape) -
                                  np.expand_dims(self.structure.coords[atom_indices], axis=1), axis=-1) < r


class Xyz:
    def __init__(self, structure, comment):
        self.structure = structure
        self.comment = comment

    @staticmethod
    def from_file(filepath):
        with open(filepath, 'rt') as file:
            natoms = int(file.readline().strip())
            comment = file.readline()

            coords = np.zeros((natoms, 3))
            species = []
            for i in range(natoms):
                line = file.readline().split()
                species.append(line[0])
                coords[i] = [float(j) for j in line[1:]]

            struct = Structure(np.zeros((3, 3)), species, coords, coords_are_cartesian=True)

        return Xyz(struct, comment)


class XyzTrajectory:
    def __init__(self, first_xyz, trajectory):
        self.first_xyz = first_xyz
        self.trajectory = trajectory

    @staticmethod
    def from_file(filepath):
        first_xyz = Xyz.from_file(filepath)

        trajectory = []
        with open(filepath, 'rt') as file:
            while True:
                try:
                    natoms = int(file.readline().strip())
                except:
                    break
                file.readline()

                coords = np.zeros((natoms, 3))
                for i in range(natoms):
                    line = file.readline().split()
                    coords[i] = [float(j) for j in line[1:]]
                trajectory.append(coords)

        return XyzTrajectory(first_xyz, np.array(trajectory))

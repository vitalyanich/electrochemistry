import numpy as np
from electrochemistry.core.constants import ElemNum2Name, Bohr2Angstom
from electrochemistry.core.structure import Structure


class Cube:
    def __init__(self, structure, comment, Ns_arr, charges, data):
        self.structure = structure
        self.comment = comment
        self.Ns = Ns_arr
        self.charges = charges
        self.volumetric_data = data

    def __repr__(self):
        return f'{self.comment}\n' + f'NX: {self.Ns[0]}\nNY: {self.Ns[1]}\nNZ: {self.Ns[2]}' + repr(self.structure)

    @staticmethod
    def from_file(filepath):
        with open(filepath, 'rt') as file:
            comment_1 = file.readline()
            comment_2 = file.readline()
            comment = comment_1 + comment_2

            line = file.readline().split()
            natoms = int(line[0])

            origin = np.array([float(line[1]), float(line[2]), float(line[3])])
            if not all(i == 0 for i in origin):
                raise ValueError('Cube file with not zero origin can\'t be processed')
            line = file.readline().split()
            NX = int(line[0])
            xaxis = NX * np.array([float(line[1]), float(line[2]), float(line[3])])
            line = file.readline().split()
            NY = int(line[0])
            yaxis = NY * np.array([float(line[1]), float(line[2]), float(line[3])])
            line = file.readline().split()
            NZ = int(line[0])
            zaxis = NZ * np.array([float(line[1]), float(line[2]), float(line[3])])
            lattice = np.array([xaxis, yaxis, zaxis])

            if NX > 0 and NY > 0 and NZ > 0:
                units = 'Bohr'
            elif NX < 0 and NY < 0 and NZ < 0:
                units = 'Angstrom'
            else:
                raise ValueError('The sign of the number of all voxels should be > 0 or < 0')

            species = np.zeros(natoms, dtype='<U1')
            charges = np.zeros(natoms)
            coords = np.zeros((natoms, 3))

            for atom in range(natoms):
                line = file.readline().split()
                species[atom] = ElemNum2Name[int(line[0])]
                charges[atom] = float(line[1])
                coords[atom, :] = line[2:]

            if units == 'Bohr':
                lattice = Bohr2Angstom * lattice
                coords = Bohr2Angstom * coords

            structure = Structure(lattice, species, coords, coords_are_cartesian=True)

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

            #data = np.loadtxt(file).reshape((NX, NY, NZ))

            #i = 0
            #for line in file:
            #    for value in line.split():
            #        data[int(i / (NY * NZ)), int((i / NZ) % NY), int(i % NZ)] = float(value)
            #        i += 1

            return Cube(structure, comment, np.array([NX, NY, NZ]), charges, data)

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

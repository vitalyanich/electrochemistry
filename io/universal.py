import numpy as np
import collections


class Cube:
    def __init__(self):
        self.filepath = None
        self.natoms = None
        self.cell = None
        self.structure = None
        self.data = None

    def __repr__(self):
        data = {'filepath': self.filepath, 'natoms': self.natoms, 'cell': self.cell, 'structure': self.structure}
        return data.__repr__()

    def from_file(self, filepath):
        self.filepath = filepath
        with open(self.filepath, 'rt') as file:
            next(file)
            next(file)

            line = file.readline().split()
            self.natoms = int(line[0])
            origin = np.array([line[1], line[2], line[3]])
            line = file.readline().split()
            NX = int(line[0])
            xaxis = np.array([line[1], line[2], line[3]])
            line = file.readline().split()
            NY = int(line[0])
            yaxis = np.array([line[1], line[2], line[3]])
            line = file.readline().split()
            NZ = int(line[0])
            zaxis = np.array([line[1], line[2], line[3]])

            Cell = collections.namedtuple('Cell', 'origin, NX, NY, NZ, xaxis, yaxis, zaxis')
            self.cell = Cell(origin, NX, NY, NZ, xaxis, yaxis, zaxis)

            atom_numbers = np.zeros((self.natoms, ), dtype=int)
            charges = np.zeros((self.natoms, ))
            coords = np.zeros((self.natoms, 3))

            for atom in range(self.natoms):
                line = file.readline().split()
                atom_numbers[atom] = line[0]
                charges[atom] = line[1]
                coords[atom, :] = line[2:]

            Structure = collections.namedtuple('Structure', 'atom_number charge coords')
            self.structure = Structure(atom_numbers, charges, coords)

            self.data = np.zeros((self.cell.NX, self.cell.NY, self.cell.NZ))
            i = 0
            for line in file:
                for value in line.split():
                    self.data[int(i / (self.cell.NY * self.cell.NZ)),
                              int((i / self.cell.NZ) % self.cell.NY),
                              int(i % self.cell.NZ)] = float(value)
                    i += 1

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
            return np.mean(self.data, (0, 1))
        elif axis == 1:
            return np.mean(self.data, (0, 2))
        elif axis == 0:
            return np.mean(self.data, (1, 2))
        else:
            raise ValueError('axis can be only 0, 1 or 2')

    def get_vacuum_lvl(self, axis: int, scale=None):
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

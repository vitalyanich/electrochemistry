from matplotlib import colors
import numpy as np
from monty.re import regrep
from echem.core.structure import Structure


class AtomicNetCharges:
    """Class that operates with DDEC output file DDEC6_even_tempered_net_atomic_charges.xyz"""
    def __init__(self, structure: Structure, net_charges, dipoles_xyz=None,
                 dipoles_mag=None, Qs=None, quadrupole_tensor_eigs=None, date=None):
        """
        Create a DDEC class object.
        Args:
            structure (Structure class): a base class that contains lattice, coords and species information
            net_charges:
            dipoles_xyz:
            dipoles_mag:
            Qs:
            quadrupole_tensor_eigs:
        """
        self.structure = structure
        self.net_charges = net_charges
        self.dipoles_xyz = dipoles_xyz
        self.dipoles_mag = dipoles_mag
        self.Qs = Qs
        self.quadrupole_tensor_eigs = quadrupole_tensor_eigs
        self.date = date

    @staticmethod
    def from_file(filepath):
        """
        Read the positions of atoms and theirs charges
        from file "DDEC6_even_tempered_net_atomic_charges.xyz"

        Parameters:
        ----------
        filepath: str
            Path to file with atomic charges

        Returns:
        -------
        DDEC class instance
        """
        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'date': r'\s+(\d\d\d\d/\d\d/\d\d\s+\d\d:\d\d:\d\d)'}
        matches = regrep(filepath, patterns)
        date = matches['date'][0][0][0]

        natoms = int(data[0])
        x_axis = data[1].split()[10:13]
        y_axis = data[1].split()[15:18]
        z_axis = data[1].split()[20:23]
        lattice = np.array([x_axis, y_axis, z_axis], dtype=np.float32)

        for start_line, string in enumerate(data):
            if 'The following XYZ coordinates are in angstroms' in string:
                break

        coords = np.zeros((natoms, 3))
        species = [0] * natoms
        net_charges = np.zeros(natoms)
        dipoles_xyz = np.zeros((natoms, 3))
        dipoles_mag = np.zeros(natoms)
        Qs = np.zeros((natoms, 5))
        quadrupole_tensor_eigs = np.zeros((natoms, 3))

        for i, j in enumerate(range(start_line + 2, start_line + 2 + natoms)):
            line_splitted = data[j].split()
            species[i] = line_splitted[1]
            coords[i] = line_splitted[2:5]
            net_charges[i] = line_splitted[5]
            dipoles_xyz[i] = line_splitted[6:9]
            dipoles_mag[i] = line_splitted[9]
            Qs[i] = line_splitted[10:15]
            quadrupole_tensor_eigs[i] = line_splitted[15:18]

        structure = Structure(lattice, species, coords, coords_are_cartesian=True)
        return AtomicNetCharges(structure, net_charges, dipoles_xyz, dipoles_mag, Qs, quadrupole_tensor_eigs, date)


class AtomicSpinMoments:
    """Class that operates with DDEC output file DDEC6_even_tempered_atomic_spin_moments.xyz"""
    def __init__(self, structure: Structure, spin_moments, date):
        self.structure = structure
        self.spin_moments = spin_moments
        self.date = date

    @staticmethod
    def from_file(filepath):
        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'date': '\s+(\d\d\d\d/\d\d/\d\d\s+\d\d:\d\d:\d\d)'}
        matches = regrep(filepath, patterns)
        date = matches['date'][0][0][0]

        natoms = int(data[0])
        x_axis = data[1].split()[10:13]
        y_axis = data[1].split()[15:18]
        z_axis = data[1].split()[20:23]
        lattice = np.array([x_axis, y_axis, z_axis], dtype=np.float32)

        coords = np.zeros((natoms, 3))
        species = []
        spin_moments = np.zeros(natoms)

        for i, j in enumerate(range(2, 2 + natoms)):
            line_splitted = data[j].split()
            species += [line_splitted[0]]
            coords[i] = line_splitted[1:4]
            spin_moments[i] = line_splitted[4]

        structure = Structure(lattice, species, coords, coords_are_cartesian=True)
        return AtomicSpinMoments(structure, spin_moments, date)


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

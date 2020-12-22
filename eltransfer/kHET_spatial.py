import numpy as np
from . import tip_types
from .GerischerMarkus import GM
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from electrochemistry.core import constants
from electrochemistry.io import vasp

class kHET():
    """
    This class calculates heterogeneous electron transfer rate constant with spatial resolution
    """
    # TODO update tips types

    AVAILABLE_TIPS_TYPES = ['oxygen', 'IrCl6', 'RuNH3_6', 'RuNH3_6_NNN_plane', 'RuNH3_6_perpendicular',
                                 'oxygen_parallel_x', 'oxygen_parallel_y']

    def __init__(self, working_folder=''):
        if working_folder == '':
            working_folder = '.'
        self.outcar = vasp.Outcar.from_file(working_folder + '/OUTCAR')
        self.poscar = vasp.Poscar.from_file(working_folder + '/POSCAR')
        self.working_folder = working_folder
        self.path_to_data = working_folder + '/Saved_data'
        self.wavecar = None

        self.C_EDL = None
        self.T = None
        self.lambda_ = None
        self.sheet_area = None
        self.V_std = None
        self.overpot = None
        self.dE_Q = None
        self.kb_array = None
        self.E = None


    def set_parameters(self, T, lambda_, overpot=0, V_std=None, C_EDL=None, dE_Q=None, linear_constant=26, threshold_value=1e-5):
        """
        :param C_EDL: float
        Capacitance of electric double layer (microF/cm^2)
        :param T: int, float
        Temperature. It is used in computing Fermi function and distribution function of redox system states
        :param lambda_: float
        Reorganization energy in eV
        :param V_std: float
        Standart potential of the redox couple (Volts)
        :param overpot: float
        Overpotential (Volts). It shifts the electrode Fermi energy to -|e|*overpot
        :return:
        """
        #TODO chack get_DOS parameters
        self.E, DOS = self.outcar.get_DOS(zero_at_fermi=True, smearing='Gaussian', sigma=0.1, dE = 0.01)
        self.T = T
        self.lambda_ = lambda_
        self.overpot = overpot
        self.sheet_area = self._get_sheet_area() # Area of investigated surface(XY) in cm^2
        self.linear_constant = linear_constant
        if V_std == None or C_EDL == None:
            if dE_Q == None:
                raise ValueError ("Set either V_std and C_EDL or dE_Q parameter")
            else:
                self.dE_Q = dE_Q
        else:
            self.V_std = V_std
            self.C_EDL = C_EDL
            self.dE_Q = self._calculate_dE_Q()
        self.kb_array = self._get_kb_array(threshold_value)
        if self.kb_array == []:
            print('Error! kb_array is empty. Try to decrease threshold_value')

    def load_wavecar(self):
        self.wavecar = vasp.Wavecar.from_file(self.working_folder+'/WAVECAR', self.kb_array)

    def plot_distributions(self):
        #TODO make it
        pass

    def _get_kb_array(self, threshold_value):
        fermi_distribution = GM.fermi_func(self.E - self.dE_Q, self.T)
        W_ox = GM.W_ox(self.E - self.dE_Q - self.overpot, self.T, self.lambda_)
        E_satisfy_mask = W_ox * fermi_distribution > np.max(W_ox) * np.max(fermi_distribution) * threshold_value
        list_of_E_indices = [i for i, E in enumerate(E_satisfy_mask) if E]
        min_E_ind = min(list_of_E_indices)
        max_E_ind = max(list_of_E_indices)
        Erange = [self.E[min_E_ind], self.E[max_E_ind]]

        kb_array = []
        for band in range(1, self.outcar.nbands + 1):
            for kpoint in range(1, self.outcar.nkpts + 1):
                energy = self.outcar.eigenvalues[kpoint - 1][band - 1] - self.outcar.efermi
                if energy >= Erange[0] and energy < Erange[1]:
                    kb_array.append([kpoint, band])
        return kb_array

    def calculate_kHET_spatial(self, tip_type='s', z_pos=None, from_center=False, cutoff_in_Angstroms=5, all_z=False, dim='2D'):
        xlen, ylen, zlen = np.shape(self.wavecar.wavefunctions[0])

        if dim == '2D':
            k_HET_ = np.zeros((xlen, ylen))
            if z_pos == None:
                raise ValueError('For dim=2D z_pos is obligatory parameter')
            b1, b2, b3 = self.poscar._structure.lattice
            bn1 = b1 / xlen
            bn2 = b2 / ylen
            bn3 = b3 / zlen
            if b3[0] != 0.0 or b3[1] != 0.0:
                print("WARNING! You z_vector is not perpendicular to XY plane, Check calculate_kHET_spatial_2D function")
            if from_center:
                z = int(zlen // 2 + z_pos // np.linalg.norm(bn3))
            else:
                z = int(z_pos // np.linalg.norm(bn3))
            real_z_position = z_pos // np.linalg.norm(bn3) * np.linalg.norm(bn3)
            print(f"Real z_pos of tip = {real_z_position}")

            if any(tip_type == kw for kw in self.AVAILABLE_TIPS_TYPES):
                cutoff = int(cutoff_in_Angstroms // np.linalg.norm(bn1))
                if cutoff > xlen // 2 or cutoff > ylen // 2:
                    print("ERROR: Cutoff should be less than 1/2 of each dimension of the cell.  "
                          "Try to reduce cutoff. Otherwise, result could be unpredictible")
                acc_orbitals = self.generate_acceptor_orbitals(tip_type, (xlen, ylen, zlen), z_shift=z)
                if all_z == True:
                    zmin = 0
                    zmax = zlen
                elif z - cutoff >= 0 and z + cutoff <= zlen:
                    zmin = z - cutoff
                    zmax = z + cutoff
                else:
                    print("Can't reduce integrating area in z dimention. "
                          "Will calculate overlapping for all z. You can ignore this message "
                          "if you don't care about efficiency")
                    zmin = 0
                    zmax = zlen
                for i, kb in enumerate(self.wavecar.kb_array):
                    kpoint, band = kb[0], kb[1]
                    energy = self.outcar.eigenvalues[kpoint - 1][band - 1] - self.outcar.efermi
                    weight = self.outcar.weights[kpoint - 1]
                    f_fermi = GM.fermi_func(energy - self.dE_Q, self.T)
                    w_redox = GM.W_ox(energy - self.dE_Q - self.overpot, self.T, self.lambda_)
                    overlap_integrals_squared = self._get_overlap_integrals_squared(self.wavecar.wavefunctions[i], cutoff,
                                                                                    acc_orbitals, zmin, zmax)
                    # TODO check eq below
                    matrix_elements_squared = overlap_integrals_squared * self.linear_constant * np.linalg.norm(bn3) ** 3
                    k_HET_ += matrix_elements_squared * f_fermi * w_redox * weight

            elif tip_type == 's':
                for i, kb in enumerate(self.wavecar.kb_array):
                    kpoint, band = kb[0], kb[1]
                    energy = self.outcar.eigenvalues[kpoint - 1][band - 1] - self.outcar.efermi
                    weight = self.outcar.weights[kpoint - 1]
                    f_fermi = GM.fermi_func(energy - self.dE_Q, self.T)
                    w_redox = GM.W_ox(energy - self.dE_Q - self.overpot, self.T, self.lambda_)
                    matrix_elements_squared = np.abs(self.wavecar.wavefunctions[i][:, :, z]) ** 2
                    k_HET_ += matrix_elements_squared * f_fermi * w_redox * weight

            elif tip_type == 'pz':
                for i, kb in enumerate(self.wavecar.kb_array):
                    kpoint, band = kb[0], kb[1]
                    energy = self.outcar.eigenvalues[kpoint - 1][band - 1] - self.outcar.efermi
                    weight = self.outcar.weights[kpoint - 1]
                    f_fermi = GM.fermi_func(energy - self.dE_Q, self.T)
                    w_redox = GM.W_ox(energy - self.dE_Q - self.overpot, self.T, self.lambda_)
                    wf_grad_z = np.gradient(self.wavecar.wavefunctions[i], axis=2)
                    matrix_elements_squared = np.abs(wf_grad_z[:, :, z]) ** 2
                    k_HET_ += matrix_elements_squared * f_fermi * w_redox * weight
            else:
                print(f"Try another tip type, for now {tip_type} is unavailiable")

        elif dim == '3D':
            k_HET_ = np.zeros((xlen, ylen, zlen))

            if tip_type == 's':
                for i, kb in enumerate(self.wavecar.kb_array):
                    kpoint, band = kb[0], kb[1]
                    energy = self.outcar.eigenvalues[kpoint - 1][band - 1] - self.outcar.efermi
                    weight = self.outcar.weights[kpoint - 1]
                    f_fermi = GM.fermi_func(energy - self.dE_Q, self.T)
                    w_redox = GM.W_ox(energy - self.dE_Q - self.overpot, self.T, self.lambda_)
                    matrix_elements_squared = np.abs(self.wavecar.wavefunctions[i]) ** 2
                    k_HET_ += matrix_elements_squared * f_fermi * w_redox * weight

            elif tip_type == 'pz':
                for i, kb in enumerate(self.wavecar.kb_array):
                    kpoint, band = kb[0], kb[1]
                    energy = self.outcar.eigenvalues[kpoint - 1][band - 1] - self.outcar.efermi
                    weight = self.outcar.weights[kpoint - 1]
                    f_fermi = GM.fermi_func(energy - self.dE_Q, self.T)
                    w_redox = GM.W_ox(energy - self.dE_Q - self.overpot, self.T, self.lambda_)
                    wf_grad_z = np.gradient(self.wavecar.wavefunctions[i], axis=2)
                    matrix_elements_squared = np.abs(wf_grad_z) ** 2
                    k_HET_ += matrix_elements_squared * f_fermi * w_redox * weight

        else:
            raise ValueError("dim should be 3D or 2D")

        # TODO: check THIS below
        #k_HET_ *= 2 * np.pi / constants.PLANCK_CONSTANT
        return k_HET_

    def plot_2D(self, func, show=True, save=False, filename='fig.png'):
        """
        Function for plotting 2D images
        """
        xlen, ylen = np.shape(func)
        b1, b2, b3 = self.poscar._structure.lattice
        bn1 = b1 / xlen
        bn2 = b2 / ylen

        R = []
        for j in range(ylen):
            for i in range(xlen):
                R.append(i * bn1 + j * bn2)
        X = np.array([x[0] for x in R])
        Y = np.array([x[1] for x in R])

        Z = func.transpose().flatten()
        xi = np.linspace(X.min(), X.max(), 1000)
        yi = np.linspace(Y.min(), Y.max(), 1000)
        zi = griddata((X, Y), Z, (xi[None, :], yi[:, None]), method='cubic')
        plt.contourf(xi, yi, zi, 500, cmap=plt.cm.rainbow)
        ax = plt.gca()
        ax.set_aspect('equal')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad=0.1)
        plt.colorbar(cax=cax)
        if show == True:
            plt.show()
        if save == True:
            plt.savefig(filename, dpi=300)
            plt.close()

    def _calculate_dE_Q(self):
        """
        This function calls GerischerMarcus module with GM class to calculate distribution of redox species
        and Fermi-Dirac distribution according to Gerischer-Marcus formalism
        :return:
        """
        gm = GM(path_to_data=self.path_to_data)
        gm.set_params(self.C_EDL, self.T, self.lambda_, self.sheet_area)
        return gm.compute_distributions(self.V_std, overpot=self.overpot, add_info=True)[2]

    def _get_sheet_area(self):
        """
        Inner function to calculate sheet_area (XY plane) in cm^2
        """
        b1, b2, b3 = self.poscar._structure.lattice
        return np.linalg.norm(np.cross(b1, b2))*1e-16

    def generate_acceptor_orbitals(self, orb_type, shape, z_shift=0, x_shift=0, y_shift=0):
        """
        This is generator of acceptor orbitals using tip_types.py module
        :return:
        """
        bohr_radius = 0.529177  #TODO Check
        b1, b2, b3 = self.poscar._structure.lattice
        bn1 = b1 / shape[0]
        bn2 = b2 / shape[1]
        bn3 = b3 / shape[2]
        basis = np.array([bn1, bn2, bn3])
        transition_matrix = basis.transpose()
        number_of_orbitals = len(tip_types.orbitals(orb_type))
        acc_orbitals = []
        for i in range(number_of_orbitals):
            acc_orbitals.append(np.zeros(shape))
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if i - x_shift >= shape[0] / 2:
                        x = i - shape[0] - x_shift
                    else:
                        x = i - x_shift
                    if j - y_shift >= shape[1] / 2:
                        y = j - shape[1] - y_shift
                    else:
                        y = j - y_shift
                    if k - z_shift >= shape[2] / 2:
                        z = k - shape[2] - z_shift
                    else:
                        z = k - z_shift
                    r = np.dot(transition_matrix, np.array([x, y, z])) / bohr_radius
                    for o, orbital in enumerate(acc_orbitals):
                        orbital[i][j][k] = tip_types.orbitals(orb_type)[o](r)
        return acc_orbitals

    def _get_overlap_integrals_squared(self, wf, cutoff, acc_orbitals, zmin, zmax):
        xlen, ylen, zlen = np.shape(wf)
        overlap_integrals_squared = np.zeros((xlen, ylen))
        for i in range(xlen):
            if i - cutoff < 0:
                wf_rolled_x = np.roll(wf, cutoff, axis=0)
                orb_rolled_x = []
                for orbital in acc_orbitals:
                    orb_rolled_x.append(np.roll(orbital, i + cutoff, axis=0))
                xmin = i
                xmax = i + cutoff * 2
            elif i - cutoff >= 0 and i + cutoff <= xlen:
                wf_rolled_x = np.copy(wf)
                orb_rolled_x = []
                for orbital in acc_orbitals:
                    orb_rolled_x.append(np.roll(orbital, i, axis = 0))
                xmin = i - cutoff
                xmax = i + cutoff
            elif i + cutoff > xlen:
                wf_rolled_x = np.roll(wf, -cutoff, axis=0)
                orb_rolled_x = []
                for orbital in acc_orbitals:
                    orb_rolled_x.append(np.roll(orbital, i - cutoff, axis=0))
                xmin = i - cutoff * 2
                xmax = i
            else:
                print(f"ERROR: for i = {i} something with rolling arrays along x goes wrong")
            for j in range(ylen):
                if j - cutoff < 0:
                    wf_rolled = np.roll(wf_rolled_x, cutoff, axis=1)
                    orb_rolled = []
                    for orbital in orb_rolled_x:
                        orb_rolled.append(np.roll(orbital, j + cutoff, axis=1))
                    ymin = j
                    ymax = j + cutoff * 2
                elif j - cutoff >= 0 and j + cutoff <= ylen:
                    wf_rolled = np.copy(wf_rolled_x)
                    orb_rolled = []
                    for orbital in orb_rolled_x:
                        orb_rolled.append(np.roll(orbital, j, axis=1))
                    ymin = j - cutoff
                    ymax = j + cutoff
                elif j + cutoff > ylen:
                    wf_rolled = np.roll(wf_rolled_x, -cutoff, axis=1)
                    orb_rolled = []
                    for orbital in orb_rolled_x:
                        orb_rolled.append(np.roll(orbital, j - cutoff, axis=1))
                    ymin = j - cutoff * 2
                    ymax = j
                else:
                    print(f"ERROR: for i = {i} something with rolling arrays along y goes wrong")

                integral = []
                for orbital in orb_rolled:
                    integral.append(np.linalg.norm(wf_rolled[xmin:xmax, ymin:ymax, zmin:zmax]*\
                                   orbital[xmin:xmax, ymin:ymax, zmin:zmax]))
                overlap_integrals_squared[i][j] = max(integral) ** 2
        return overlap_integrals_squared

    def save_as_cube(self, array, name, dir):
        # TODO: rewrite
        import os
        if not os.path.exists(dir):
            print(f"Directory {dir} does not exist. Creating directory")
            os.mkdir(dir)

        shape = np.shape(array)
        with open(self.working_folder + '/POSCAR') as inf: #TODO get data from poscar class would be better
            lines = inf.readlines()
            natoms = sum(map(int, lines[6].strip().split()))
            atomtypes = lines[5].strip().split()
            numbers_of_atoms = list(map(int, lines[6].strip().split()))
            type_of_i_atom = []
            basis = []
            for i in [2, 3, 4]:
                vector = list(map(float, lines[i].strip().split()))
                basis.append(vector)
            basis = np.array(basis)
            for i, number in enumerate(numbers_of_atoms):
                for j in range(number):
                    type_of_i_atom.append(atomtypes[i])

        with open(dir + '/' + name + '.cube', 'w') as ouf:
            ouf.write(' This file is generated using stm.py module\n')
            ouf.write(' Good luck\n')
            ouf.write(' ' + str(natoms) + '\t0.000\t0.000\t0.000\n')
            ouf.write(' ' + str(-shape[0]) + lines[2])
            ouf.write(' ' + str(-shape[1]) + lines[3])
            ouf.write(' ' + str(-shape[2]) + lines[4])
            for i, line in enumerate(lines[8:natoms + 8]):
                coordinate = np.array(list(map(float, line.strip().split())))
                if lines[7].strip() == 'Direct':
                    coordinate = coordinate.dot(basis)
                    coordinate *= constants.BOHR_RADIUS
                elif lines[7].strip() == 'Cartesian':
                    coordinate *= constants.BOHR_RADIUS
                else:
                    print('WARNING!!! Cannot read POSCAR correctly')
                atomtype = type_of_i_atom[i]
                atomnumber = list(constants.ElemNum2Name.keys())[list(constants.ElemNum2Name.values()).index(atomtype)]
                ouf.write(
                    ' ' + str(atomnumber) + '\t0.00000\t' + str(coordinate[0]) + '\t' + str(coordinate[1]) + '\t' + str(
                        coordinate[2]) + '\n')
            counter = 0
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        ouf.write(str('%.5E' % array[i][j][k]) + '  ')
                        counter += 1
                        if counter % 6 == 0:
                            ouf.write('\n  ')
                    ouf.write('\n  ')
        print(f"File {name} saved")
import numpy as np
import os
import time
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage.filters import gaussian_filter1d
import tip_types
from echem.core import ElemNum2Name
plt.switch_backend('agg')


class STM:
    """
    This class is for computing 2D and 3D STM images with different theory levels:
    Tersoff-Hummann, Chen, analytical acceptor wavefucntions (oxygen)
    Also this class can calculate 2D ECSTM images using GerisherMarcus module

    In current version of program this class is no more needed. Most of functions are relocated to kHET_spatial.py
    """

    PLANCK_CONSTANT = 4.135667662e-15       # Planck's constant in eV*s
    BOLTZMANN_CONSTANT = 8.617333262145e-5  # Boltzmann's constant in eV/K
    ELEM_CHARGE = 1.60217662e-19            # Elementary charge in Coulombs
    BOHR_RADIUS = 1.88973
    AVAILABLE_TIPS_TYPES = ['oxygen', 'IrCl6', 'RuNH3_6', 'RuNH3_6_NNN_plane', 'RuNH3_6_perpendicular', 'oxygen_parallel_x', 'oxygen_parallel_y']

    def __init__(self, working_folder):
        self.WF_ijke = None
        self.CD_ijke = None
        self.dE = None
        self.energy_range = None
        self.working_folder = working_folder
        self.path_to_data = self.working_folder+'/Saved_data'
        self.WF_data_path = self.path_to_data+'/WF_data.npy'
        self.WF_ijke_path = self.path_to_data+'/WF_ijke.npy'
        self.CD_ijke_path = self.path_to_data+'/CD_ijke.npy'
        self.poscar_path = working_folder+'/POSCAR'

    def save_as_vasp(self, array, name, dir):
        if not os.path.exists(dir):
            print(f"Directory {dir} does not exist. Creating directory")
            os.mkdir(dir)
        shape = np.shape(array)
        with open(self.poscar_path) as inf:
            lines = inf.readlines()
            natoms = int(lines[6].strip())
        with open(dir+'/'+ name + '.vasp', 'w') as ouf:
            ouf.writelines(lines[:natoms+8])
            ouf.write('\n  ' + str(shape[0]) + '  ' + str(shape[1]) + '  ' + str(shape[2]) + '\n  ')
            counter = 0
            for k in range(shape[2]):
                for j in range(shape[1]):
                    for i in range(shape[0]):
                        ouf.write(str('%.8E' % array[i][j][k]) + '  ')
                        counter += 1
                        if counter % 10 == 0:
                            ouf.write('\n  ')
        print(f"File {name} saved")

    def save_as_cube(self, array, name, dir):
        if not os.path.exists(dir):
            print(f"Directory {dir} does not exist. Creating directory")
            os.mkdir(dir)
            
        shape = np.shape(array)
        with open(self.poscar_path) as inf:
            lines = inf.readlines()
            natoms = sum(map(int, lines[6].strip().split()))
            atomtypes = lines[5].strip().split()
            numbers_of_atoms = list(map(int, lines[6].strip().split()))
            type_of_i_atom = []
            basis = []
            for i in [2,3,4]:
                vector = list(map(float, lines[i].strip().split()))
                basis.append(vector)
            basis = np.array(basis)
            for i, number in enumerate(numbers_of_atoms):
                for j in range(number):
                    type_of_i_atom.append(atomtypes[i])
                
        with open(dir+'/'+ name + '.cube', 'w') as ouf:
            ouf.write(' This file is generated using stm.py module\n')
            ouf.write(' Good luck\n')
            ouf.write(' ' + str(natoms) + '\t0.000\t0.000\t0.000\n')
            ouf.write(' ' + str(-shape[0]) + lines[2])
            ouf.write(' ' + str(-shape[1]) + lines[3])
            ouf.write(' ' + str(-shape[2]) + lines[4])
            for i, line in enumerate(lines[8:natoms+8]):
                coordinate = np.array(list(map(float, line.strip().split())))
                if lines[7].strip() == 'Direct':
                    coordinate = coordinate.dot(basis)
                    coordinate *= self.BOHR_RADIUS
                elif lines[7].strip() == 'Cartesian':
                    coordinate *= self.BOHR_RADIUS
                else:
                    print ('WARNING!!! Cannot read POSCAR correctly')
                atomtype = type_of_i_atom[i]
                atomnumber = list(ElemNum2Name.keys())[list(ElemNum2Name.values()).index(atomtype)]
                ouf.write(' ' + str(atomnumber) + '\t0.00000\t' + str(coordinate[0])+ '\t' + str(coordinate[1])+ '\t' + str(coordinate[2])+'\n')
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


    def load_data(self):
        WF_data = np.load(self.WF_data_path, allow_pickle=True).item()
        self.dE = WF_data['dE']
        self.energy_range = WF_data['energy_range']
        self.WF_ijke = np.load(self.WF_ijke_path)
        self.CD_ijke = np.load(self.CD_ijke_path)

    def set_ecstm_parameters(self, C_EDL, T, lambda_, V_std, overpot,
                             effective_freq, linear_constant, threshold_value):
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
        :param effective_freq: float
        Effective frequency of redox species motion
        :param effective_freq: float
        Linear constant of proportionality of Hif and Sif: Hif = linear_constant * Sif
        :param threshold_value: float
        Minimum value of y_redox and y_fermi to be considered in integral
        :return:
        """
        self.C_EDL = C_EDL
        self.T = T
        self.lambda_ = lambda_
        self.sheet_area = self._get_sheet_area() # Area of investigated surface(XY) in cm^2
        self.V_std = V_std
        self.overpot = overpot
        self.effective_freq = effective_freq
        self.linear_constant = linear_constant
        self.threshold_value = threshold_value

    def load_data_for_ecstm(self):
        """
        This inner function load necessary data for ECSTM calculations
        :param outcar_path: str
        path to OUTCAR vasp file
        :param locpot_path: str
        path to LOCPOT vasp file
        :return:
        """
        try:
            self.E = np.load(self.path_to_data+'/E.npy')
            self.DOS = np.load(self.path_to_data+'/DOS.npy')
        except:
            import preprocessing_v2 as preprocessing
            p = preprocessing.Preprocessing(working_folder=self.working_folder)
            p.process_OUTCAR()
            self.E, self.DOS, dE_new = p.get_DOS(self.energy_range, self.dE)
            if dE_new != self.dE:
                print("WARNING! Something wrong with dE during DOS calculations")
            np.save(self.path_to_data+'/DOS.npy', self.DOS)
            np.save(self.path_to_data+'/E.npy', self.E)
        try:
            self.efermi = np.load(self.path_to_data+'/efermi.npy')
        except:
            print(f"ERROR! {self.path_to_data}/efermi.npy does not exist. Try to preprocess data")
        try:
            self.vacuum_lvl = np.load(self.path_to_data+'vacuum_lvl.npy')
        except:
            from pymatgen.io.vasp.outputs import Locpot
            locpot = Locpot.from_file(self.working_folder+'/LOCPOT')
            avr = locpot.get_average_along_axis(2)
            self.vacuum_lvl = np.max(avr)
            np.save(self.path_to_data+'/vacuum_lvl.npy', self.vacuum_lvl)

    def _calculate_distributions(self):
        """
        This function calls GerisherMarcus module with GM class to calculate distribution of redox species
        and Fermi-Dirac distribution according to Gerisher-Marcformalismizm
        :return:
        """
        import GerischerMarkus as gm
        gerisher_marcus_obj = gm.GM(path_to_data = self.path_to_data)
        gerisher_marcus_obj.set_params(self.C_EDL, self.T, self.lambda_, self.sheet_area)
        self.y_fermi, self.y_redox = gerisher_marcus_obj.compute_distributions(self.V_std, overpot=self.overpot)
        return gerisher_marcus_obj

    @staticmethod
    def _nearest_array_indices(array, value):
        i = 0
        while value > array[i]:
            i += 1
        return i - 1, i

    def plot_distributions(self, E_range=[-7, 4], dE=None, sigma=2, fill_area_lower_Fermi_lvl=True,
                 plot_Fermi_Dirac_distib=True, plot_redox_distrib=True):
        a = self._calculate_distributions()
        if dE == None:
            dE = self.dE
        if dE != self.dE or E_range[0] < self.energy_range[0] or E_range[1] > self.energy_range[1]:
            import preprocessing_v2 as prep
            p = prep.Preprocessing(working_folder = self.working_folder)
            E, DOS, dE_new = p.get_DOS(E_range, dE)
        else:
            E, DOS = a.E, a.DOS
        n_1, n_2 = self._nearest_array_indices(E, a.dE_Q_eq)
        if sigma > 0 and sigma != None:
            DOS = gaussian_filter1d(DOS, sigma)
        plt.plot(E, DOS)
        if fill_area_lower_Fermi_lvl == True:
            plt.fill_between(E[:n_2], DOS[:n_2])
        if plot_Fermi_Dirac_distib == True:
            plt.plot(a.E, self.y_fermi * 30)
        if plot_redox_distrib == True:
            plt.plot(a.E, self.y_redox * 10)
        plt.xlabel('E, eV')
        plt.ylabel('DOS, states/eV/cell')
        plt.xlim(E_range)
        plt.savefig(f"distributions_{self.V_std}_{self.overpot}.png", dpi=300)
        plt.close()

    def _kT_in_eV(self, T):
        return T * self.BOLTZMANN_CONSTANT

    def _get_kappa(self, matrix_elements_squared):
        lz_factor = 2 * matrix_elements_squared / self.PLANCK_CONSTANT / self.effective_freq * math.sqrt(
                                            math.pi / self.lambda_ / self._kT_in_eV(self.T))
        kappa = 1 - np.exp(-2 * math.pi * lz_factor)
        return kappa

    def _get_basis_vectors(self):
        """
        This function processes POSCAR file to extract vectors of the simulation box
        :param poscar_path:
        :return:
        """
        with open(self.poscar_path) as inf:
            lines = inf.readlines()
            b1 = np.array(list(map(float, lines[2].strip().split())))
            b2 = np.array(list(map(float, lines[3].strip().split())))
            b3 = np.array(list(map(float, lines[4].strip().split())))
        return b1, b2, b3

    def _get_sheet_area(self):
        """
        Inner function to calculate sheet_area (XY plane) in cm^2
        """
        b1, b2, b3 = self._get_basis_vectors()
        return np.linalg.norm(np.cross(b1,b2))*1e-16

    def _plot_contour_map(self, function, X, Y, dir, filename):
        """
        Function for plotting 2D images
        :param function: Z values
        :param X: X values
        :param Y: Y values
        :param dir: directory in which contour map will be saved
        :param filename: filename of image
        :return:
        """
        if not os.path.exists(dir):
            print(f"Directory {dir} does not exist. Creating directory")
            os.mkdir(dir)

        t = time.time()
        Z = function.transpose().flatten()
        xi = np.linspace(X.min(), X.max(), 1000)
        yi = np.linspace(Y.min(), Y.max(), 1000)
        zi = griddata((X, Y), Z, (xi[None, :], yi[:, None]), method='cubic')
        plt.contourf(xi, yi, zi, 500, cmap=plt.cm.rainbow)
        ax = plt.gca()
        ax.set_aspect('equal')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad=0.1)
        plt.colorbar(cax=cax)
        plt.savefig(dir+'/'+filename+'.png', dpi=300)
        plt.close()
        print (f"Plotting map takes {time.time()-t} sec")


    def _get_overlap_integrals_squared(self, e, cutoff, acc_orbitals, zmin, zmax):
        WF_ijk = self.WF_ijke[:, :, :, e]
        xlen, ylen, zlen = np.shape(WF_ijk)
        overlap_integrals_squared = np.zeros((xlen, ylen))
        if np.allclose(WF_ijk, np.zeros((xlen, ylen, zlen))):
            # If WF_ijk array for current energy is empty, code goes to the next energy
            print(f"WF_ijk array for energy {e} is empty. Going to the next")
            return overlap_integrals_squared
        for i in range(xlen):
            if i - cutoff < 0:
                WF_ijk_rolled_x = np.roll(WF_ijk, cutoff, axis=0)
                orb_rolled_x = []
                for orbital in acc_orbitals:
                    orb_rolled_x.append(np.roll(orbital, i + cutoff, axis=0))
                xmin = i
                xmax = i + cutoff * 2
            elif i - cutoff >= 0 and i + cutoff <= xlen:
                WF_ijk_rolled_x = np.copy(WF_ijk)
                orb_rolled_x = []
                for orbital in acc_orbitals:
                    orb_rolled_x.append(np.roll(orbital, i, axis = 0))
                xmin = i - cutoff
                xmax = i + cutoff
            elif i + cutoff > xlen:
                WF_ijk_rolled_x = np.roll(WF_ijk, -cutoff, axis=0)
                orb_rolled_x = []
                for orbital in acc_orbitals:
                    orb_rolled_x.append(np.roll(orbital, i - cutoff, axis=0))
                xmin = i - cutoff * 2
                xmax = i
            else:
                print(f"ERROR: for i = {i} something with rolling arrays along x goes wrong")
            for j in range(ylen):
                if j - cutoff < 0:
                    WF_ijk_rolled = np.roll(WF_ijk_rolled_x, cutoff, axis=1)
                    orb_rolled = []
                    for orbital in orb_rolled_x:
                        orb_rolled.append(np.roll(orbital, j + cutoff, axis=1))
                    ymin = j
                    ymax = j + cutoff * 2
                elif j - cutoff >= 0 and j + cutoff <= ylen:
                    WF_ijk_rolled = np.copy(WF_ijk_rolled_x)
                    orb_rolled = []
                    for orbital in orb_rolled_x:
                        orb_rolled.append(np.roll(orbital, j, axis=1))
                    ymin = j - cutoff
                    ymax = j + cutoff
                elif j + cutoff > ylen:
                    WF_ijk_rolled = np.roll(WF_ijk_rolled_x, -cutoff, axis=1)
                    orb_rolled = []
                    for orbital in orb_rolled_x:
                        orb_rolled.append(np.roll(orbital, j - cutoff, axis=1))
                    ymin = j - cutoff * 2
                    ymax = j
                else:
                    print(f"ERROR: for i = {i} something with rolling arrays along y goes wrong")

                integral = []
                for orbital in orb_rolled:
                    integral.append(np.linalg.norm(WF_ijk_rolled[xmin:xmax, ymin:ymax, zmin:zmax]*\
                                   orbital[xmin:xmax, ymin:ymax, zmin:zmax]))
                overlap_integrals_squared[i][j] = max(integral) ** 2

        return overlap_integrals_squared

    def generate_acceptor_orbitals(self, orb_type, shape, z_shift=0, x_shift=0, y_shift=0):
        """
        This is generator of acceptor orbitals using tip_types.py module
        :return:
        """
        bohr_radius = 0.529 #TODO better to get it from core.constants
        b1, b2, b3 = self._get_basis_vectors()
        bn1 = b1 / shape[0]
        bn2 = b2 / shape[1]
        bn3 = b3 / shape[2]
        basis = np.array([bn1, bn2, bn3])
        transition_matrix = basis.transpose()
        numer_of_orbitals = len(tip_types.orbitals(orb_type))
        acc_orbitals = []
        for i in range(numer_of_orbitals):
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


    def calculate_2D_ECSTM(self, z_position, tip_type='oxygen', dir='ECSTM', cutoff_in_Angstroms=5, all_z=False, from_CD=False):
        """
        This function calculate 2D ECSTM images using GerisherMarcus module
        :param z_position: position of tip under the investigated surface
        :param tip_type: Type of tip orbital. Currenty only "oxygen" is available
        :param dir: directiry to save images
        :param cutoff_in_Angstroms: Cutoff over x,y and z, when overlap integral is calculated.
        E.g. for x direction area of integration will be from x-cutoff to x+cutoff
        :return:
        """

        print(f"Starting to calculate 2D ECSTM; tip_type = {tip_type};")
        xlen, ylen, zlen, elen = np.shape(self.WF_ijke)
        b1, b2, b3 = self._get_basis_vectors()
        bn1 = b1 / xlen
        bn2 = b2 / ylen
        bn3 = b3 / zlen
        
        print (bn1, bn2, xlen, ylen)

        if b3[0] != 0.0 or b3[1] != 0.0:
            print("WARNING! You z_vector is not perpendicular to XY plane, Check calculate_2D_STM function")

        z = int(zlen // 2 + z_position // np.linalg.norm(bn3))
        real_z_position = z_position // np.linalg.norm(bn3) * np.linalg.norm(bn3)
        print(f"Real z_position of tip = {real_z_position}")

        R = []
        for j in range(ylen):
            for i in range(xlen):
                R.append(i * bn1 + j * bn2)
        X = np.array([x[0] for x in R])
        Y = np.array([x[1] for x in R])

        if any(tip_type == kw for kw in self.AVAILABLE_TIPS_TYPES):
            cutoff = int(cutoff_in_Angstroms // np.linalg.norm(bn1))
            if cutoff > xlen // 2 or cutoff > ylen // 2:
                print("ERROR: Cutoff should be less than 1/2 of each dimension of the cell.  "
                      "Try to reduce cutoff. Otherwise, result could be unpredictible")
            print(f"Cutoff in int = {cutoff}")
            ECSTM_ij = np.zeros((xlen, ylen))
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
            self._calculate_distributions()
            for e in range(elen):
                if self.y_redox[e] < self.threshold_value or self.y_fermi[e] < self.threshold_value:
                    continue
                overlap_integrals_squared = self._get_overlap_integrals_squared(e, cutoff, acc_orbitals, zmin, zmax)
                matrix_elements_squared = overlap_integrals_squared * self.linear_constant * np.linalg.norm(bn3)**3
                #print ('Hif = ', matrix_elements_squared, 'eV')
                #kappa = self._get_kappa(matrix_elements_squared)
                #ECSTM_ij += kappa * self.y_fermi[e] * self.DOS[e] * self.y_redox[e] * self.dE * 2 * math.pi *\
                #            self.ELEM_CHARGE / self.PLANCK_CONSTANT
                ECSTM_ij += 2 * np.pi / self.PLANCK_CONSTANT * matrix_elements_squared * self.y_fermi[e] * self.DOS[e] * self.y_redox[e] * self.dE


                
        elif tip_type == 's':
            ECSTM_ij = np.zeros((xlen, ylen))
            self._calculate_distributions()
            for e in range(elen):
                if self.y_redox[e] < self.threshold_value or self.y_fermi[e] < self.threshold_value:
                    continue
                if from_CD == True:
                    matrix_elements_squared = self.CD_ijke[:, :, z, e]
                    ECSTM_ij += 2 * np.pi / self.PLANCK_CONSTANT * matrix_elements_squared * self.y_fermi[e] * self.DOS[e] * self.y_redox[e] * self.dE
                else:   
                    matrix_elements_squared = np.abs(self.WF_ijke[:, :, z, e]) ** 2
                    #print ('Hif = ', matrix_elements_squared, 'eV')
                    #kappa = self._get_kappa(matrix_elements_squared)
                    #ECSTM_ij += kappa * self.y_fermi[e] * self.DOS[e] * self.y_redox[e] * self.dE * 2 * math.pi *\
                    #            self.ELEM_CHARGE / self.PLANCK_CONSTANT\
                    #TODO - check formula!!!
                    ECSTM_ij += 2 * np.pi / self.PLANCK_CONSTANT * matrix_elements_squared * self.y_fermi[e] * self.DOS[e] * self.y_redox[e] * self.dE

                
        elif tip_type == 'pz':
            ECSTM_ij = np.zeros((xlen, ylen))
            self._calculate_distributions()
            for e in range(elen):
                if self.y_redox[e] < self.threshold_value or self.y_fermi[e] < self.threshold_value:
                    continue
                grad_z_WF_for_e = np.gradient(self.WF_ijke[:, :, :, e], axis=2)
                matrix_elements_squared = (np.abs(grad_z_WF_for_e[:, :, z])) ** 2
                print ('Hif = ', matrix_elements_squared, 'eV')
                #kappa = self._get_kappa(matrix_elements_squared)
                #ECSTM_ij += kappa * self.y_fermi[e] * self.DOS[e] * self.y_redox[e] * self.dE * 2 * math.pi *\
                #            self.ELEM_CHARGE / self.PLANCK_CONSTANT
                ECSTM_ij += 2 * np.pi / self.PLANCK_CONSTANT * matrix_elements_squared * self.y_fermi[e] * self.DOS[e] * self.y_redox[e] * self.dE
                
        if from_CD:
            cd_flag='from_CD'
        else:
            cd_flag=''
        filename = f"ecstm_{'%.2f'%real_z_position}_{tip_type}_{cd_flag}_{self.V_std}_{self.overpot}_{self.lambda_}"
        self._plot_contour_map(ECSTM_ij, X, Y, dir, filename)
        np.save(dir+'/'+filename, ECSTM_ij)


    def calculate_2D_STM(self, STM_energy_range, z_position, tip_type='s', dir='STM_2D', cutoff_in_Angstroms=5, all_z=False, from_CD=False):
        """
        Function for calculation 2D STM images
        :param STM_energy_range: Energy range regarding Fermi level
        :param z_position: Position of tip under the surface. This distance is in Angstroms assuming
        that investigated surface XY is in the center of calculation cell (the middle of z axes)
        :param tip_type: type of tip orbital
        :param dir: directory for saving images
        :param cutoff_in_Angstroms: Cutoff over x,y and z, when overlap integral is calculated.
        E.g. for x direction area of integration will be from x-cutoff to x+cutoff
        :return:
        """
        emin = int((STM_energy_range[0] - self.energy_range[0]) / self.dE)
        emax = int((STM_energy_range[1] - self.energy_range[0]) / self.dE)
        print(f"Starting to calculate 2D STM; tip_type = {tip_type}; STM energy range = {STM_energy_range}")
        e_array = np.arange(emin, emax)
        xlen, ylen, zlen, elen = np.shape(self.WF_ijke)
        b1, b2, b3 = self._get_basis_vectors()
        bn1 = b1 / xlen
        bn2 = b2 / ylen
        bn3 = b3 / zlen

        if b3[0] != 0.0 or b3[1] != 0.0 :
            print("WARNING! You z_vector is not perpendicular to XY plane, Check calculate_2D_STM function")

        z = int(zlen // 2 + z_position // np.linalg.norm(bn3))
        real_z_position = z_position // np.linalg.norm(bn3) * np.linalg.norm(bn3)
        print(f"Real z_position of tip = {real_z_position}")
        
        R = []
        for j in range(ylen):
            for i in range(xlen):
                R.append(i * bn1 + j * bn2)
        X = np.array([x[0] for x in R])
        Y = np.array([x[1] for x in R])

        if any(tip_type == kw for kw in self.AVAILABLE_TIPS_TYPES):
            cutoff = int(cutoff_in_Angstroms // np.linalg.norm(bn1))
            if cutoff > xlen//2 or cutoff > ylen//2:
                print("ERROR: Cutoff should be less than 1/2 of each dimension of the cell.  "
                      "Try to reduce cutoff. Otherwise, result could be unpredictible")
            print (f"Cutoff in int = {cutoff}")
            STM_ij = np.zeros((xlen, ylen))
            acc_orbitals = self.generate_acceptor_orbitals(tip_type, (xlen, ylen, zlen), z_shift=z)
            if all_z == True:
                zmin = 0
                zmax = zlen
            elif z - cutoff >=0 and z + cutoff <= zlen:
                zmin = z-cutoff
                zmax = z+cutoff
            else:
                print("Can't reduce integrating area in z dimention. "
                      "Will calculate overlapping for all z. You can ignore this message "
                      "if you don't care about efficiency")
                zmin = 0
                zmax = zlen
            for e in e_array:
                overlap_integrals_squared = self._get_overlap_integrals_squared(e, cutoff, acc_orbitals, zmin, zmax)
                STM_ij += overlap_integrals_squared
                
        elif tip_type == 's':
            if from_CD == True:
                STM_ij = np.zeros((xlen, ylen))
                for e in e_array:
                    STM_ij += self.CD_ijke[:, :, z, e]
            else:
                STM_ij = np.zeros((xlen, ylen))
                for e in e_array:
                    STM_ij += np.abs(self.WF_ijke[:, :, z, e])**2
            
        elif tip_type == 'pz':
            STM_ij = np.zeros((xlen, ylen))
            for e in e_array:
                grad_z_WF_for_e = np.gradient(self.WF_ijke[:, :, :, e], axis=2)
                STM_ij += (np.abs(grad_z_WF_for_e[:, :, z])) ** 2

        elif tip_type == 'px':
            STM_ij = np.zeros((xlen, ylen))
            for e in e_array:
                grad_x_WF_for_e = np.gradient(self.WF_ijke[:, :, z, e], axis=0)
                STM_ij += (np.abs(grad_x_WF_for_e)) ** 2

        elif tip_type == 'p30':
            STM_ij = np.zeros((xlen, ylen))
            for e in e_array:
                grad_x_WF_for_e = np.gradient(self.WF_ijke[:, :, z, e], axis=0)
                grad_y_WF_for_e = np.gradient(self.WF_ijke[:, :, z, e], axis=1)
                STM_ij += (np.abs(grad_y_WF_for_e + grad_x_WF_for_e)) ** 2

        elif tip_type == 'p60':
            STM_ij = np.zeros((xlen, ylen))
            for e in e_array:
                grad_y_WF_for_e = np.gradient(self.WF_ijke[:, :, z, e], axis=1)
                STM_ij += (np.abs(grad_y_WF_for_e)) ** 2

        elif tip_type == 'p90':
            STM_ij = np.zeros((xlen, ylen))
            for e in e_array:
                grad_x_WF_for_e = np.gradient(self.WF_ijke[:, :, z, e], axis=0)
                grad_y_WF_for_e = np.gradient(self.WF_ijke[:, :, z, e], axis=1)
                STM_ij += (np.abs(grad_y_WF_for_e - grad_x_WF_for_e / 2.0)) ** 2

        elif tip_type == 'p120':
            STM_ij = np.zeros((xlen, ylen))
            for e in e_array:
                grad_x_WF_for_e = np.gradient(self.WF_ijke[:, :, z, e], axis=0)
                grad_y_WF_for_e = np.gradient(self.WF_ijke[:, :, z, e], axis=1)
                STM_ij += (np.abs(grad_y_WF_for_e - grad_x_WF_for_e)) ** 2

        elif tip_type == 'p150':
            STM_ij = np.zeros((xlen, ylen))
            for e in e_array:
                grad_x_WF_for_e = np.gradient(self.WF_ijke[:, :, z, e], axis=0)
                grad_y_WF_for_e = np.gradient(self.WF_ijke[:, :, z, e], axis=1)
                STM_ij += (np.abs(grad_y_WF_for_e / 2.0 - grad_x_WF_for_e)) ** 2

        elif tip_type == 's+pz':
            STM_ij = np.zeros((xlen, ylen))
            for e in e_array:
                grad_z_WF_for_e = np.gradient(self.WF_ijke[:, :, :, e], axis=2)
                STM_ij += (np.abs(self.WF_ijke[:, :, z, e] + grad_z_WF_for_e[:, :, z])) ** 2
       
        if from_CD:
            cd_flag='from_CD'
        else:
            cd_flag=''                
        filename = 'stm_' + str('%.2f' % real_z_position) + '_' + tip_type + '_' + cd_flag
        self._plot_contour_map(STM_ij, X, Y, dir, filename)
        np.save(dir+'/'+filename, STM_ij)


    def calculate_3D_STM(self, STM_energy_range, tip_type='s', dir='STM_3D', format='cube', from_CD=False):
        emin = int((STM_energy_range[0] - self.energy_range[0]) / self.dE)
        emax = int((STM_energy_range[1] - self.energy_range[0]) / self.dE)
        print ('Starting to calculate 3D STM; tip_type ='+tip_type+'; STM energy range = ', STM_energy_range)
        e_array = np.arange(emin, emax)
        xlen, ylen, zlen, elen = np.shape(self.WF_ijke)

        if tip_type == 's':
            if from_CD == True:
                STM_ijk = np.zeros((xlen, ylen, zlen))
                for e in e_array:
                    STM_ijk += self.CD_ijke[:, :, :, e]
            else:
                STM_ijk = np.zeros((xlen, ylen, zlen))
                for e in e_array:
                    STM_ijk += np.abs(self.WF_ijke[:, :, :, e]) ** 2

        elif tip_type == 'pz':
            STM_ijk = np.zeros((xlen, ylen, zlen))
            for e in e_array:
                grad_z_WF_for_e = np.gradient(self.WF_ijke[:, :, :, e], axis=2)
                STM_ijk += (np.abs(grad_z_WF_for_e)) ** 2

        elif tip_type == 'px':
            STM_ijk = np.zeros((xlen, ylen, zlen))
            for e in e_array:
                grad_x_WF_for_e = np.gradient(self.WF_ijke[:, :, :, e], axis=0)
                STM_ijk += (np.abs(grad_x_WF_for_e)) ** 2

        elif tip_type == 'p30':
            STM_ijk = np.zeros((xlen, ylen, zlen))
            for e in e_array:
                grad_x_WF_for_e = np.gradient(self.WF_ijke[:, :, :, e], axis=0)
                grad_y_WF_for_e = np.gradient(self.WF_ijke[:, :, :, e], axis=1)
                STM_ijk += (np.abs(grad_y_WF_for_e + grad_x_WF_for_e)) ** 2

        elif tip_type == 'p60':
            STM_ijk = np.zeros((xlen, ylen, zlen))
            for e in e_array:
                grad_y_WF_for_e = np.gradient(self.WF_ijke[:, :, :, e], axis=1)
                STM_ijk += (np.abs(grad_y_WF_for_e)) ** 2

        elif tip_type == 'p90':
            STM_ijk = np.zeros((xlen, ylen, zlen))
            for e in e_array:
                grad_x_WF_for_e = np.gradient(self.WF_ijke[:, :, :, e], axis=0)
                grad_y_WF_for_e = np.gradient(self.WF_ijke[:, :, :, e], axis=1)
                STM_ijk += (np.abs(grad_y_WF_for_e - grad_x_WF_for_e / 2.0)) ** 2

        elif tip_type == 'p120':
            STM_ijk = np.zeros((xlen, ylen, zlen))
            for e in e_array:
                grad_x_WF_for_e = np.gradient(self.WF_ijke[:, :, :, e], axis=0)
                grad_y_WF_for_e = np.gradient(self.WF_ijke[:, :, :, e], axis=1)
                STM_ijk += (np.abs(grad_y_WF_for_e - grad_x_WF_for_e)) ** 2

        elif tip_type == 'p150':
            STM_ijk = np.zeros((xlen, ylen, zlen))
            for e in e_array:
                grad_x_WF_for_e = np.gradient(self.WF_ijke[:, :, :, e], axis=0)
                grad_y_WF_for_e = np.gradient(self.WF_ijke[:, :, :, e], axis=1)
                STM_ijk += (np.abs(grad_y_WF_for_e / 2.0 - grad_x_WF_for_e)) ** 2

        elif tip_type == 's+pz':
            STM_ijk = np.zeros((xlen, ylen, zlen))
            for e in e_array:
                grad_z_WF_for_e = np.gradient(self.WF_ijke[:, :, :, e], axis=2)
                STM_ijk += (np.abs(self.WF_ijke[:, :, :, e] + grad_z_WF_for_e)) ** 2

        if not os.path.exists(dir):
            print(f"Directory {dir} does not exist. Creating directory")
            os.mkdir(dir)

        filename = 'stm_'+str(STM_energy_range[0])+'_'+str(STM_energy_range[1])+'_'+tip_type
        if format == 'vasp':
            self.save_as_vasp(STM_ijk, filename, dir)
        elif format == 'cube':
            self.save_as_cube(STM_ijk, filename, dir)


if __name__=="__main__":
    t = time.time()
    stm = STM(path_to_data='Saved_data', poscar_path='../POSCAR')
    stm.load_data()

    ###CALCULATE 2D STM IMAGES###
    #stm.calculate_2D_STM((-0.5, 0.0), 3.5, tip_type='oxygen', cutoff_in_Angstroms=5)
    #stm.calculate_2D_STM((-0.5, 0.0), 3.0, tip_type='s')
    #stm.calculate_2D_STM((-0.5, 0.0), 3.0, tip_type='pz')
    #stm.calculate_2D_STM((-0.5, 0.0), 3.0, tip_type='s+pz')
    #stm.calculate_2D_STM((-0.5, 0.0), 4.5, tip_type='IrCl6', cutoff_in_Angstroms=8)
    #stm.calculate_2D_STM((-0.5, 0.0), 5.5, tip_type='RuNH3_6_NNN_plane', cutoff_in_Angstroms=8)

    ###CALCULATE 2D ECSTM IMAGES###
    stm.load_data_for_ecstm(outcar_path='../OUTCAR', locpot_path='../LOCPOT')
   
    #for V_std in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    #for V_std in [-0.5]:
    #    stm.set_ecstm_parameters(C_EDL=20, T=298, lambda_=0.9, V_std=V_std,
    #                         overpot=0, effective_freq = 1E13, linear_constant=26, threshold_value=1e-5)
    #    stm.plot_distributions(E_range=[-7, 4], dE=0.05, sigma=1, fill_area_lower_Fermi_lvl=True,
    #                plot_Fermi_Dirac_distib=False, plot_redox_distrib=True)
    #    #stm.calculate_2D_ECSTM(z_position=4.5, tip_type='IrCl6', cutoff_in_Angstroms=6)
    #    #stm.calculate_2D_ECSTM(z_position=3.0, tip_type='s')
    #    #stm.calculate_2D_ECSTM(z_position=3.5, tip_type='pz')
    #    stm.calculate_2D_ECSTM(z_position=4.5, tip_type='s')
   
   
    #stm.set_ecstm_parameters(C_EDL=20, T=298, lambda_=1.63, V_std=-0.6,
    #                         overpot=0, effective_freq = 1E13, linear_constant=26, threshold_value=1e-5)
    #stm.plot_distributions(E_range=[-7, 4], dE=0.05, sigma=1, fill_area_lower_Fermi_lvl=True,
    #             plot_Fermi_Dirac_distib=False, plot_redox_distrib=True)
    #stm.calculate_2D_ECSTM(z_position=3.5, tip_type='oxygen', cutoff_in_Angstroms=5)
    
                 
    #stm.set_ecstm_parameters(C_EDL=20, T=298, lambda_=0.8, V_std=0.1,
    #                         overpot=0, effective_freq = 1E13, linear_constant=26, threshold_value=1e-5)
    #stm.plot_distributions(E_range=[-7, 4], dE=0.05, sigma=1, fill_area_lower_Fermi_lvl=True,
    #             plot_Fermi_Dirac_distib=False, plot_redox_distrib=True)
    #stm.calculate_2D_ECSTM(z_position=5.5, tip_type='RuNH3_6_NNN_plane', cutoff_in_Angstroms=8)
    #stm.calculate_2D_ECSTM(z_position=3.0, tip_type='s')
    #stm.calculate_2D_ECSTM(z_position=3.0, tip_type='pz')
    #stm.calculate_2D_ECSTM(z_position=5.0, tip_type='s')
    #stm.calculate_2D_ECSTM(z_position=5.0, tip_type='pz')
    #stm.calculate_2D_ECSTM(z_position=4.5, tip_type='s')
    #stm.calculate_2D_ECSTM(z_position=4.5, tip_type='pz')
    
    stm.set_ecstm_parameters(C_EDL=20, T=298, lambda_=1.2, V_std=0.87,
                             overpot=0, effective_freq = 1E13, linear_constant=26, threshold_value=1e-5)
    stm.plot_distributions(E_range=[-7, 4], dE=0.05, sigma=1, fill_area_lower_Fermi_lvl=True,
                 plot_Fermi_Dirac_distib=False, plot_redox_distrib=True)
    stm.calculate_2D_ECSTM(z_position=5.5, tip_type='IrCl6', cutoff_in_Angstroms=6)
    #stm.calculate_2D_ECSTM(z_position=3.0, tip_type='s')
    #stm.calculate_2D_ECSTM(z_position=3.0, tip_type='pz')
    #stm.calculate_2D_ECSTM(z_position=5.0, tip_type='s')
    #stm.calculate_2D_ECSTM(z_position=5.0, tip_type='pz')
    #stm.calculate_2D_ECSTM(z_position=4.5, tip_type='s')
    #stm.calculate_2D_ECSTM(z_position=4.5, tip_type='pz')             
    
    #stm.set_ecstm_parameters(C_EDL=20, T=298, lambda_=0.9, V_std=0.0,
    #                         overpot=0, effective_freq = 1E13, linear_constant=26, threshold_value=1e-5)
    #stm.plot_distributions(E_range=[-7, 4], dE=0.05, sigma=1, fill_area_lower_Fermi_lvl=True,
    #             plot_Fermi_Dirac_distib=False, plot_redox_distrib=True)
    #stm.calculate_2D_ECSTM(z_position=3.0, tip_type='oxygen', cutoff_in_Angstroms=5)
                 
                 
    #stm.set_ecstm_parameters(C_EDL=20, T=298, lambda_= 1.0, V_std=-0.5,
    #                         overpot=0, effective_freq = 1E13, linear_constant=26, threshold_value=1e-5)
    #stm.plot_distributions(E_range=[-7, 4], dE=0.05, sigma=1, fill_area_lower_Fermi_lvl=True,
    #             plot_Fermi_Dirac_distib=False, plot_redox_distrib=True)
    #stm.calculate_2D_ECSTM(z_position=3.0, tip_type='oxygen', cutoff_in_Angstroms=5)
    #stm.calculate_2D_ECSTM(z_position=5, tip_type='RuNH3_6_NNN_plane', cutoff_in_Angstroms=8, all_z=True)
    #stm.calculate_2D_ECSTM(z_position=7, tip_type='RuNH3_6_NNN_plane', cutoff_in_Angstroms=8, all_z=True)

    
    #stm.calculate_2D_ECSTM(z_position=4.5, tip_type='oxygen', cutoff_in_Angstroms=5)
    #stm.calculate_2D_ECSTM(z_position=3.5, tip_type='oxygen', cutoff_in_Angstroms=5)

    ##CACLULATE 3D STM IMAGE OPENABLE WITH VESTA PACKAGE###
    #stm.calculate_3D_STM((-0.5, 0.0), tip_type='s', format='cube')
    #stm.calculate_3D_STM((0.0, 0.5), tip_type='s', format='cube')


    ###CHECK OXYGEN ORBITAL GENERATION###
    #acc_orbitals = stm.generate_acceptor_orbitals('RuNH3_6_NNN_plane', np.shape(stm.WF_ijke)[0:3], z_shift = 20, x_shift=50, y_shift=50)
    #for i, orbitals in enumerate(acc_orbitals):
    #    stm.save_as_cube(orbitals, 'RuNH3_NNN'+str(i), 'orbitals')
    #acc_orbitals = stm.generate_acceptor_orbitals('RuNH3_6', np.shape(stm.WF_ijke)[0:3], z_shift = 20, x_shift=50, y_shift=50)
    #for i, orbitals in enumerate(acc_orbitals):
    #    stm.save_as_cube(orbitals, 'RuNH3_standart_orient'+str(i), 'orbitals')
    #acc_orbitals = stm.generate_acceptor_orbitals('RuNH3_6_perpendicular', np.shape(stm.WF_ijke)[0:3], z_shift = 20, x_shift=50, y_shift=50)
    #for i, orbitals in enumerate(acc_orbitals):
    #    stm.save_as_cube(orbitals, 'RuNH3_perpendicular'+str(i), 'orbitals')        
    #acc_orbitals = stm.generate_acceptor_orbitals('oxygen_parallel_y', np.shape(stm.WF_ijke)[0:3], z_shift = 20, x_shift=50, y_shift=50)
    #for i, orbitals in enumerate(acc_orbitals):
    #    stm.save_as_vasp(orbitals, 'ox_par_y_orb_'+str(i), 'orbitals')
    
    #print(f"Job done! It takes: {time.time()-t} sec")
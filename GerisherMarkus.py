import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt


class GM:
    """This class calculates the final Fermi and Redox species distributions according
    to the Gerisher-Marcus formalism.

    Parameters:
    -----------
    DOS: np.ndarray, optional
        The values of DOS in 1D numpy array. If not specified values will be taken from saved data.

    E: np.ndarray, optional
        The corresponding to the DOS energy mesh. If not specified values will be taken from saved data.

    efermi: np.ndarray. optional
        System Fermi level. If not specified values will be taken from saved data.

    vacuum_lvl: np.ndarray, optional
        System vacuum level. If not specified values will be taken from saved data.
    """
    def __init__(self, DOS=None, E=None, efermi=None, vacuum_lvl=None):
        self.C_EDL = None
        self.T = None
        self.l = None
        self.sheet_area = None

        self.E = None
        self.DOS = None

        self.sigma_eq = None
        self.dE_Q_eq = None
        self.y_fermi = None
        self.y_redox = None

        if DOS is None:
            try:
                self.DOS = np.load('Saved_data/DOS.npy')
            except OSError:
                print('File DOS.npy does not exist')

        if E is None:
            try:
                self.E = np.load('Saved_data/E.npy')
            except OSError:
                print('File E_DOS.npy does not exist')

        if efermi is None:
            try:
                self.efermi = np.load('Saved_data/efermi.npy')
            except OSError:
                print('File efermi.npy does not exist')

        if vacuum_lvl is None:
            try:
                self.vacuum_lvl = np.load('Saved_data/vacuum_lvl.npy')
            except OSError:
                print('File vacuum_lvl.npy does not exist')

    def set_params(self, C_EDL, T, l, sheet_area):
        """Set parameters of calculation

        Parameters:
        ----------
        C_EDL: float
            Capacitance of electric double layer (microF/cm^2)

        T: int, float
            Temperature. It is used in computing Fermi function and distribution function of redox system states

        l: float
            Reorganization energy in eV
        """
        self.C_EDL = C_EDL
        self.T = T
        self.l = l

        self.sheet_area = sheet_area

    def save(self, variable='all', dir='Saved_data', postfix=''):
        """
        This function saves variables in .npy files
        :param variable: desired variable
        :param dir: directory name to save file
        :param postfix: postfix for saved files
        :return: nothing
        """
        if variable == 'all':
            for var in ['y_fermi', 'y_redox']:
                if self.var is not None:
                    self.save(var, postfix=postfix)
        else:
            if not os.path.exists(dir):
                print('Directory ', dir, ' does not exist. Creating directory')
                os.mkdir(dir)
            np.save(dir + '/' + variable + postfix + '.npy', getattr(self, variable))
            print('Variable ', variable, ' saved to directory ', dir)

    def load(self, variable='all', dir='Saved_data', postfix=''):
        """
        This function loads variables from .npy files to class variables
        :param variable: desired variable
        :param dir: directory from which load files
        :param postfix: postfix for loading files
        :return: nothing
        """
        if variable == 'all':
            for var in ['y_fermi', 'y_redox']:
                self.load(var, postfix=postfix)
        else:
            setattr(self, variable, np.load(dir + '/' + str(variable) + postfix + '.npy'))
            print('Variable ', variable, ' loaded')

    def check_param(self, variable):
        """
        This function checks whether desires variable is not None and if necessary load it from file or process VASP data
        :param variable: desired variable
        :return: nothing
        """
        if getattr(self, variable) is None:
            raise ValueError(f'{variable} is not defined')

    @staticmethod
    def nearest_array_indices(array, value):
        i = 0
        while value < array[i]:
            i += 1
        return i - 1, i

    @staticmethod
    def fermi_func(E, T):
        k = 8.617e-5  # eV/K
        return 1 / (1 + np.exp(E / (k * T)))

    @staticmethod
    def W_ox(E, T, l):
        k = 8.617e-5  # eV/K
        W_0 = (1 / np.sqrt(4 * k * T * l))
        return W_0 * np.exp(- (E - l) ** 2 / (4 * k * T * l))

    @staticmethod
    def W_red(E, T, l):
        k = 8.617e-5  # eV/K
        W_0 = (1 / np.sqrt(4 * k * T * l))
        return W_0 * np.exp(- (E + l) ** 2 / (4 * k * T * l))

    def compute_C_quantum(self, dE_Q_arr):

        self.check_param('T')
        self.check_param('sheet_area')

        k = 8.617e-5  # eV/K

        elementary_charge = 1.6e-19  # C
        k_1 = 1.38e-23  # J/K
        const = (1e6 * elementary_charge ** 2) / (4 * k_1 * self.sheet_area)  # micro F * K / cm^2

        if type(dE_Q_arr) is np.ndarray:

            C_q_arr = []

            for dE_Q in dE_Q_arr:
                E_2 = self.E - dE_Q  # energy range for cosh function
                integrand = (self.DOS / np.cosh(E_2 / (2 * k * self.T))) / np.cosh(E_2 / (2 * k * self.T))
                C_q = (const / self.T) * integrate.simps(integrand, self.E)
                C_q_arr.append(C_q)

            return C_q_arr

    def compute_sigma_EDL(self, dE_EDL):

        self.check_param('C_EDL')
        return self.C_EDL * dE_EDL

    def compute_sigma_quantum(self, dE_Q_arr):
        """Compute surface charge density induced by depletion or excess of electrons

        Parameters:
        ----------
        dE_Q_arr: np.ndarray, float
            Shift in Fermi level due to quantum capacitance

        Returns:
        -------
        sigmas: np.ndarray, float
            Computed values (or one value) of surface charge densities
        """

        self.check_param('T')
        self.check_param('sheet_area')

        elementary_charge = 1.6e-13  # micro coulomb

        if type(dE_Q_arr) is np.ndarray:
            y_fermi = self.fermi_func(self.E, self.T)

            sigmas = []

            for dE_Q in dE_Q_arr:
                E_2 = self.E - dE_Q  # energy range for shifted Fermi_Dirac function
                y_fermi_shifted = self.fermi_func(E_2, self.T)
                integrand = self.DOS * (y_fermi - y_fermi_shifted)
                sigma = (elementary_charge / self.sheet_area) * integrate.simps(integrand, self.E)
                sigmas.append(sigma)

            return sigmas

        elif type(dE_Q_arr) is float or type(dE_Q_arr) is np.float64:
            y_fermi = self.fermi_func(self.E, self.T)

            E_2 = self.E - dE_Q_arr  # energy range for shifted Fermi_Dirac function
            y_fermi_shifted = self.fermi_func(E_2, self.T)
            integrand = self.DOS * (y_fermi - y_fermi_shifted)
            sigma = (elementary_charge / self.sheet_area) * integrate.simps(integrand, self.E)

            return sigma
        else:
            raise TypeError(f'Invalid type of dE_Q_arr: {type(dE_Q_arr)}')

    def compute_distributions(self, V_std, reverse=False, overpot=0, SIGMA_0=0.1, ACCURACY_SIGMA=1e-3, SIGMA_RANGE=4):

        """Function computes Fermi-Dirac and Redox species distributions according to Gerisher-Markus formalism

        Parameters:
        ----------
        V_std: float
            Standard potential of a redox couple (Volts)

        reverse: bool, optional
            if reverse is False the process of electron transfer from electrode to the oxidized state of the
            redox particles is considered and vice versa

        overpot: float, optional
            Overpotential (Volts). It shifts the electrode Fermi energy to -|e|*overpot

        SIGMA_0: float, optional
            Initial guess for charge at equilibrium

        ACCURACY_SIGMA: float, optional
            Accuracy of charge calculation

        SIGMA_RANGE: float, optional
            It defines the minimum and maximum calculated charge
        """
        def error_E_diff(sigma, E_diff, sigma_Q_arr):
            i_1, i_2 = self.nearest_array_indices(sigma_Q_arr, sigma)
            dE_Q = E_start + E_step * i_1
            dE_EDL = - sigma / self.C_EDL
            dE_total = dE_Q + dE_EDL

            return (dE_total - E_diff) ** 2

        for var in ['T', 'l', 'C_EDL']:
            self.check_param(var)

        E_step = ACCURACY_SIGMA
        E_start = - SIGMA_RANGE
        E_range = np.arange(E_start, -E_start, E_step)
        sigma_Q_arr = self.compute_sigma_quantum(E_range)

        sigma_0 = SIGMA_0

        E_F_redox = -4.5 - self.efermi - V_std + self.vacuum_lvl - overpot

        # compute equilibrium case
        result = minimize(error_E_diff, np.array([sigma_0]), args=(E_F_redox, sigma_Q_arr))
        sigma_eq = result.x[0]

        self.sigma_eq = sigma_eq

        i_1, i_2 = self.nearest_array_indices(sigma_Q_arr, sigma_eq)
        dE_Q_eq = E_start + E_step * i_1

        self.dE_Q_eq = dE_Q_eq

        E_fermi = self.E - dE_Q_eq
        E_DOS_redox = self.E - dE_Q_eq - overpot

        if reverse:
            y_fermi = 1 - self.fermi_func(E_fermi, self.T)
            y_redox = self.W_red(E_DOS_redox, self.T, self.l)
        else:
            y_fermi = self.fermi_func(E_fermi, self.T)
            y_redox = self.W_ox(E_DOS_redox, self.T, self.l)

        self.y_fermi = y_fermi
        self.y_redox = y_redox

        return y_fermi, y_redox

    def compute_k_HET(self, V_std_pot_arr, overpot_arr, reverse=False):
        """Compute integral k_HET using Geresher-Markus formalism

        Parameters:
        ----------
        V_std_pot_arr: float, np.ndarray
            A range of varying a standard potential
        overpot_arr: float, np.ndarray
            A range of varying an overpotential
        reverse: bool, optional
            if reverse is False the process of electron transfer from electrode to the oxidized state of the
            redox particles is considered and vice versa

        Returns:
        -------
        k_HET: np.ndarray
            Calculated heterogeneous electron transfer rate constant according to Gerischer-Marcus model
        """

        if type(V_std_pot_arr) is np.ndarray:
            if type(overpot_arr) is np.float64 or type(overpot_arr) is float:

                k_HET = np.zeros_like(V_std_pot_arr)

                for i, V_std in enumerate(V_std_pot_arr):
                    print(i)
                    y_fermi, y_redox = self.compute_distributions(V_std, reverse=reverse, overpot=overpot_arr)
                    integrand = self.DOS * y_fermi * y_redox
                    k_HET[i] = integrate.simps(integrand, self.E)

                return k_HET

        elif type(overpot_arr) is np.ndarray:
            if type(V_std_pot_arr) is np.float64 or type(V_std_pot_arr) is float:

                k_HET = np.zeros_like(overpot_arr)

                for i, overpot in enumerate(overpot_arr):
                    print(i)
                    y_fermi, y_redox = self.compute_distributions(V_std_pot_arr, reverse=reverse, overpot=overpot)
                    integrand = self.DOS * y_fermi * y_redox
                    k_HET[i] = integrate.simps(integrand, self.E)

                return k_HET

if __name__ == '__main__':
    def nearest_array_indices(array, value):
        i = 0
        while value > array[i]:
            i += 1
        return i - 1, i
    a = GM()
    a.set_params(10, 298, 0.9, 5.265949070860207e-14)
    #x = np.arange(-2, 2, 0.2)
    #k_HET = a.compute_k_HET(0.36, x, reverse=False)
    #k_HET_reverse = a.compute_k_HET(0.36, x, reverse=True)
    #plt.plot(x, k_HET)
    #plt.plot(x, k_HET_reverse)
    y_fermi, y_redox = a.compute_distributions(0.36, overpot=4)
    print(a.sigma_eq)
    n_1, n_2 = nearest_array_indices(a.E, a.dE_Q_eq)
    print(n_1, n_2, a.dE_Q_eq)
    print(integrate.simps(a.DOS[:400], a.E[:400]))
    plt.fill_between(a.E[:n_2], a.DOS[:n_2])
    plt.plot(a.E, a.DOS)
    plt.plot(a.E, y_redox * 10)
    plt.plot(a.E, y_fermi * 30)
    plt.xlim([-8, 5])
    plt.show()

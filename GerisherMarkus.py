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

        self.E = None
        self.DOS = None

        self.sigma_eq = None
        elf.dE_Q_eq = None
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

    def compute_sigma_EDL(self, dE_EDL):

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

    def compute_distributions(self, V_std, reverse=False, dE_eta=0, SIGMA_0=0.1, ACCURACY_SIGMA=1e-3, SIGMA_RANGE=5):
        """Function computes Fermi-Dirac and Redox species distributions according to Gerisher-Markus formalism

        Parameters:
        ----------
        V_std: float
            Standard potential of a redox pair (Volts)

        dE_eta: float
            Shift of the electrode Fermi energy due to the overpotential (eV)

        reverse: bool, optional
            if reverse is False the process of electron transfer from electrode to the oxidized state of the
            redox particles is considered and vice versa
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

        E_F_redox = -4.5 - self.efermi - V_std + self.vacuum_lvl

        # compute equilibrium case
        result = minimize(error_E_diff, np.array([sigma_0]), args=(E_F_redox, sigma_Q_arr))
        sigma_eq = result.x[0]

        self.sigma_eq = sigma_eq

        i_1, i_2 = self.nearest_array_indices(sigma_Q_arr, sigma_eq)
        dE_Q_eq = E_start + E_step * i_1

        self.dE_Q_eq = dE_Q_eq

        # compute the case with nonzero overpotential
        if dE_eta != 0:
            pass
        else:
            E_fermi = self.E - dE_Q_eq
            E_DOS_redox = self.E - dE_Q_eq

        if reverse:
            y_fermi = 1 - self.fermi_func(E_fermi, self.T)
            y_redox = self.W_red(E_DOS_redox, self.T, self.l)
        else:
            y_fermi = self.fermi_func(E_fermi, self.T)
            y_redox = self.W_ox(E_DOS_redox, self.T, self.l)

        self.y_fermi = y_fermi
        self.y_redox = y_redox

        return y_fermi, y_redox

    '''def compute_k_HET(self, mode, V_std_pot_arr, dE_eta_arr, n_jobs=1):
        """Compute integral k_HET using Geresher-Markus formalism

        Parameters:
        ----------
        mode: string
            Mode of computing that defines which parameters are fixed and which are varied

            Possible values:
                'fixed_overpot' - overpotential is fixed, Standard redox potential might be varied.
                The process of electron transfer from electrode to redox particle is observed.

                'fixed_overpot_reversed' - overpotential is fixed, Standard redox potential might be varied.
                The process of electron transfer from redox particle to electrode is observed.

                'fixed_std_pot' - standard redox potential is fixed, overpotential might be varied.
                The process of electron transfer from electrode to redox particle is observed.

                'fixed_std_pot_reversed' - standard redox potential is fixed, overpotential might be varied.
                The process of electron transfer from redox particle to electrode is observed.
        V_std_pot_arr: float, np.ndarray
            A range of varying a standard potential
        dE_eta_arr: float, np.ndarray
            A range of varying an overpotential
        n_jobs: int, optional
            Define the number of threads.
            -1 means that all available cores will be used

        Returns:
        -------
        df: pd.DataFrame
            Pandas DataFrame that contain columns:
                E_std_pot - Standard Redox Potential

                dE_eta - Overpotential

                E_F_redox - Difference between Fermi energy of redox couple and electrode.
                Fermi level of electrode is zero.
                The sum of dE_Q_eq and dE_EDL_eq must be equal to E_F_redox

                efermi - Fermi Energy

                vacuul_lvl - Vacuum Level

                dE_Q_eq - Displacement of Fermi energy due to quantum capacitance in
                equilibrium case (overpotential = 0). Countered from Fermi level of
                electrode

                dE_EDL_eq - Difference between E_F_redox and Fermi level of electrode
                plus dE_Q_eq in equilibrium case

                sigma_eq - Charge density in equilibrium case.

                dE_Q_overpot - ^^^ since overpotential is not equal to zero

                dE_EDL_overpot - ^^^ since overpotential is not equal to zero

                sigma_overpot - ^^^ since overpotential is not equal to zero

                k_HET - constant of heterogeneous electron transfer
        """

        if n_jobs == -1:
            n_jobs = int(os.environ["NUMBER_OF_PROCESSORS"])

        df = pd.DataFrame()

        if mode == 'fixed_overpot':
            reverse = False
            arrays_len = len(V_std_pot_arr)
            thread_func = self.fixed_overpot_thread
            df['V_std_pot'] = V_std_pot_arr
            df['dE_eta'] = [dE_eta_arr] * len(V_std_pot_arr)
        elif mode == 'fixed_std_pot':
            reverse = False
            arrays_len = len(dE_eta_arr)
            thread_func = self.fixed_std_pot_thread
            df['dE_eta'] = dE_eta_arr
            df['V_std_pot'] = [V_std_pot_arr] * len(dE_eta_arr)
        elif mode == 'fixed_overpot_reversed':
            reverse = True
            arrays_len = len(V_std_pot_arr)
            thread_func = self.fixed_overpot_thread
            df['V_std_pot'] = V_std_pot_arr
            df['dE_eta'] = [dE_eta_arr] * len(V_std_pot_arr)
        elif mode == 'fixed_std_pot_reversed':
            reverse = True
            arrays_len = len(dE_eta_arr)
            thread_func = self.fixed_std_pot_thread
            df['dE_eta'] = dE_eta_arr
            df['V_std_pot'] = [V_std_pot_arr] * len(dE_eta_arr)
        else:
            raise ValueError('mode has unsupported value')

        # Init arrays with shared memory
        E_F_redox = mp.Array('f', arrays_len)
        dE_Q_eq = mp.Array('f', arrays_len)
        dE_EDL_eq = mp.Array('f', arrays_len)
        sigma_eq = mp.Array('f', arrays_len)
        dE_Q_overpot = mp.Array('f', arrays_len)
        dE_EDL_overpot = mp.Array('f', arrays_len)
        sigma_overpot = mp.Array('f', arrays_len)
        k_HET = mp.Array('f', arrays_len)
        shared_arrays = [E_F_redox, dE_Q_eq, dE_EDL_eq, sigma_eq,
                         dE_Q_overpot, dE_EDL_overpot, sigma_overpot, k_HET]

        # Distribute tasks over n_jobs
        start, end = self.split_array_into_nproc(arrays_len, n_jobs)
        processes = []
        for Id in range(n_jobs):
            i_1 = start[Id]
            i_2 = end[Id]
            p = mp.Process(target=thread_func, args=(Id, i_1, i_2, V_std_pot_arr, dE_eta_arr,
                                                     self.T, self.l, reverse, *shared_arrays))
            processes.append(p)

        # Start all threads and wait while they are finished
        for proc in processes:
            proc.start()
        for proc in processes:
            proc.join()

        df['E_F_redox'] = E_F_redox[:]
        df['efermi'] = [self.efermi] * arrays_len
        df['vacuul_lvl'] = [self.vacuum_lvl] * arrays_len
        df['dE_Q_eq'] = dE_Q_eq[:]
        df['dE_EDL_eq'] = dE_EDL_eq[:]
        df['sigma_eq'] = sigma_eq[:]
        df['dE_Q_overpot'] = dE_Q_overpot[:]
        df['dE_EDL_overpot'] = dE_EDL_overpot[:]
        df['sigma_overpot'] = sigma_overpot[:]
        df['k_HET'] = k_HET[:]

        return df'''


if __name__ == '__main__':
    def nearest_array_indices(array, value):
        i = 0
        while value > array[i]:
            i += 1
        return i - 1, i
    a = GM()
    a.set_params(10, 298, 0.9, 5.265949070860207e-14)
    y_fermi, y_redox = a.compute_distributions(-1)
    n_1, n_2 = nearest_array_indices(a.E, a.dE_Q_eq)
    #print(n_1, n_2, a.dE_Q_eq)
    #print(integrate.simps(a.DOS[:400], a.E[:400]))
    plt.fill_between(a.E[:n_2], a.DOS[:n_2])
    plt.plot(a.E, a.DOS)
    plt.plot(a.E, y_redox * 10)
    plt.plot(a.E, y_fermi * 30)
    plt.xlim([-8, 5])
    plt.show()

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize


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

        self.E_DOS = None
        self.y_fermi = None
        self.y_redox = None

        if DOS is None:
            try:
                self.DOS = np.load('Saved_data/DOS.npy').item()
            except OSError:
                print('File DOS.npy does not exist')

        if E is None:
            try:
                self.E = np.load('Saved_data/E.npy').item()
            except OSError:
                print('File E_DOS.npy does not exist')

        if efermi is None:
            try:
                self.efermi = np.load('Saved_data/efermi.npy').item()
            except OSError:
                print('File efermi.npy does not exist')

        if vacuum_lvl is None:
            try:
                self.vacuum_lvl = np.load('Saved_data/vacuum_lvl.npy').item()
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

    @staticmethod
    def nearest_array_indices(array, value):
        i = 0
        while value < array[i]:
            i += 1
        return i - 1, i

    @staticmethod
    def compute_sigma_EDL(C_EDL, dE_EDL):

        return C_EDL * dE_EDL

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

    def compute_sigma_quantum(self, dE_Q_arr, T, E_DOS, DOS, efermi):
        """Compute surface charge density induced by depletion or excess of electrons

        Parameters:
        ----------
        dE_Q_arr: np.ndarray, float
            Shift in Fermi level due to quantum capacitance

        T: int, float
            Temperature in K

        E_DOS: np.ndarray
            1D numpy array that contains all energy points according to DOS points

        DOS: np.ndarray
            1D numpy array that contains all electron densities according to E_DOS values

        efermi: float
            Fermi energy in eV

        Returns:
        -------
        sigmas: np.ndarray, float
            Computed values (or one value) of surface charge densities
        """
        elementary_charge = 1.6e-13  # micro coulomb

        if type(dE_Q_arr) is np.ndarray:
            E_1 = E_DOS - efermi  # energy range for DOS and not shifted Fermi-Dirac function
            y_fermi = self.fermi_func(E_1, T)

            sigmas = []

            for dE_Q in dE_Q_arr:
                E_2 = E_DOS - efermi - dE_Q  # energy range for shifted Fermi_Dirac function
                y_fermi_shifted = self.fermi_func(E_2, T)
                integrand = DOS * (y_fermi - y_fermi_shifted)
                sigma = (elementary_charge / self.sheet_area) * integrate.simps(integrand, E_1)
                sigmas.append(sigma)

            return sigmas

        elif type(dE_Q_arr) is float or type(dE_Q_arr) is np.float64:
            E_1 = E_DOS - efermi  # energy range for DOS and not shifted Fermi-Dirac function
            y_fermi = self.fermi_func(E_1, T)

            E_2 = E_DOS - efermi - dE_Q_arr  # energy range for shifted Fermi_Dirac function
            y_fermi_shifted = self.fermi_func(E_2, T)
            integrand = DOS * (y_fermi - y_fermi_shifted)
            sigma = (elementary_charge / self.sheet_area) * integrate.simps(integrand, E_1)

            return sigma
        else:
            raise TypeError(f'Invalid type of dE_Q_arr: {type(dE_Q_arr)}')

    def compute_distributions(self, V_std, dE_eta, reverse=False, SIGMA_0=0.1, ACCURACY_SIGMA=1e-3, SIGMA_RANGE=11):
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

        E_step = ACCURACY_SIGMA
        E_start = - SIGMA_RANGE
        E_range = np.arange(E_start, -E_start, E_step)
        sigma_Q_arr = self.compute_sigma_quantum(E_range, self.T, self.E, self.DOS, self.efermi)
        sigma_EDL_arr = self.compute_sigma_EDL(self.C_EDL, E_range)

        sigma_0 = SIGMA_0

        E_F_redox = -4.5 - V_std - self.efermi + self.vacuum_lvl

        # compute equilibrium case
        result = minimize(error_E_diff, np.array([sigma_0]), args=(E_F_redox, sigma_Q_arr))
        sigma_eq = result.x[0]

        i_1, i_2 = self.nearest_array_indices(sigma_Q_arr, sigma_eq)
        dE_Q_eq = E_start + E_step * i_1

        i_1, i_2 = self.nearest_array_indices(sigma_EDL_arr, sigma_eq)
        dE_EDL_eq = E_start + E_step * i_1

        # compute the case with nonzero overpotential
        if dE_eta != 0:

            dE_Q_overpot = dE_Q_eq + dE_eta
            sigma_overpot = self.compute_sigma_quantum(dE_Q_overpot, self.T, self.E,
                                                       self.DOS, self.efermi)
            i_1, i_2 = self.nearest_array_indices(sigma_EDL_arr, sigma_overpot)
            dE_EDl_overpot = E_start + E_step * i_1
            E_redox_couple = E_F_redox - dE_EDl_overpot
            E_DOS = self.E - self.efermi
            E_fermi = self.E - self.efermi - dE_Q_overpot
            E_DOS_redox = self.E - self.efermi - E_redox_couple
        else:
            E_DOS = self.E - self.efermi
            E_fermi = self.E - self.efermi - dE_Q_eq
            E_DOS_redox = self.E - self.efermi - dE_Q_eq

        if reverse:
            y_fermi = 1 - self.fermi_func(E_fermi, self.T)
            y_redox = self.W_red(E_DOS_redox, self.T, self.l)
        else:
            y_fermi = self.fermi_func(E_fermi, self.T)
            y_redox = self.W_ox(E_DOS_redox, self.T, self.l)

        self.E_DOS = E_DOS
        self.y_fermi = y_redox
        self.y_redox = y_redox

        return E_DOS, y_fermi, y_redox

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
    a = GM()
    #a.compute_sigma_quantum(np.arange(-1, 1, 0.1), 298, a.E_DOS, a.DOS, a.efermi)
    a.efermi
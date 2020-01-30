import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize
import numbers
import typing
from tqdm import tqdm
from core.useful_funcs import nearest_array_indices, ClassMethods


class GM(ClassMethods):
    """This class calculates the final Fermi and Redox species distributions according
    to the Gerischer-Marcus formalism.

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
        # variables that might be defined through __init__ function
        self.E = E
        self.DOS = DOS
        self.efermi = efermi
        self.vacuum_lvl = vacuum_lvl

        # variables that should be defined through set_params function
        self.C_EDL = None
        self.T = None
        self.l = None
        self.sheet_area = None

        # variables that will be created during calculations
        self.sigma_Q_arr = None

        # variable that define numerical parameters of quantum charge calculation
        self.__SIGMA_0 = 0.5
        self.__SIGMA_ACCURACY = 1e-3
        self.__SIGMA_RANGE = 4

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
        """Sets parameters of calculation

        Parameters:
        ----------
        C_EDL: float, str
            float: Capacitance of electric double layer (microF/cm^2)
            str: 'Q' calculating in the Quantum Capacitance Dominating limit (C_Q << C_EDL)
            str: 'Cl' calculating in the Classical limit (C_Q >> C_EDL)

        T: int, float
            Temperature. It is used in computing Fermi function and distribution function of redox system states

        l: float
            Reorganization energy in eV
        """
        self.C_EDL = C_EDL
        self.T = T
        self.l = l

        self.sheet_area = sheet_area

    def set_params_advance(self, SIGMA_0=0.5, ACCURACY_SIGMA=1e-3, SIGMA_RANGE=4):
        """
        Sets numerical parameters that are used in quantum charge density calculations. Delete cashed
        results of charge calculations.
        Args:
            SIGMA_0: float, optional
                Initial guess for charge at equilibrium
            ACCURACY_SIGMA: float, optional
                Accuracy of charge calculation
            SIGMA_RANGE: float, optional
                It defines the minimum and maximum calculated charge
        """
        self.__SIGMA_0 = SIGMA_0
        self.__SIGMA_ACCURACY = ACCURACY_SIGMA
        self.__SIGMA_RANGE = SIGMA_RANGE
        self.sigma_Q_arr = None

    @staticmethod
    def fermi_func(E, T):
        """
        Calculates Fermi-Dirac Distribution
        Args:
            E: Energies
            T: Temperature in K
        """
        k = 8.617e-5  # eV/K
        return 1 / (1 + np.exp(E / (k * T)))

    @staticmethod
    def W_ox(E, T, l):
        """
        Distribution of oxidized states
        Args:
            E (np.array): Energies
            T (float): Temperature
            l (float): Reorganization energy
        """
        k = 8.617e-5  # eV/K
        W_0 = (1 / np.sqrt(4 * k * T * l))
        return W_0 * np.exp(- (E - l) ** 2 / (4 * k * T * l))

    @staticmethod
    def W_red(E, T, l):
        """
        Distribution of reduced states
        Args:
            E (np.array): Energies
            T (float): Temperature
            l (float): Reorganization energy
        """
        k = 8.617e-5  # eV/K
        W_0 = (1 / np.sqrt(4 * k * T * l))
        return W_0 * np.exp(- (E + l) ** 2 / (4 * k * T * l))

    def compute_C_quantum(self, dE_Q_arr):
        """
        Calculates differential quantum capacitance
        Q = e * int{DOS(E) * [f(E) - f(E + deltaE)] dE}
        C_Q = - dQ/d(deltaE) = - (e / (4*k*T)) *  int{DOS(E) * sech^2[(E+deltaE)/(2*k*T)] dE}
        Args:
            dE_Q_arr (np.array, float): Energy shift at which C_Q is calculated
        Returns:
            Quantum capacitance in accordance with energy displacement(s)
        TODO check constants
        """
        self.check_existence('T')
        self.check_existence('sheet_area')

        k = 8.617e-5  # eV/K

        elementary_charge = 1.6e-19  # C
        k_1 = 1.38e-23  # J/K
        const = - (1e6 * elementary_charge ** 2) / (4 * k_1 * self.sheet_area)  # micro F * K / cm^2

        if isinstance(dE_Q_arr, typing.Iterable):

            C_q_arr = np.zeros_like(dE_Q_arr)

            for i, dE_Q in enumerate(dE_Q_arr):
                E_2 = self.E - dE_Q  # energy range for cosh function
                cosh = np.cosh(E_2 / (2 * k * self.T))
                integrand = (self.DOS / cosh) / cosh
                C_q = (const / self.T) * integrate.simps(integrand, self.E)
                C_q_arr[i] = C_q

            return C_q_arr

    def compute_sigma_EDL(self, dE_EDL):
        """
        Calculates charge corresponding to the potential drop of -dE_EDL/|e|.
        Takes into account integral capacitance C_EDL
        Args:
            dE_EDL (float, np.array): Electron energy shift due to potential drop
        Returns:
            Charge or Sequence of charges
        """
        self.check_existence('C_EDL')
        return - self.C_EDL * dE_EDL

    def compute_sigma_quantum(self, dE_Q_arr):
        """
        Computes surface charge density induced by depletion or excess of electrons

        Parameters:
        ----------
        dE_Q_arr: np.ndarray, float
            Shift in Fermi level due to quantum capacitance

        Returns:
        -------
        sigmas: np.ndarray, float
            Computed values (or one value) of surface charge densities
        """

        self.check_existence('T')
        self.check_existence('sheet_area')

        elementary_charge = 1.6e-13  # micro coulomb

        if isinstance(dE_Q_arr, typing.Iterable):
            y_fermi = self.fermi_func(self.E, self.T)

            sigmas = []

            for dE_Q in dE_Q_arr:
                E_2 = self.E - dE_Q  # energy range for shifted Fermi_Dirac function
                y_fermi_shifted = self.fermi_func(E_2, self.T)
                integrand = self.DOS * (y_fermi - y_fermi_shifted)
                sigma = (elementary_charge / self.sheet_area) * integrate.simps(integrand, self.E)
                sigmas.append(sigma)

            return sigmas

        elif isinstance(dE_Q_arr, numbers.Real):
            y_fermi = self.fermi_func(self.E, self.T)

            E_2 = self.E - dE_Q_arr  # energy range for shifted Fermi_Dirac function
            y_fermi_shifted = self.fermi_func(E_2, self.T)
            integrand = self.DOS * (y_fermi - y_fermi_shifted)
            sigma = (elementary_charge / self.sheet_area) * integrate.simps(integrand, self.E)

            return sigma
        else:
            raise TypeError(f'Invalid type of dE_Q_arr: {type(dE_Q_arr)}')

    def compute_distributions(self, V_std, overpot=0, reverse=False, add_info=False):
        """Computes Fermi-Dirac and Redox species distributions according to Gerischer-Markus formalism
        with Quantum Capacitance

        Parameters:
        ----------
        V_std: float
            Standard potential of a redox couple (Volts)
        overpot: float, optional
            Overpotential (Volts). It shifts the electrode Fermi energy to -|e|*overpot
        reverse: bool, optional
            If reverse is False the process of electron transfer from electrode to the oxidized state of the
            redox species is considered and vice versa
        add_info: bool, optional
            If False the func returns Fermi-Dirac and Redox species distributions
            If True additionally returns dE_Q (Fermi energy shift due to the quantum capacitance),
            sigma (surface charge) and E_diff (the whole energy shift with respect to the original Fermi level)

        Returns:
        -------
        y_fermi: np.array
            Fermi-Dirac distribution
        y_redox: np.array
            Redox species distributions
        dE_Q: np.array, optional (if add_info == True)
            Total shift of the Fermi energy due to the Quantum Capacitance
        sigma: np.array, optional (if add_info == True)
            surface charge in microF/cm^2
        E_F_redox: np.array, optional (if add_info == True)
            The sum of two energy displacement of the electrode due to the difference in Fermi level of Redox couple
            and the electrode and overpotential. It splits into dE_Q and dE_EDL
        """
        def error_E_diff(sigma, E_diff, sigma_Q_arr):
            i_1, i_2 = nearest_array_indices(sigma_Q_arr, sigma)
            dE_Q = E_start + E_step * i_1
            dE_EDL = - sigma / self.C_EDL
            dE_total = dE_Q + dE_EDL

            return (dE_total - E_diff) ** 2

        for var in ['T', 'l', 'C_EDL']:
            self.check_existence(var)

        # check if we've already calculated sigma_Q_arr in another run
        if self.sigma_Q_arr is None:
            E_step = self.__SIGMA_ACCURACY
            E_start = - self.__SIGMA_RANGE
            E_range = np.arange(E_start, -E_start, E_step)
            sigma_Q_arr = self.compute_sigma_quantum(E_range)
            sigma_0 = self.__SIGMA_0
            self.sigma_Q_arr = sigma_Q_arr
        else:
            E_step = self.__SIGMA_ACCURACY
            E_start = - self.__SIGMA_RANGE
            sigma_0 = self.__SIGMA_0
            sigma_Q_arr = self.sigma_Q_arr

        E_F_redox = -4.5 - self.efermi - V_std + self.vacuum_lvl - overpot

        result = minimize(error_E_diff, np.array([sigma_0]), args=(E_F_redox, sigma_Q_arr))
        sigma = result.x[0]

        i_1, i_2 = nearest_array_indices(sigma_Q_arr, sigma)
        dE_Q = E_start + E_step * i_1

        E_fermi = self.E - dE_Q
        E_DOS_redox = self.E - dE_Q - overpot

        if reverse:
            y_fermi = 1 - self.fermi_func(E_fermi, self.T)
            y_redox = self.W_red(E_DOS_redox, self.T, self.l)
        else:
            y_fermi = self.fermi_func(E_fermi, self.T)
            y_redox = self.W_ox(E_DOS_redox, self.T, self.l)

        if not add_info:
            return y_fermi, y_redox
        else:
            return y_fermi, y_redox, dE_Q, sigma, E_F_redox

    def compute_k_HET(self, V_std_pot_arr, overpot_arr, reverse: bool = False, add_info: bool = False):
        """Computes integral k_HET using Gerischer-Markus formalism with quantum capacitance

        Parameters:
        ----------
        V_std_pot_arr: float, np.array
            A range of varying a standard potential
        overpot_arr: float, np.array
            A range of varying an overpotential
        reverse: bool, optional
            if reverse is False the process of electron transfer from electrode to the oxidized state of the
            redox mediator is considered and vice versa

        Returns:
        -------
        k_HET: np.array
            Calculated heterogeneous electron transfer rate constant according to Gerischer-Marcus model with quantum
            capacitance
        dE_Q_arr: np.array, optional (if add_info == True)
            Total shift of the Fermi energy due to the Quantum Capacitance for all calculated redox potentials or
            overpotentials
        sigma_arr: np.array, optional (if add_info == True)
            surface charge in microF/cm^2 for all calculated redox potentials or overpotentials
        E_F_redox_arr: np.array, optional (if add_info == True)
            The sum of two energy displacement of the electrode due to the difference in Fermi level of Redox couple
            and the electrode and overpotential. It splits into dE_Q and dE_EDL. For all calculated redox potentials
            or overpotentials
        y_fermi_arr: 2D np.ndarray, optional (if add_info == True)
            Fermi-Dirac distribution for all calculated redox potentials or overpotentials
        y_redox_arr: 2D np.ndarray, optional (if add_info == True)
            Redox species distributions for all calculated redox potentials or overpotentials
        """

        if isinstance(self.C_EDL, numbers.Real):
            if isinstance(V_std_pot_arr, typing.Sequence) and isinstance(overpot_arr, numbers.Real):
                k_HET = np.zeros_like(V_std_pot_arr)
                if not add_info:
                    for i, V_std in tqdm(enumerate(V_std_pot_arr), total=len(V_std_pot_arr)):
                        y_fermi, y_redox = self.compute_distributions(V_std, reverse=reverse, overpot=overpot_arr)
                        integrand = self.DOS * y_fermi * y_redox
                        k_HET[i] = integrate.simps(integrand, self.E)
                    return k_HET
                else:
                    dE_Q_arr = np.zeros_like(V_std_pot_arr)
                    sigma_arr = np.zeros_like(V_std_pot_arr)
                    E_F_redox_arr = np.zeros_like(V_std_pot_arr)
                    y_fermi_arr = np.zeros((len(V_std_pot_arr), len(self.E)))
                    y_redox_arr = np.zeros((len(V_std_pot_arr), len(self.E)))
                    for i, V_std in tqdm(enumerate(V_std_pot_arr), total=len(V_std_pot_arr)):
                        y_fermi, y_redox, dE_Q, sigma, E_F_redox = self.compute_distributions(V_std, reverse=reverse,
                                                                                              overpot=overpot_arr,
                                                                                              add_info=add_info)
                        integrand = self.DOS * y_fermi * y_redox
                        k_HET[i] = integrate.simps(integrand, self.E)
                        dE_Q_arr[i] = dE_Q
                        sigma_arr[i] = sigma
                        E_F_redox_arr[i] = E_F_redox
                        y_fermi_arr[i] = y_fermi
                        y_redox_arr[i] = y_redox
                    return k_HET, dE_Q_arr, sigma_arr, E_F_redox_arr, y_fermi_arr, y_redox_arr

            elif isinstance(overpot_arr, typing.Sequence) and isinstance(V_std_pot_arr, numbers.Real):
                k_HET = np.zeros_like(overpot_arr)
                if not add_info:
                    for i, overpot in tqdm(enumerate(overpot_arr), total=len(overpot_arr)):
                        y_fermi, y_redox = self.compute_distributions(V_std_pot_arr, reverse=reverse, overpot=overpot)
                        integrand = self.DOS * y_fermi * y_redox
                        k_HET[i] = integrate.simps(integrand, self.E)

                    return k_HET
                else:
                    dE_Q_arr = np.zeros_like(overpot_arr)
                    sigma_arr = np.zeros_like(overpot_arr)
                    E_F_redox_arr = np.zeros_like(overpot_arr)
                    y_fermi_arr = np.zeros((len(overpot_arr), len(self.E)))
                    y_redox_arr = np.zeros((len(overpot_arr), len(self.E)))
                    for i, overpot in tqdm(enumerate(overpot_arr), total=len(overpot_arr)):
                        y_fermi, y_redox, dE_Q, sigma, E_F_redox = self.compute_distributions(V_std_pot_arr,
                                                                                              reverse=reverse,
                                                                                              overpot=overpot,
                                                                                              add_info=add_info)
                        integrand = self.DOS * y_fermi * y_redox
                        k_HET[i] = integrate.simps(integrand, self.E)
                        dE_Q_arr[i] = dE_Q
                        sigma_arr[i] = sigma
                        E_F_redox_arr[i] = E_F_redox
                        y_fermi_arr[i] = y_fermi
                        y_redox_arr[i] = y_redox
                    return k_HET, dE_Q_arr, sigma_arr, E_F_redox_arr, y_fermi_arr, y_redox_arr

            else:
                raise ValueError('One and only one type of V_std_pot_arr and overpot arr must be Sequence. The other \
                                 must be a Real number')

        elif self.C_EDL == 'Cl':
            if isinstance(V_std_pot_arr, typing.Iterable) and isinstance(overpot_arr, numbers.Real):
                E_fermi = self.E
                E_DOS_redox = self.E - overpot_arr

                if reverse:
                    y_fermi = 1 - self.fermi_func(E_fermi, self.T)
                    y_redox = self.W_red(E_DOS_redox, self.T, self.l)
                else:
                    y_fermi = self.fermi_func(E_fermi, self.T)
                    y_redox = self.W_ox(E_DOS_redox, self.T, self.l)

                integrand = self.DOS * y_fermi * y_redox
                k_HET = np.ones_like(V_std_pot_arr) * integrate.simps(integrand, self.E)

                return k_HET

            elif isinstance(overpot_arr, typing.Sequence) and isinstance(V_std_pot_arr, numbers.Real):
                k_HET = np.zeros_like(overpot_arr)

                for i, overpot in tqdm(enumerate(overpot_arr), total=len(overpot_arr)):
                    E_fermi = self.E
                    E_DOS_redox = self.E - overpot

                    if reverse:
                        y_fermi = 1 - self.fermi_func(E_fermi, self.T)
                        y_redox = self.W_red(E_DOS_redox, self.T, self.l)
                    else:
                        y_fermi = self.fermi_func(E_fermi, self.T)
                        y_redox = self.W_ox(E_DOS_redox, self.T, self.l)

                    integrand = self.DOS * y_fermi * y_redox
                    k_HET[i] = integrate.simps(integrand, self.E)

                return k_HET

            else:
                raise ValueError('One and only one type of V_std_pot_arr and overpot arr must be Sequence. The other \
                                 must be Real number')

        elif self.C_EDL == 'Q':
            if isinstance(V_std_pot_arr, typing.Sequence) and isinstance(overpot_arr, numbers.Real):
                k_HET = np.zeros_like(V_std_pot_arr)

                for i, V_std in tqdm(enumerate(V_std_pot_arr), total=len(V_std_pot_arr)):
                    E_F_redox = -4.5 - self.efermi - V_std + self.vacuum_lvl
                    E_DOS_redox = self.E - E_F_redox
                    E_fermi = E_DOS_redox - overpot_arr

                    if reverse:
                        y_fermi = 1 - self.fermi_func(E_fermi, self.T)
                        y_redox = self.W_red(E_DOS_redox, self.T, self.l)
                    else:
                        y_fermi = self.fermi_func(E_fermi, self.T)
                        y_redox = self.W_ox(E_DOS_redox, self.T, self.l)

                    integrand = self.DOS * y_fermi * y_redox
                    k_HET[i] = integrate.simps(integrand, self.E)

                return k_HET

            elif isinstance(overpot_arr, typing.Sequence) and isinstance(V_std_pot_arr, numbers.Real):
                k_HET = np.zeros_like(overpot_arr)

                for i, overpot in tqdm(enumerate(overpot_arr), total=len(overpot_arr)):
                    E_F_redox = -4.5 - self.efermi - V_std_pot_arr + self.vacuum_lvl - overpot
                    E_fermi = self.E - E_F_redox
                    E_DOS_redox = self.E - E_F_redox - overpot

                    if reverse:
                        y_fermi = 1 - self.fermi_func(E_fermi, self.T)
                        y_redox = self.W_red(E_DOS_redox, self.T, self.l)
                    else:
                        y_fermi = self.fermi_func(E_fermi, self.T)
                        y_redox = self.W_ox(E_DOS_redox, self.T, self.l)

                    integrand = self.DOS * y_fermi * y_redox
                    k_HET[i] = integrate.simps(integrand, self.E)

                return k_HET

            else:
                raise ValueError('One and only one type of V_std_pot_arr and overpot arr must be Sequence. The other \
                                 must be Real number')

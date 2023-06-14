import numpy as np
from nptyping import NDArray, Shape, Number
import warnings


class ThermalProperties:
    """
    Class for calculation thermal properties based on the calculated phonon spectra

    Args:
        eigen_freq (np.ndarray [nkpts, nfreq]): The energies of phonon eigenfrequencies in eV
        weights (np.ndarray [nkpts, ], optional): weights of k-points. The sum of weights should be equal to 1.
        If weights are not provided, all k-points will be considered with equal weights (1 / nkpts).
    """
    k_B_J = 1.380649e-23  # J/K
    k_B_eV = 8.617333262e-5  # eV/K
    hbar_J = 1.054571817e-34  # J*s

    def __init__(self,
                 eigen_freq: NDArray[Shape['Nkpts, Nfreq'], Number],
                 weights: NDArray[Shape['Nkpts'], Number] = None):

        if weights is None:
            self.weights = np.ones(eigen_freq.shape[0]) / eigen_freq.shape[0]
        else:
            self.weights = weights

        if np.sum(eigen_freq < 0) > 0:
            warnings.warn('\nThere is at least one imaginary frequency in given eigenfrequencies. '
                          '\nAll imaginary frequencies will be dropped from any further calculations'
                          f'\nImaginary frequencies: {eigen_freq[eigen_freq < 0]}')
        self.eigen_freq = np.maximum(0, eigen_freq)

    def get_Gibbs_ZPE(self) -> float:
        r"""
        Calculate Zero Point Energy

        .. math::
            E_{ZPE} = \sum_k weight(k) \sum_i \frac{hw_{i}(k)}{2}

        Returns:
            ZPE in eV
        """
        return np.sum(self.weights * self.eigen_freq) / 2

    def get_enthalpy_vib(self, T) -> float:
        r"""
        Calculate the thermal term in vibrational energy

        .. math::
            E_{temp}(k) = \sum_i \left( \frac{hw_{i}(k)}{( \exp (hw_{i}(k) / k_B T) - 1)} \right) \\\\
            E_{temp} = \sum_k weight(k) \cdot E_{temp}(k)

        Args:
            T: Temperature in K
        Returns:
            Thermal vibrational energy in eV
        """
        k_B = 8.617333262145e-5  # Boltzmann's constant in eV/K

        return np.sum(np.nan_to_num(self.weights * self.eigen_freq / (np.exp(self.eigen_freq / (k_B * T)) - 1)))

    def get_TS_vib(self, T) -> float:
        r"""
        Calculate the vibrational entropy contribution

        .. math::
            S_{vib}(k) = \sum_i \left( \frac{hw_{i}(k)}{( \exp (hw_{i}(k) / k_B T) - 1)} -
            k_B ln \left( 1 - \exp \left(- \frac{hw_i(k)}{k_B T} \right) \right) \right) \\\\
            T * S_{vib} = T \sum_k (weight_k S_{vib}(k))

        Args:
            T: Temperature in K

        Returns:
            TS in eV

        """
        k_B = 8.617333262145e-5  # Boltzmann's constant in eV/K
        second_term = - np.sum(self.weights * k_B * T * np.nan_to_num(np.log(1 - np.exp(- self.eigen_freq / (k_B * T))),
                                                                      neginf=0))
        return self.get_enthalpy_vib(T) + second_term

    def get_Gibbs_vib(self, T: float) -> float:
        return self.get_Gibbs_ZPE() + self.get_enthalpy_vib(T) - self.get_TS_vib(T)

    @classmethod
    def get_Gibbs_trans(cls,
                        V: float,
                        mass: float,
                        T: float):
        return - cls.k_B_eV * T * np.log(V * (mass * cls.k_B_J * T / (2 * np.pi * cls.hbar_J**2))**1.5)

    @classmethod
    def get_Gibbs_rot(cls,
                      I: float | list[float] | NDArray[Shape['3'], Number],
                      sigma: int,
                      T: float):
        if type(I) is float or len(I) == 1:
            return - cls.k_B_eV * T * np.log(2 * I * cls.k_B_J * T / (sigma * cls.hbar_J ** 2))
        elif len(I) == 3:
            return - cls.k_B_eV * T * np.log((2 * cls.k_B_J * T)**1.5 * (np.pi * I[0] * I[1] * I[2])**0.5 /
                                           (sigma * cls.hbar_J**3))
        else:
            raise ValueError(f'I should be either float or array with length of 3, however {len(I)=}')

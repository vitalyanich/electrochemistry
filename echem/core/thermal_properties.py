import numpy as np
from nptyping import NDArray, Shape, Number
import warnings


class Thermal_properties:
    """
    Class for calculation thermal properties based on the calculated phonon spectra

    Args:
        eigen_freq (np.ndarray [nkpts, nfreq]): The energies of phonon eigenfrequencies in eV
        weights (np.ndarray [nkpts, ], optional): weights of k-points. The sum of weights should be equal to 1.
        If weights are not provided, all k-points will be considered with equal weights (1 / nkpts).
    """
    def __init__(self,
                 eigen_freq: NDArray[Shape['Nkpts, Nfreq'], Number],
                 weights: NDArray[Shape['Nkpts'], Number] = None):

        if weights is None:
            self.weights = np.ones(eigen_freq.shape[0]) / eigen_freq.shape[0]
        else:
            self.weights = weights

        if np.sum(eigen_freq < 0) > 0:
            warnings.warn('\nThere is at least one imaginary frequency in given eigenfrequencies. '
                          'All imaginary frequencies will be dropped from any further calculations')
        self.eigen_freq = np.maximum(0, eigen_freq)

    def get_E_zpe(self) -> float:
        r"""
        Calculate Zero Point Energy

        .. math::
            E_{ZPE} = \sum_k weight(k) \sum_i \frac{hw_{i}(k)}{2}

        Returns:
            ZPE in eV
        """
        return np.sum(self.weights * self.eigen_freq) / 2

    def get_E_temp(self, T) -> float:
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

    def get_TS(self, T) -> float:
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
        return self.get_E_temp(T) + second_term

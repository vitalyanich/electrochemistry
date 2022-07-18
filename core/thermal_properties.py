import numpy as np
from nptyping import NDArray, Shape, Number
import warnings


class Thermal_properties:
    """
    Class for calculation thermal properties based on the calculated phonons

    Args:
        eigen_freq (np.ndarray [nkpts, nfreq]): The energies of phonon eigenfrequencies in eV
        weights (np.ndarray [nkpts, ], optional): weights of k-points. The sum of weights should be equal to 1
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
        """
        Calculate Zero Point Energy

        Returns:
            ZPE in eV
        """
        return np.sum(self.weights * self.eigen_freq) / 2

    def get_E_temp(self, T) -> float:
        """
        Calculate the thermal term in vibrational energy. E_temp(k) = sum_i (hw_ki / (exp(hw_ki / k_B * T) - 1)).
        E_temp = sum_k (weights_k * E_temp(k))

        Args:
            T: Temperature in K
        Returns:
            Thermal energy in eV
        """
        k_B = 8.617333262145e-5  # Boltzmann's constant in eV/K

        return np.sum(np.nan_to_num(self.weights * self.eigen_freq / (np.exp(self.eigen_freq / (k_B * T)) - 1)))

    def get_TS(self, T) -> float:
        """
        Calculate the vibrational entropy contribution.
        T * S_vib (k) = sum_i (hw_ki / (exp(hw_ki / k_B * T) - 1) - k_B ln(1 - exp(- hw_ki / k_B * T)))
        T * S_vib = sum_k (weight_k * T * S_vib (k))

        Args:
            T: Temperature in K

        Returns:
            TS in eV

        """
        k_B = 8.617333262145e-5  # Boltzmann's constant in eV/K
        second_term = - np.sum(self.weights * k_B * T * np.nan_to_num(np.log(1 - np.exp(- self.eigen_freq / (k_B * T))),
                                                                      neginf=0))
        return self.get_E_temp(T) + second_term

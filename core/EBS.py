import numpy as np
from typing import Union, Iterable


class EBS:
    def __init__(self):
        self.eigenvalues = None
        self.occupations = None
        self.weights = None
        self.efermi = None

    @property
    def nbands(self):
        return self.eigenvalues.shape[-1]

    @property
    def nkpts(self):
        return len(self.weights)

    @property
    def nspin(self):
        if len(self.eigenvalues.shape) == 2:
            return 1
        elif len(self.eigenvalues.shape) == 3:
            return self.eigenvalues.shape[0]

    @staticmethod
    def _GaussianSmearing(x, x0, sigma):
        """Simulate the Delta function by a Gaussian shape function"""

        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def __get_by_bands(self,  property: str, bands: Union[int, Iterable]):
        property = getattr(self, property)
        if type(bands) is int:
            if self.nspin == 1:
                return property[:, bands]
            elif self.nspin > 1:
                return property[:, :, bands]
        elif isinstance(bands, Iterable):
            if self.nspin == 1:
                return property[:, bands].transpose(1, 0)
            elif self.nspin > 1:
                return property[:, :, bands].transpose(2, 0, 1)
        else:
            raise ValueError('Variable bands should be int or iterable')

    def get_band_eigs(self, bands: Union[int, Iterable]):
        return self.__get_by_bands('eigenvalues', bands)

    def get_band_occ(self, bands: Union[int, Iterable]):
        return self.__get_by_bands('occupations', bands)

    def get_DOS(self, **kwargs):

        # TODO: Add if smearing == False, add if smearing == "Lorenz", Check *2 electrons/states?
        """Calculate Density of States based on eigenvalues and its weights

        Args:
            dE (float, optional): step of energy array in function's output. Default value is 0.01
            zero_at_fermi (bool, optional): if True Fermi energy will be equal to zero
            emin (float, optional): minimum value in DOS calculation.
            emax (float, optional): maximum value in DOS calculation.
            smearing (str, optional): define whether will be used smearing or not. Default value is 'Gaussian'.
            Possible options: 'Gaussian'
            sigma (float, optional): define the sigma parameter in Gaussian smearing. Default value is 0.02

        Returns:
            E, DOS - Two 1D np.arrays that contain energy and according DOS values
        """
        if 'zero_at_fermi' in kwargs:
            zero_at_fermi = kwargs['zero_at_fermi']
        else:
            zero_at_fermi = False

        if 'dE' in kwargs:
            dE = kwargs['dE']
        else:
            dE = 0.01

        if 'smearing' in kwargs:
            smearing = kwargs['smearing']
        else:
            smearing = 'Gaussian'

        if smearing == 'Gaussian':
            if 'sigma' in kwargs:
                sigma = kwargs['sigma']
            else:
                sigma = 0.02
            if 'emin' in kwargs:
                E_min = kwargs['emin']
            else:
                E_min = np.min(self.eigenvalues)
            if 'emax' in kwargs:
                E_max = kwargs['emax']
            else:
                E_max = np.max(self.eigenvalues)
            E_arr = np.arange(E_min, E_max, dE)

            if self.nspin == 1:
                DOS_arr = np.zeros_like(E_arr)
                for energy_kpt, weight in zip(self.eigenvalues, self.weights):
                    for energy in energy_kpt:
                        DOS_arr += 2 * weight * self._GaussianSmearing(E_arr, energy, sigma)
                        # 2 above means occupancy for spin unrestricted calculation
            elif self.nspin > 1:
                DOS_arr = np.zeros((self.nspin,) + np.shape(E_arr))
                for spin in range(self.nspin):
                    for energy_kpt, weight in zip(self.eigenvalues[spin], self.weights):
                        for energy in energy_kpt:
                            DOS_arr[spin] += weight * self._GaussianSmearing(E_arr, energy, sigma)
            else:
                raise ValueError(f'nspin should be equal to 1 or 2 but you set {self.nspin=}')

            if zero_at_fermi:
                return E_arr - self.efermi, DOS_arr
            else:
                return E_arr, DOS_arr

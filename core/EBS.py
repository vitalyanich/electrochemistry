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
        return self.eigenvalues.shape[0]

    @staticmethod
    def GaussianSmearing(E: np.ndarray, E0: np.ndarray, sigma: Union[int, float]):
        """
        Blur the Delta function by a Gaussian function
        Args:
            E: Numpy array with the shape (ngrid, ) that represents the energy range for the Gaussian smearing
            E0: Numpy array with any shape (i.e. (nspin, nkpts, nbands)) that contains eigenvalues
            sigma: the broadening parameter for the Gaussian function

        Returns:
            Smeared eigenvalues on grind E with the shape E0.shape + E.shape
        """
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(
            -(np.broadcast_to(E, E0.shape + E.shape) - np.expand_dims(E0, len(E0.shape))) ** 2 / (2 * sigma ** 2))

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
            DOS_arr = np.sum(self.weights[None, :, None, None] *
                             self.GaussianSmearing(E_arr, self.eigenvalues, sigma), axis=(1, 2))

            # 2 means occupancy for spin non-spinpolarized calculation
            if self.nspin == 1:
                DOS_arr *= 2

            if zero_at_fermi:
                return E_arr - self.efermi, DOS_arr
            else:
                return E_arr, DOS_arr
import numpy as np
from typing import Union, Iterable
from nptyping import NDArray, Shape, Number


class EBS:
    """
    Class for calculating DOS

    Args:
        eigenvalues
    """
    def __init__(self,
                 eigenvalues: NDArray[Shape['Nspin, Nkpts, Nbands'], Number],
                 weights: NDArray[Shape['Nkpts'], Number] = None,
                 efermi: float = None,
                 occupations: NDArray[Shape['Nspin, Nkpts, Nbands'], Number] = None):

        self.eigenvalues = eigenvalues
        self.occupations = occupations
        self.efermi = efermi

        if weights is None:
            self.weight = np.ones(eigenvalues.shape[1]) / eigenvalues.shape[1]
        else:
            self.weight = weights

    @property
    def nspin(self):
        return self.eigenvalues.shape[0]

    @property
    def nkpts(self):
        return self.eigenvalues.shape[1]

    @property
    def nbands(self):
        return self.eigenvalues.shape[2]

    @staticmethod
    def gaussian_smearing(E: NDArray[Shape['Ngrid'], Number],
                          E0: NDArray[Shape['*, ...'], Number],
                          sigma: float):
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

    def __get_by_bands(self,
                       property: str,
                       bands: Union[int, Iterable[int]]):
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
        if self.occupations is not None:
            return self.__get_by_bands('occupations', bands)
        else:
            raise ValueError('Occupations has not been defined')

    def get_DOS(self,
                dE: float = 0.01,
                emin: float = None,
                emax: float = None,
                zero_at_fermi: bool = False,
                smearing: str = 'gaussian',
                sigma: float = 0.02) -> tuple[NDArray[Shape['Ngrid'], Number], NDArray[Shape['Nspin, Ngrid'], Number]]:
        """Calculate Density of States based on eigenvalues and its weights

        Args:
            dE (float, optional): step of energy array in function's output. Default value is 0.01
            zero_at_fermi (bool, optional): if True Fermi energy will be equal to zero
            emin (float, optional): minimum value in DOS calculation.
            emax (float, optional): maximum value in DOS calculation.
            smearing (str, optional): define whether will be used smearing or not. Default value is 'Gaussian'.
            Possible options: 'gaussian'
            sigma (float, optional): define the sigma parameter in Gaussian smearing. Default value is 0.02

        Returns:
            E, DOS - Two 1D np.arrays that contain energy and according DOS values
        """
        if zero_at_fermi is True and self.efermi is None:
            raise ValueError('You can not set zero_at_fermi=True if you did not specify efermi value')

        if emin is None:
            E_min = np.min(self.eigenvalues) - 1
        if emax is None:
            E_max = np.max(self.eigenvalues) + 1

        E_arr = np.arange(E_min, E_max, dE)

        if smearing.lower() == 'gaussian':
            DOS_arr = np.sum(self.weights[None, :, None, None] *
                             self.gaussian_smearing(E_arr, self.eigenvalues, sigma), axis=(1, 2))
        else:
            raise NotImplemented(f'Smearing {smearing} is not implemented. Please use \'gaussian\' instead.')

        # 2 means occupancy for non-spinpolarized calculation
        if self.nspin == 1:
            DOS_arr *= 2

        if zero_at_fermi:
            return E_arr - self.efermi, DOS_arr
        else:
            return E_arr, DOS_arr

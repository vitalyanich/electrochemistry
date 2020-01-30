import numpy as np
from seaborn.distributions import stats
from scipy.integrate import simps
from monty.re import regrep
import itertools
import typing


class Output:
    def __init__(self):
        self.patterns = {'nkpts': r'number of k points=\s+([\d]+)',
                         'kpts_coord': r'k\s*=\s*(-?\d.[\d]+)\s*(-?\d.[\d]+)\s*(-?\d.[\d]+)\s*\([\d]+ PWs\)',
                         'occupations': 'occupation numbers',
                         'efermi': r'the Fermi energy is\s*(-?[\d]+.[\d]+) ev'}
        self.eigenvalues = None
        self.weights = None
        self.occupations = None
        self.efermi = None
        self.nkpt = None

    def from_file(self, filepath):
        matches = regrep(filepath, self.patterns)

        if len(matches['kpts_coord']) != 0:
            with open(filepath, 'rt') as file:
                file_data = file.readlines()
                eigenvalues = []
                for start, end in zip(matches['kpts_coord'], matches['occupations']):
                    data = file_data[start[1] + 2:end[1] - 1]
                    data = [float(i) for i in itertools.chain.from_iterable([line.split() for line in data])]
                    eigenvalues.append(data)
                self.eigenvalues = np.array(eigenvalues)

                occupations = []
                n_strings_occups = matches['occupations'][0][1] - matches['kpts_coord'][0][1] - 1
                for start in matches['occupations']:
                    data = file_data[start[1] + 1: start[1] + n_strings_occups]
                    data = [float(i) for i in itertools.chain.from_iterable([line.split() for line in data])]
                    occupations.append(data)
                self.occupations = np.array(occupations)

        self.efermi = float(matches['efermi'][0][0][0])
        self.nkpt = int(matches['nkpts'][0][0][0])

        weights = np.zeros(self.nkpt)

        for i in range(self.nkpt):
            weights[i] = file_data[matches['nkpts'][0][1]+2+i].split()[-1]
        self.weights = weights

    def get_band_eigs(self, bands):
        if type(bands) is int:
            return np.array([eig for eig in self.eigenvalues[:, bands]])
        elif isinstance(bands, typing.Iterable):
            return np.array([[eig for eig in self.eigenvalues[:, band]] for band in bands])
        else:
            raise ValueError('Variable bands should be int or iterable')

    def get_band_occ(self, bands):
        if type(bands) is int:
            return [occ for occ in self.occupations[:, bands]]
        elif isinstance(bands, typing.Iterable):
            return np.array([[occ for occ in self.occupations[:, band]] for band in bands])
        else:
            raise ValueError('Variable bands should be int or iterable')

    def get_DOS(self, dE, zero_at_fermi=False, sm_param=None):
        """Calculate Density of States based on eigenvalues and its weights

        Args:
            dE (float): step of energy array in function output
            zero_at_fermi (bool, optional): if True Fermi energy will be equal to zero
            sm_param (dict, optional): parameters for smooth DOS.
                E_min (float, str): minimum value in DOS calculation. If E_min == 'min' left border of energy
                is equal to the minimum eigenvalue
                E_max (float, str): maximum value in DOS calculation. If E_max == 'max' right border of energy
                is equal to the maximum eigenvalue
                bw_method (float): The method used to calculate the estimator bandwidth. This can be 'scott',
                'silverman', a scalar constant or a callable. If a scalar, this will be used directly as `kde.factor`.
                If a callable, it should take a `gaussian_kde` instance as only parameter and return a scalar.
                If None (default), 'scott' is used.
                nelec (int): Number of electrons in the system. DOS integral to efermi should be equal to the nelec

        Returns:
            E, DOS - Two 1D np.arrays that contain energy and according DOS values
        """
        if sm_param is None:
            E_min = np.min(self.eigenvalues.flatten())
            E_max = np.max(self.eigenvalues.flatten())
            E_arr = np.arange(E_min, E_max, dE)
            energies_number = len(E_arr)
            DOS_arr = np.zeros(energies_number)
            for energy_band, weight in zip(self.eigenvalues, self.weights):
                for energy in energy_band:
                    place = int((energy - E_arr[0]) / dE)
                    DOS_arr[place] += weight / dE
            if zero_at_fermi is False:
                return E_arr, DOS_arr
            else:
                return E_arr - self.efermi, DOS_arr
        else:
            weights_flatten = []
            for energy_band, weight in zip(self.eigenvalues, self.weights):
                weights_flatten.append(np.ones(len(energy_band)) * weight)
            weights_flatten = np.array(weights_flatten).flatten()
            a = stats.kde.gaussian_kde(self.eigenvalues.flatten(), bw_method=sm_param['bw_method'],
                                       weights=weights_flatten)
            if sm_param['E_min'] == 'min':
                sm_param['E_min'] = np.min(self.eigenvalues.flatten())
            if sm_param['E_max'] == 'max':
                sm_param['E_max'] = np.max(self.eigenvalues.flatten())
            x = np.arange(sm_param['E_min'], sm_param['E_max'], dE)
            y = a.evaluate(x)

            i = 0
            while self.efermi > x[i]:
                i += 1

            integral = simps(y[:i], x[:i])
            y = (sm_param['nelec'] / integral) * y

            if zero_at_fermi is False:
                return x, y
            else:
                return x - self.efermi, y

from monty.re import regrep
import itertools
import numpy as np
from pymatgen.io.vasp.outputs import Locpot


class Preprocessing:

    def __init__(self):
        self.efermi = None
        self.nkpts = None
        self.nbands = None
        self.weights = None
        self.eigenvalues = None
        self.occupations = None
        self.vacuum_lvl = None

    def process_WAVECAR(self):
        pass

    def process_OUTCAR(self, file_path):
        pass

    def get_DOS(self, E_DOS):
        pass

    def get_band_eigs(self, bands):
        pass

    def get_band_occ(self, bands):
        pass

    def process_LOCPOT(self, file_path=None):

        if file_path is None:
            file_path = 'LOCPOT'

        locpot = Locpot.from_file(file_path)
        avr = locpot.get_average_along_axis(2)
        vacuum_lvl = np.max(avr)

        self.vacuum_lvl = vacuum_lvl

        return vacuum_lvl

    def save_all(self):
        pass

    def load(self, variable):

        if object == 'all':
            self.efermi = np.load(f'Saved_data/efermi.npy').items()
            self.nkpts = np.load(f'Saved_data/nkpts.npy').items()
            self.nbands = np.load(f'Saved_data/nbands.npy').items()
            self.weights = np.load(f'Saved_data/weights.npy').items()
            self.eigenvalues = np.load(f'Saved_data/eigenvalues.npy').items()
            self.occupations = np.load(f'Saved_data/occupations.npy').items()
            self.vacuum_lvl = np.load(f'Saved_data/vacuum_lvl.npy').items()
        else:
            class_variable = getattr(self, variable, lambda: 'Invalid variable')
            class_variable = np.load(f'Saved_data/{variable}').items()


class Espresso:
    def __init__(self, filepath):
        self.filepath = filepath
        self.patterns = {"nkpts": r"number of k points=\s+([\d]+)",
                         'kpts_coord': r'k\s*=\s*(-?\d.[\d]+)\s*(-?\d.[\d]+)\s*(-?\d.[\d]+)\s*\([\d]+ PWs\)',
                         'occupations': 'occupation numbers',
                         'efermi': r'the Fermi energy is\s*(-?[\d]+.[\d]+) ev'}
        self.eigenvalues = None
        self.efermi = None

    def read_data(self):
        matches = regrep(self.filepath, self.patterns)
        self.matches = matches

        if len(self.matches['kpts_coord']) != 0:
            with open(self.filepath, 'r') as file:
                file_data = file.readlines()
                self.eigenvalues = []
                for start, end in zip(self.matches['kpts_coord'], self.matches['occupations']):
                    data = file_data[start[1] + 2:end[1] - 1]
                    for i, line in enumerate(data):
                        data[i] = line.split()
                    data = [float(i) for i in itertools.chain.from_iterable(data)]
                    self.eigenvalues.append(data)
                self.eigenvalues = np.array(self.eigenvalues)

                self.occupations = []
                n_strings_occups = self.matches['occupations'][0][1] - self.matches['kpts_coord'][0][1] - 1
                for start in self.matches['occupations']:
                    data = file_data[start[1] + 1: start[1] + n_strings_occups]
                    for i, line in enumerate(data):
                        data[i] = line.split()
                    data = [float(i) for i in itertools.chain.from_iterable(data)]
                    self.occupations.append(data)
                self.occupations = np.array(self.occupations)

        self.efermi = float(self.matches['efermi'][0][0][0])
        self.nkpt = int(self.matches['nkpts'][0][0][0])

    def get_band_eigs(self, bands):
        if type(bands) is int:
            return np.array([eig for eig in self.eigenvalues[:, bands]])
        if (type(bands) is list) or (type(bands) is np.ndarray):
            return np.array([[eig for eig in self.eigenvalues[:, band]] for band in bands])

    def get_band_occ(self, band):
        if type(band) is int:
            return [occ for occ in self.occupations[:, band]]

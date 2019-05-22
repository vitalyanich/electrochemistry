from monty.re import regrep
import itertools
import numpy as np
from pymatgen.io.vasp.outputs import Locpot
import os


class Preprocessing():

    def __init__(self):
        self.efermi = None
        self.nkpts = None
        self.nbands = None
        self.weights = None
        self.eigenvalues = None
        self.occupations = None
        self.vacuum_lvl = None

    def process_WAVECAR(self, file_path):
        """

        :param file_path:
        :return:
        """


    def process_OUTCAR(self, file_path):
        """
        process OUTCAR file obtained from VASP
        get following variables:
        self.nkpts - number of k-points (int)
        self.efermi - Fermi level (float)
        self.nbands - number of bands (int)
        self.eigenvalues - 2D np.array, eigenvalues[i][j] contains energy for i k-point and j band
        self.occupations - 2D np.array, occupations[i][j] contains occupation for i k-point and j band
        :param file_path: path to OUTCAR file
        :return: nothing
        """
        patterns = {'nkpts': r'Found\s+(\d+)\s+irreducible\sk-points',
                    'weights':'k-points in units of 2pi/SCALE and weight: K-Points',
                    'efermi': 'E-fermi\s:\s+([-.\d]+)',
                    'kpoints': r'k-point\s+(\d+)\s:\s+[-.\d]+\s+[-.\d]+\s+[-.\d]+\n'}
        matches = regrep(file_path, patterns)

        self.nkpts = int(matches['nkpts'][0][0][0])
        self.efermi = float(matches['efermi'][0][0][0])
        self.nbands = int(matches['kpoints'][1][1] - matches['kpoints'][0][1]-3)
        self.eigenvalues = []
        self.occupations = []

        with open(file_path) as file:
            lines = file.readlines()
            for kpoint in range(self.nkpts-1):
                self.eigenvalues.append([])
                self.occupations.append([])
                startline = matches['kpoints'][kpoint][1]+2
                for i in range(startline, startline + self.nbands):
                    self.eigenvalues[kpoint].append(float(lines[i].split()[1]))
                    self.occupations[kpoint].append(float(lines[i].split()[2]))
        self.eigenvalues = np.array(self.eigenvalues)
        self.occupations = np.array(self.occupations)
        if not os.path.exists('Saved_data'):
            os.mkdir('Saved_data')
        self.save('all')

    def get_DOS(self, E_range, dE):

        pass

    def get_band_eigs(self, band, path_to_OUTCAR = 'OUTCAR'):
        if self.eigenvalues == None:
            try:
                print("Variable is not define. Try load from file")
                self.load('eigenvalues')
            except:
                print("Loading failed. Start processing OUTCAR")
                self.process_OUTCAR(path_to_OUTCAR)
            return self.eigenvalues[:,band]
        else:
            return self.eigenvalues[:,band]

    def get_band_occ(self, band, path_to_OUTCAR = 'OUTCAR'):
        if self.occupations == None:
            try:
                print("Variable is not define. Try load from file")
                self.load('occupations')
            except:
                print("Loading failed. Start processing OUTCAR")
                self.process_OUTCAR(path_to_OUTCAR)
            return self.occupations[:,band]
        else:
            return self.occupations[:,band]

    def     process_LOCPOT(self, file_path=None):

        if file_path is None:
            file_path = 'LOCPOT'

        locpot = Locpot.from_file(file_path)
        avr = locpot.get_average_along_axis(2)
        vacuum_lvl = np.max(avr)

        self.vacuum_lvl = vacuum_lvl

        return vacuum_lvl

    def save(self, variable='all'):
        if variable == 'all':
            np.save('Saved_data/efermi.npy', self.efermi)
            np.save('Saved_data/nkpts.npy', self.nkpts)
            np.save('Saved_data/nbands.npy', self.nbands)
            np.save('Saved_data/weights.npy', self.weights)
            np.save('Saved_data/eigenvalues.npy', self.eigenvalues)
            np.save('Saved_data/occupations.npy', self.occupations)
            np.save('Saved_data/vacuum_lvl.npy', self.vacuum_lvl)
            print('All variables Saved')
        else:
            np.save('Saved_data/'+variable+'.npy', getattr(self, variable))

    def load(self, variable='all'):
        if variable == 'all':
            self.efermi = np.load('Saved_data/efermi.npy')
            self.nkpts = np.load('Saved_data/nkpts.npy')
            self.nbands = np.load('Saved_data/nbands.npy')
            self.weights = np.load('Saved_data/weights.npy')
            self.eigenvalues = np.load('Saved_data/eigenvalues.npy')
            self.occupations = np.load('Saved_data/occupations.npy')
            self.vacuum_lvl = np.load('Saved_data/vacuum_lvl.npy')
        else:
            setattr(self, variable, np.load('Saved_data/'+str(variable)+'.npy'))


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

if __name__=='__main__':
    outcar_path = 'OUTCAR'
    preprocessing = Preprocessing()
    print(preprocessing.get_band_eigs(1))
    #preprocessing.load('occupations')

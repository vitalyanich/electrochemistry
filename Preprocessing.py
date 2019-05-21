from monty.re import regrep
import itertools


class Preprocessing():

    


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
import numpy as np
from monty.re import regrep
import itertools
from pathlib import Path
from ase.atoms import Atoms
from ase.io.espresso import read_espresso_in
from echem.core.useful_funcs import is_int, is_float
import re


def to_bool(string: str):
    if string.strip('.').lower() == 'true':
        return True
    elif string.strip('.').lower() == 'false':
        return False
    else:
        return string


class Input:
    def __init__(self,
                 atoms: Atoms = None,
                 params: dict = None,
                 kpts: set = None,
                 koffset: set = None,
                 pseudopotentials: dict[str, str] = None,
                 additional_cards: list[str] = None):
        self.atoms = atoms
        self.params = params
        self.kpts = kpts
        self.koffset = koffset
        self.pseudopotentials = pseudopotentials
        self.additional_cards = additional_cards

    @staticmethod
    def __strip_lines(data: list[str]):
        new_data = []
        for line in data:
            line = re.sub('!.*', '', line)
            line = line.strip('\n').strip(',').strip()
            if line == '':
                continue

            if ',' in line:
                for line in line.split(','):
                    new_data.append(line)
            else:
                new_data.append(line)

        return new_data

    @staticmethod
    def from_file(filepath: str | Path):
        if isinstance(filepath, str):
            filepath = Path(filepath)

        file = open(filepath, 'r')
        data = file.readlines()
        file.close()
        data = Input.__strip_lines(data)

        params = {}
        subparams = {}
        namelist = None
        matches = {}

        for i, line in enumerate(data):
            if line.startswith('&'):
                namelist = line.strip('&').upper()
                continue
            elif line == '/':
                params[namelist] = subparams
                namelist = None
                subparams = {}
                continue

            if namelist is not None:
                key, val = line.split('=')
                key = key.strip()
                val = val.strip()
                if is_int(val):
                    val = int(val)
                elif is_float(val):
                    val = float(val)
                else:
                    val = val.strip('\'').strip('\"')
                    val = to_bool(val)
                subparams[key] = val
                continue

            if line.startswith('K_POINTS'):
                matches['kpoints'] = i
            elif line.startswith('ATOMIC_SPECIES'):
                matches['species'] = i
            elif line.startswith('SOLVENTS'):
                matches['solvents'] = i

        if 'kpoints' in matches.keys():
            i = matches['kpoints']
            tag = data[i].split()[1]
            if tag == 'gamma':
                kpts = None
                koffset = None
            elif tag == 'automatic':
                k1, k2, k3, s1, s2, s3 = [int(_) for _ in data[i + 1].split()]
                kpts = (k1, k2, k3)
                koffset = (s1, s2, s3)
            else:
                raise NotImplemented('Only K_POINTS == "automatic" or "gamma" is supported')

        if 'species' in matches.keys():
            nspecies = int(params['SYSTEM']['ntyp'])
            pseudopotentials = {}
            for i in range(nspecies):
                line_splitted = data[matches['species'] + 1 + i].split()
                pseudopotentials[line_splitted[0]] = line_splitted[2]
        else:
            pseudopotentials = None

        if 'solvents' in matches.keys():
            nsolvs = int(params['RISM']['nsolv'])
            additional_cards = [data[matches['solvents'] + i] for i in range(nsolvs + 1)]
        else:
            additional_cards = None

        try:
            atoms = read_espresso_in(filepath)
        except TypeError:
            atoms = None

        return Input(atoms, params, kpts, koffset, pseudopotentials, additional_cards)


class QEOutput:
    def __init__(self):
        self.patterns = {'nkpts': r'number of k points=\s+([\d]+)',
                         'kpts_coord': r'k\s*=\s*(-?\d.[\d]+)\s*(-?\d.[\d]+)\s*(-?\d.[\d]+)\s*\(\s*[\d]+ PWs\)',
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

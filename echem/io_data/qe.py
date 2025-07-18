import numpy as np
from monty.re import regrep
import itertools
from pathlib import Path
from ase.atoms import Atoms
from ase.io.espresso import read_espresso_in
from echem.core.useful_funcs import is_int, is_float
import re
from nptyping import NDArray, Shape, Number


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
    def __to_bool(string: str):
        if string.strip('.').lower() == 'true':
            return True
        elif string.strip('.').lower() == 'false':
            return False
        else:
            return string

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
                namelist = line.strip('&').lower()
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
                    val = Input.__to_bool(val)
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
            nspecies = int(params['system']['ntyp'])
            pseudopotentials = {}
            for i in range(nspecies):
                line_splitted = data[matches['species'] + 1 + i].split()
                pseudopotentials[line_splitted[0]] = line_splitted[2]
        else:
            pseudopotentials = None

        if 'solvents' in matches.keys():
            nsolvs = int(params['rism']['nsolv'])
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


class BandsOut:
    def __init__(self,
                 coords: NDArray[Shape['Nkpts, 3'], Number],
                 eigenvalues: NDArray[Shape['Nkpts, Nbands'], Number]):
        self.coords = coords
        self.eigenvalues = eigenvalues

    @staticmethod
    def from_file(filepath: str | Path):
        if isinstance(filepath, str):
            filepath = Path(filepath)

        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        nbands = int(re.findall(r'nbnd\s*=\s*(\d+)', data[0])[0])
        nkpts = int(re.findall(r'nks\s*=\s*(\d+)', data[0])[0])
        nspin = 1

        x, y = divmod(nbands, 10)
        if y == 0:
            nstr_eigs = x
        else:
            nstr_eigs = x + 1

        coords = np.zeros((nkpts, 3))
        lines_coords = range(1, nkpts * (nstr_eigs + 1), nstr_eigs + 1)
        for i, n_line in enumerate(lines_coords):
            coords[i] = [float(j) for j in data[n_line].split()]

        eigenvalues = np.zeros((nspin, nkpts, nbands))
        lines_eigs = range(2, nkpts * (nstr_eigs + 1), nstr_eigs + 1)
        for i, n_line in enumerate(lines_eigs):
            eigs = [line.strip().split() for line in data[n_line: n_line + nstr_eigs]]
            eigs = list(itertools.chain(*eigs))
            eigenvalues[0, i] = eigs

        return BandsOut(coords, eigenvalues)


class ProjWFC_out:
    def __init__(self, atomic_states, projections):
        self.atomic_states = atomic_states
        self.projections = projections

    @staticmethod
    def from_file(filepath):
        patterns = {'eigenvalues': r'====\se\(\s*\d+\)\s=\s*(\S+)\seV\s====',
                    'psi': r'psi\s+=\s+',
                    'psi2': r'\|psi\|\^2',
                    'nbands': r'nbnd\s+=\s+(\d+)',
                    'natomwfc': r'natomwfc\s+=\s+(\d+)',
                    'atomic_states': r'atom\s+(\d+)\s+\(([a-zA-Z0-9]+)\s*\)\,\s+wfc\s+(\d+)\s*\(l=(\d+)\s*m=\s*(\d+)\)'}

        matches = regrep(str(filepath), patterns)

        nbands = int(matches['nbands'][0][0][0])
        natomwfc = int(matches['natomwfc'][0][0][0])

        eigenvalues = [i[0][0] for i in matches['eigenvalues']]
        eigenvalues = np.array(eigenvalues).reshape(-1, nbands)

        nkpts = eigenvalues.shape[0]

        atomic_states = [[int(i[0][0]) - 1, i[0][1], int(i[0][3]), int(i[0][4])] for i in matches['atomic_states']]

        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        projections = np.zeros((nkpts * nbands, natomwfc))
        psis = [i[1] for i in matches['psi']]
        psi2s = [i[1] for i in matches['psi2']]
        for i, (psi, psi2) in enumerate(zip(psis, psi2s)):
            for j in range(psi, psi2):
                coeffs = re.findall(r'([.\d]+)\*\[#\s*(\d+)\]', data[j])
                for coef, atomband in zip([float(l[0]) for l in coeffs],
                                          [int(l[1]) for l in coeffs]):
                    projections[i, atomband - 1] = coef

        projections = projections.reshape(nkpts, nbands, natomwfc)

        return ProjWFC_out(atomic_states, projections)

    @property
    def nkpts(self):
        return self.projections.shape[0]

    @property
    def nbands(self):
        return self.projections.shape[1]

    @property
    def natomwfc(self):
        return self.projections.shape[2]

    @property
    def species(self):
        return [i[1] for i in self.atomic_states]

    @staticmethod
    def __in_atom_numbers(s, atom_numbers):
        if atom_numbers is None:
            return True
        else:
            return s[0] in atom_numbers

    @staticmethod
    def __in_species(s, species):
        if species is None:
            return True
        else:
            return s[1] in species

    @staticmethod
    def __in_ls(s, ls):
        if ls is None:
            return True
        else:
            return s[2] in ls

    @staticmethod
    def __in_ms(s, ms):
        if ms is None:
            return True
        else:
            return s[3] in ms

    def select_atomic_states(self,
                             atom_numbers=None,
                             species=None,
                             ls=None,
                             ms=None):
        if isinstance(atom_numbers, int):
            atom_numbers = [atom_numbers]
        if isinstance(species, str):
            species = [species]
        if isinstance(ls, int):
            ls = [ls]
        if isinstance(ms, int):
            ms = [ms]

        return [i for i, s in enumerate(self.atomic_states) if
                self.__in_atom_numbers(s, atom_numbers) and
                self.__in_species(s, species) and
                self.__in_ls(s, ls) and
                self.__in_ms(s, ms)]


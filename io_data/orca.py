if __name__ == '__main__':
    import sys, importlib
    from pathlib import Path

    def import_parents(level=1):
        global __package__
        file = Path(__file__).resolve()
        parent, top = file.parent, file.parents[level]
        sys.path.append(str(top))
        try:
            sys.path.remove(str(parent))
        except ValueError:  # already removed
            pass
        __package__ = '.'.join(parent.parts[len(top.parts):])
        importlib.import_module(__package__)  # won't be needed after that

    if __name__ == '__main__' and __package__ is None:
        import_parents(level=2)


import numpy as np
import pandas as pd
import re
from monty.re import regrep
from tqdm import tqdm


class SCFLog:
    """"""
    def __init__(self, eigenvalues=None, occupation=None, mol_orbs=None):
        """"""
        self.eigenvalues = eigenvalues
        self.occupations = occupation
        self.mol_orbs = mol_orbs

    @property
    def natoms(self):
        if self.mol_orbs is not None:
            return np.max(self.mol_orbs[0]['atom_ids']) + 1
        else:
            return ValueError('natoms might be calculated only if mol_orbs had been read')

    @property
    def nbands(self):
        if self.eigenvalues is not None:
            return len(self.eigenvalues[0])
        elif self.mol_orbs is not None:
            return len(self.mol_orbs[0].columns) - 3
        else:
            return ValueError('nbands might be calculated only if eigenvalues or mol_orbs had been read')

    @property
    def nsteps(self):
        if self.eigenvalues is not None:
            return len(self.eigenvalues)
        elif self.mol_orbs is not None:
            return len(self.mol_orbs)
        else:
            return ValueError('nbands might be calculated only if eigenvalues or mol_orbs had been read')

    @staticmethod
    def from_file(filepath):
        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'eigs': 'ORBITAL ENERGIES',
                    'mos': 'MOLECULAR ORBITALS'}
        matches = regrep(filepath, patterns)

        occs = []
        eigs = []
        for match in tqdm(matches['eigs'], desc='Eigenvalues', total=len(matches['eigs'])):
            eigs_tmp = []
            occs_tmp = []
            i = match[1] + 4
            while data[i] != '\n' and data[i] != '------------------\n':
                line = data[i].split()
                occs_tmp.append(float(line[1]))
                eigs_tmp.append(float(line[3]))
                i += 1
            occs.append(occs_tmp)
            eigs.append(eigs_tmp)

        mos_arr = []
        for match in tqdm(matches['mos'], desc='Molecular Orbitals', total=len(matches['mos'])):
            df = pd.DataFrame()
            first_columns_appended = None
            last_batch_added = False
            i = match[1] + 2

            while data[i] != '\n' and data[i] != '------------------\n':
                if re.match(r'\s*\w+\s+\w+\s+([-+]?\d*\.\d*\s+)+', data[i]) is not None:
                    last_batch_added = False
                    line = data[i].split()
                    if first_columns_appended is False:
                        atom_number = re.match(r'\d+', line[0])
                        mos_tmp[0].append(int(atom_number[0]))
                        atom_symbol = line[0][len(atom_number[0]):]
                        mos_tmp[1].append(atom_symbol)
                        orbital = line[1]
                        mos_tmp[2].append(orbital)
                        for j, value in enumerate(line[2:]):
                            mos_tmp[3 + j].append(float(value))
                        i += 1
                    elif first_columns_appended is True:
                        for j, value in enumerate(line[2:]):
                            mos_tmp[j].append(float(value))
                        i += 1
                    else:
                        pass

                elif re.match(r'\s*(\d+\s+)+', data[i]) is not None:
                    line = data[i].split()
                    if first_columns_appended is False:
                        first_columns_appended = True
                        last_batch_added = True
                        df['atom_ids'] = mos_tmp[0][1:]
                        df['species'] = mos_tmp[1][1:]
                        df['orbital'] = mos_tmp[2][1:]
                        for j in range(3, len(mos_tmp)):
                            df[mos_tmp[j][0]] = mos_tmp[j][1:]
                        mos_tmp = [[] for _ in range(len(line))]
                        for j, n_mo in enumerate(line):
                            mos_tmp[j].append(int(n_mo))
                        i += 1
                    elif first_columns_appended is None:
                        last_batch_added = True
                        mos_tmp = [[] for j in range(len(line) + 3)]
                        mos_tmp[0].append('')
                        mos_tmp[1].append('')
                        mos_tmp[2].append('')
                        for j, n_mo in enumerate(line):
                            mos_tmp[3 + j].append(int(n_mo))
                        first_columns_appended = False
                        i += 1
                    elif first_columns_appended is True:
                        last_batch_added = True
                        #df = pd.concat([df, pd.DataFrame(mos_tmp)], axis=1)
                        for j in range(len(mos_tmp)):
                            df[mos_tmp[j][0]] = mos_tmp[j][1:]
                        mos_tmp = [[] for _ in range(len(line))]
                        for j, n_mo in enumerate(line):
                            mos_tmp[j].append(int(n_mo))
                        i += 1
                else:
                    i += 1

            if not last_batch_added:
                # df = pd.concat([df, pd.DataFrame(mos_tmp)], axis=1)
                for j in range(len(mos_tmp)):
                    df[mos_tmp[j][0]] = mos_tmp[j][1:]

            mos_arr.append(df)

        return SCFLog(np.array(eigs), np.array(occs), mos_arr)


if __name__ == '__main__':
    #from io_data.universal import Cube
    OO = SCFLog.from_file('E:/Setup/_Science/Orca_5.0.1/test.scf.log')
    #eigs = OO.eigenvalues[:13]
    #for i in range(13):
    #    cube = Cube.from_file(f'E:/Setup/_Science/Orca_5.0.1/test.mo{i}a.cube')
    #    print(i, np.nonzero(np.sum(cube.assign_top_n_data_to_atoms(50, 1), axis=-1)))

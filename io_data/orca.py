import numpy as np
import pandas as pd
import re
from monty.re import regrep
from tqdm import tqdm
from .universal import Xyz
from nptyping import NDArray, Shape, Number


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


class XyzTrajectory:
    def __init__(self,
                 first_xyz: Xyz,
                 trajectory: NDArray[Shape['Nsteps, Natoms, 3'], Number],
                 energies_pot: NDArray[Shape['Nsteps'], Number]):
        self.first_xyz = first_xyz
        self.trajectory = trajectory
        self.energies_pot = energies_pot

    @staticmethod
    def from_file(filepath):
        first_xyz = Xyz.from_file(filepath)

        trajectory = []
        energies_pot = []
        with open(filepath, 'rt') as file:
            while True:
                try:
                    natoms = int(file.readline().strip())
                except:
                    break
                line = file.readline()
                energies_pot.append(float(line.split()[8].split('=')[1]))

                coords = np.zeros((natoms, 3))
                for i in range(natoms):
                    line = file.readline().split()
                    coords[i] = [float(j) for j in line[1:]]
                trajectory.append(coords)

            return XyzTrajectory(first_xyz, np.array(trajectory), np.array(energies_pot))

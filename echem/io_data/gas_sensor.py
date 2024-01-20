import numpy as np
from typing import Union, List, Iterable
from monty.re import regrep
import re
from echem.io_data.ddec import AtomicNetCharges




class GasSensor:
    def __init__(self):
        pass

    @staticmethod
    def read_OUTCAR(filepath) -> float:
        """
        This function reads your OUTCAR file to get the final Energy of the system.
        """

        file = open(filepath, 'r')
        data = file.readlines()
        file.close()

        patterns = {'energy_ionic': r'free  energy\s+TOTEN\s+=\s+(.\d+\.\d+)\s+eV'}
        matches = regrep(filepath, patterns)

        end_energy = np.array([float(i[0][0]) for i in matches['energy_ionic']])
        
        return end_energy[-1]

    @staticmethod   
    def sort_DDEC_output(filepath, k: int) -> list[int]:
        """
        This function sorts atoms in your system to get atoms related to your molecule
        k - number of atoms consisting in your molecule
        """

        z_coords = {}

        idx = 1

        with open(filepath, 'r') as file:
            while True:
                line = file.readline()
                if re.search('\sChargemol', line) is not None:
                    break

                list_of_substrings = line.split('    ')
                if re.search('^\s|^j', list_of_substrings[0]) is None:
                    z_coords[idx] = list_of_substrings[-2]
                    idx += 1
                    continue

        sorted_z_coords = dict(sorted(z_coords.items(), key = lambda item: item[1], reverse = True))

        result = []
        counter = 0
        for item in sorted_z_coords:
            if counter < k:
                result.append(item)
                counter += 1
            else:
                break

        return result

    @staticmethod   
    def get_chrg_mol(filepath, k: int or list[int]) -> float:
        """
        This function can help you to get the molecule charge.
        filepath - DDEC6 output file: DDEC_even_tempered_net_atomic_charges.xyz
        k - the number of atoms, consisting in molecule. Besides, your can write the ordinal number of an atom in your molecule.
        """

        atomic_charges = AtomicNetCharges.from_file(filepath)
        net_charges = atomic_charges.net_charges


        if type(k) == int:
            chrg_molecule = 0       
            targets = GasSensor.sort_DDEC_output(filepath, k)
            for i in targets:
                chrg_molecule += net_charges[i - 1]
            return chrg_molecule
        elif isinstance(k, list):
            targets = k
            chrg_molecule = 0       
            for i in targets:
                chrg_molecule += net_charges[i - 1]
            return chrg_molecule
            
    @staticmethod       
    def get_Ead(filepath, E_surface, E_molecule) -> float:
        """ 
        This function can help you to get the adsorption energy, using energy of whole system, energy of the surface and energy of your         molecule.
        filepath - your OUTCAR file obtained as a result of VASP optimization.
        """
        E_system = GasSensor.read_OUTCAR(filepath)
        E_ad = E_system - E_surface - E_molecule
        #print(E_ad) 
        return E_ad
    
    @staticmethod   
    def get_energy_in_meV(filepath):
        """ 
        This function can help you to get energies in meV.
        filepath - your .txt file, consisting of energies in eV.
        """
        X = np.genfromtxt(filepath)
        X_new = []
        for i in X:
            X_new = X * 1000
        return f'Your energies in meV: {X_new}'


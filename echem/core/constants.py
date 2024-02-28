ElemNum2Name = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
                11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P ', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K ', 20: 'Ca',
                21: 'Sc', 22: 'Ti', 23: 'V ', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
                31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y ', 40: 'Zr',
                41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
                51: 'Sb', 52: 'Te', 53: 'I ', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
                61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
                71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W ', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
                81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
                91: 'Pa', 92: 'U ', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
                101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt',
                110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'}

ElemName2Num = {v: k for k, v in zip(ElemNum2Name.keys(), ElemNum2Name.values())}

Bohr2Angstrom = 0.529177
Angstrom2Bohr = 1 / Bohr2Angstrom
Hartree2eV = 27.2114
eV2Hartree = 1 / Hartree2eV
THz2eV = 4.136e-3
eV2THz = 1 / THz2eV
amu2kg = 1.6605402e-27

PLANCK_CONSTANT = 4.135667662e-15  # Planck's constant in eV*s
BOLTZMANN_CONSTANT = 8.617333262145e-5  # Boltzmann's constant in eV/K
ELEM_CHARGE = 1.60217662e-19  # Elementary charge in Coulombs
BOHR_RADIUS = 1.88973 #TODO Check

Bader_radii_Bohr = {'H': 2.88, 'He': 2.5, 'Li': 4.18, 'Be': 4.17, 'B': 3.91, 'C': 3.62, 'N': 3.35, 'O': 3.18,
                    'F': 3.03, 'Ne': 2.89, 'Na': 4.25, 'Mg': 4.6, 'Al': 4.61, 'Si': 4.45, 'P': 4.23, 'S': 4.07,
                    'Cl': 3.91, 'Ar': 3.72, 'Li_plus': 1.82, 'Na_plus': 2.47, 'F_minus': 3.49, 'Cl_minus': 4.36}

IDSCRF_radii_Angstrom = {'H': 1.77, 'He': 1.49, 'Li': 2.22, 'Be': 2.19, 'B': 2.38, 'C': 2.22, 'N': 2.05, 'O': 1.87,
                         'F': 1.89, 'Ne': 1.74}

# Colorblindness-friendly colors from https://github.com/mpetroff/accessible-color-cycles
cbf6 = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
cbf8 = ["#1845fb", "#ff5e02", "#c91f16", "#c849a9", "#adad7d", "#86c8dd", "#578dff", "#656364"]
cbf10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]

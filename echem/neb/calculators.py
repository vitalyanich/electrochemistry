import subprocess
import tempfile
import re
import numpy as np
from ase.calculators.interface import Calculator
from typing import Literal
from echem.core.constants import Hartree2eV, Angstrom2Bohr, Bohr2Angstrom
from pathlib import Path


def shell(cmd) -> str:
    '''
    Run shell command and return output as a string
    '''
    return subprocess.check_output(cmd, shell=True)


# Atomistic Simulation Environment (ASE) calculator interface for JDFTx
# See http://jdftx.org for JDFTx and https://wiki.fysik.dtu.dk/ase/ for ASE
# Authors: Deniz Gunceler, Ravishankar Sundararaman
# Modified: Vitaliy Kislenko
class JDFTx(Calculator):
    def __init__(self,
                 path_jdftx_executable: str | Path,
                 path_pseudopotentials: str | Path,
                 pseudoSet: Literal['SG15', 'GBRV', 'GBRV-pbe', 'GBRV-lda', 'GBRV-pbesol'] = 'GBRV',
                 commands: dict[str, str | int | float] | list[tuple[str, str | int | float]] = None,
                 jdftx_prefix: str = 'jdft',
                 output_name: str = 'output.out'):
        pseudoSetMap = {'SG15':         'SG15/$ID_ONCV_PBE.upf',
                        'GBRV':         'GBRV/$ID_pbe.uspp',
                        'GBRV-pbe':     'GBRV/$ID_pbe.uspp',
                        'GBRV-lda':     'GBRV/$ID_lda.uspp',
                        'GBRV-pbesol':  'GBRV/$ID_pbesol.uspp'}

        if isinstance(path_jdftx_executable, str):
            self.path_jdftx_executable = Path(path_jdftx_executable)
        else:
            self.path_jdftx_executable = path_jdftx_executable
        if isinstance(path_pseudopotentials, str):
            self.path_pseudopotentials = Path(path_pseudopotentials)
        else:
            self.path_pseudopotentials = path_pseudopotentials

        self.jdftx_prefix = jdftx_prefix
        self.output_name = output_name

        if pseudoSet in pseudoSetMap:
            self.pseudoSetCmd = 'ion-species ' + pseudoSetMap[pseudoSet]
        else:
            self.pseudoSetCmd = ''

        self.acceptableCommands = {'electronic-SCF'}
        template = str(shell(f'{self.path_jdftx_executable} -t'))
        for match in re.findall(r"# (\S+) ", template):
            self.acceptableCommands.add(match)

        self.input = [('dump-name', f'{self.jdftx_prefix}.$VAR'),
                      ('initial-state', f'{self.jdftx_prefix}.$VAR')]

        if isinstance(commands, dict):
            commands = commands.items()
        elif commands is None:
            commands = []
        for cmd, v in commands:
            self.addCommand(cmd, v)

        # Accepted pseudopotential formats
        self.pseudopotentials = ['fhi', 'uspp', 'upf']

        # Current results
        self.E = None
        self.forces = None

        # History
        self.lastAtoms = None
        self.lastInput = None

        # k-points
        self.kpoints = None

        # Dumps
        self.dumps = []
        self.addDump("End", "State")
        self.addDump("End", "Forces")
        self.addDump("End", "Ecomponents")

        # Run directory:
        self.runDir = tempfile.mkdtemp()
        print('Set up JDFTx calculator with run files in \'' + self.runDir + '\'')

    def validCommand(self, command) -> bool:
        """ Checks whether the input string is a valid jdftx command \nby comparing to the input template (jdft -t)"""
        if (type(command) != str):
            raise IOError('Please enter a string as the name of the command!\n')
        return command in self.acceptableCommands

    def addCommand(self, cmd, v) -> None:
        if not self.validCommand(cmd):
            raise IOError(f'{cmd} is not a valid JDFTx command!\n'
                          'Look at the input file template (jdftx -t) for a list of commands.')
        self.input.append((cmd, v))

    def addDump(self, when, what) -> None:
        self.dumps.append((when, what))

    def addKPoint(self, pt, w=1) -> None:
        b1, b2, b3 = pt
        if self.kpoints is None:
            self.kpoints = []
        self.kpoints.append((b1, b2, b3, w))

    def __readEnergy(self,
                     filepath: str | Path) -> float:
        Efinal = None
        for line in open(filepath):
            tokens = line.split()
            if len(tokens) == 3:
                Efinal = float(tokens[2])
        if Efinal is None:
            raise IOError('Error: Energy not found.')
        return Efinal * Hartree2eV  # Return energy from final line (Etot, F or G)

    def __readForces(self,
                     filepath: str | Path) -> np.array:
        idxMap = {}
        symbolList = self.lastAtoms.get_chemical_symbols()
        for i, symbol in enumerate(symbolList):
            if symbol not in idxMap:
                idxMap[symbol] = []
            idxMap[symbol].append(i)
        forces = [0] * len(symbolList)
        for line in open(filepath):
            if line.startswith('force '):
                tokens = line.split()
                idx = idxMap[tokens[1]].pop(0)  # tokens[1] is chemical symbol
                forces[idx] = [float(word) for word in tokens[2:5]]  # tokens[2:5]: force components

        if len(forces) == 0:
            raise IOError('Error: Forces not found.')
        return (Hartree2eV / Bohr2Angstrom) * np.array(forces)

    def calculation_required(self, atoms, quantities) -> bool:
        if (self.E is None) or (self.forces is None):
            return True
        if (self.lastAtoms != atoms) or (self.input != self.lastInput):
            return True
        return False

    def get_forces(self, atoms) -> np.array:
        if self.calculation_required(atoms, None):
            self.update(atoms)
        return self.forces

    def get_potential_energy(self, atoms, force_consistent=False):
        if self.calculation_required(atoms, None):
            self.update(atoms)
        return self.E

#    def get_stress(self, atoms):
#        """Since the stress calculation is not implemented in JDFTx, function returns zero array"""
#        return np.zeros((3, 3))

    def update(self, atoms):
        self.runJDFTx(self.constructInput(atoms))

    def runJDFTx(self, inputfile):
        """ Runs a JDFTx calculation """
        file = open(self.runDir + 'input.in', 'w')
        file.write(inputfile)
        file.close()

        shell(f'cd {self.runDir} && {self.path_jdftx_executable} -i input.in -o {self.output_name}')

        self.E = self.__readEnergy(f'{self.runDir}/Ecomponents')
        self.forces = self.__readForces(f'{self.runDir}/force')

    def constructInput(self, atoms) -> str:
        """Constructs a JDFTx input string using the input atoms and the input file arguments (kwargs) in self.input"""
        inputfile = ''

        R = atoms.get_cell() * Angstrom2Bohr
        inputfile += 'lattice \\\n'
        for i in range(3):
            for j in range(3):
                inputfile += '%f  ' % (R[j, i])
            if i != 2:
                inputfile += '\\'
            inputfile += '\n'

        inputfile += "".join(["dump %s %s\n" % (when, what) for when, what in self.dumps])

        inputfile += '\n'
        for cmd, v in self.input:
            inputfile += '%s %s\n' % (cmd, str(v))

        if self.kpoints:
            for pt in self.kpoints:
                inputfile += 'kpoint %.8f %.8f %.8f %.14f\n' % pt

        coords = [x * Bohr2Angstrom for x in list(atoms.get_positions())]
        species = atoms.get_chemical_symbols()
        inputfile += '\ncoords-type cartesian\n'
        for i in range(len(coords)):
            inputfile += 'ion %s %f %f %f \t 1\n' % (species[i], coords[i][0], coords[i][1], coords[i][2])

        inputfile += '\n'
        if not (self.path_pseudopotentials is None):
            added = []  # List of pseudopotential that have already been added
            for atom in species:
                if sum([x == atom for x in added]) == 0.:  # Add ion-species command if not already added
                    for filetype in self.pseudopotentials:
                        try:
                            shell('ls %s | grep %s.%s' % (self.path_pseudopotentials, atom, filetype))
                            inputfile += 'ion-species %s/%s.%s\n' % (self.path_pseudopotentials, atom, filetype)
                            added.append(atom)
                            break
                        except:
                            pass
        inputfile += self.pseudoSetCmd + '\n'

        inputfile += '\ncoulomb-interaction '
        pbc = list(atoms.get_pbc())
        if sum(pbc) == 3:
            inputfile += 'periodic\n'
        elif sum(pbc) == 0:
            inputfile += 'isolated\n'
        elif sum(pbc) == 1:
            inputfile += 'wire %i%i%i\n' % (pbc[0], pbc[1], pbc[2])
        elif sum(pbc) == 2:
            inputfile += 'slab %i%i%i\n' % (not pbc[0], not pbc[1], not pbc[2])
        # --- add truncation center:
        if sum(pbc) < 3:
            center = np.mean(np.array(coords), axis=0)
            inputfile += 'coulomb-truncation-embed %g %g %g\n' % tuple(center.tolist())

        # Cache this calculation to history
        self.lastAtoms = atoms.copy()
        self.lastInput = list(self.input)

        return inputfile
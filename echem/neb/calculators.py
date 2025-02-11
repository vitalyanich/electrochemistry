from __future__ import annotations
import tempfile
import numpy as np
from ase.calculators.calculator import Calculator
from echem.core.constants import Hartree2eV, Angstrom2Bohr, Bohr2Angstrom
from echem.core.useful_funcs import shell
from pathlib import Path
import logging


# Atomistic Simulation Environment (ASE) calculator interface for JDFTx
# See http://jdftx.org for JDFTx and https://wiki.fysik.dtu.dk/ase/ for ASE
# Authors: Deniz Gunceler, Ravishankar Sundararaman
# Modified: Vitaliy Kislenko
class JDFTx(Calculator):
    def __init__(self,
                 path_jdftx_executable: str | Path,
                 path_rundir: str | Path | None = None,
                 commands: list[tuple[str, str]] = None,
                 jdftx_prefix: str = 'jdft',
                 output_name: str = 'output.out'):

        self.logger = logging.getLogger(self.__class__.__name__ + ':')

        if isinstance(path_jdftx_executable, str):
            self.path_jdftx_executable = Path(path_jdftx_executable)
        else:
            self.path_jdftx_executable = path_jdftx_executable

        if isinstance(path_rundir, str):
            self.path_rundir = Path(path_rundir)
        elif isinstance(path_rundir, Path):
            self.path_rundir = path_rundir
        elif path_rundir is None:
            self.path_rundir = Path(tempfile.mkdtemp())
        else:
            raise ValueError(f'path_rundir should be str or Path or None, however you set {path_rundir=}'
                             f' with type {type(path_rundir)}')

        self.jdftx_prefix = jdftx_prefix
        self.output_name = output_name

        self.dumps = []
        self.input = [('dump-name', f'{self.jdftx_prefix}.$VAR'),
                      ('initial-state', f'{self.jdftx_prefix}.$VAR')]

        if commands is not None:
            for com, val in commands:
                if com == 'dump-name':
                    self.logger.debug(f'{self.path_rundir} You set \'dump-name\' command in commands = \'{val}\', '
                                      f'however it will be replaced with \'{self.jdftx_prefix}.$VAR\'')
                elif com == 'initial-state':
                    self.logger.debug(f'{self.path_rundir} You set \'initial-state\' command in commands = \'{val}\', '
                                      f'however it will be replaced with \'{self.jdftx_prefix}.$VAR\'')
                elif com == 'coords-type':
                    self.logger.debug(f'{self.path_rundir} You set \'coords-type\' command in commands = \'{val}\', '
                                      f'however it will be replaced with \'cartesian\'')
                elif com == 'include':
                    self.logger.debug(f'{self.path_rundir} \'include\' command is not supported, ignore it')
                elif com == 'coulomb-interaction':
                    self.logger.debug(f'{self.path_rundir} \'coulomb-interaction\' command will be replaced in accordance with ase atoms')
                elif com == 'dump':
                    self.addDump(val.split()[0], val.split()[1])
                else:
                    self.addCommand(com, val)

        if ('End', 'State') not in self.dumps:
            self.addDump("End", "State")
        if ('End', 'Forces') not in self.dumps:
            self.addDump("End", "Forces")
        if ('End', 'Ecomponents') not in self.dumps:
            self.addDump("End", "Ecomponents")

        # Current results
        self.E = None
        self.forces = None

        # History
        self.lastAtoms = None
        self.lastInput = None

        self.global_step = None

        self.logger.debug(f'Successfully initialized JDFTx calculator in \'{self.path_rundir}\'')

    def validCommand(self, command) -> bool:
        """Checks whether the input string is a valid jdftx command by comparing to the input template (jdft -t)"""
        if type(command) != str:
            raise IOError('Please enter a string as the name of the command!\n')
        return True

    def addCommand(self, cmd, v) -> None:
        if not self.validCommand(cmd):
            raise IOError(f'{cmd} is not a valid JDFTx command!\n'
                          'Look at the input file template (jdftx -t) for a list of commands.')
        self.input.append((cmd, v))

    def addDump(self, when, what) -> None:
        self.dumps.append((when, what))

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

    def update(self, atoms):
        self.runJDFTx(self.constructInput(atoms))

    def runJDFTx(self, inputfile):
        """ Runs a JDFTx calculation """
        file = open(self.path_rundir / 'input.in', 'w')
        file.write(inputfile)
        file.close()

        if self.global_step is not None:
            self.logger.info(f'Step: {self.global_step:2}. Run in {self.path_rundir}')
        else:
            self.logger.info(f'Run in {self.path_rundir}')

        shell(f'cd {self.path_rundir} && srun {self.path_jdftx_executable} -i input.in -o {self.output_name}')

        self.E = self.__readEnergy(self.path_rundir / f'{self.jdftx_prefix}.Ecomponents')

        if self.global_step is not None:
            self.logger.debug(f'Step: {self.global_step}. E = {self.E:.4f}')
        else:
            self.logger.debug(f'E = {self.E:.4f}')

        self.forces = self.__readForces(self.path_rundir / f'{self.jdftx_prefix}.force')

    def constructInput(self, atoms) -> str:
        """Constructs a JDFTx input string using the input atoms and the input file arguments (kwargs) in self.input"""
        inputfile = ''

        lattice = atoms.get_cell() * Angstrom2Bohr
        inputfile += 'lattice \\\n'
        for i in range(3):
            for j in range(3):
                inputfile += '%f  ' % (lattice[j, i])
            if i != 2:
                inputfile += '\\'
            inputfile += '\n'

        inputfile += '\n'

        inputfile += "".join(["dump %s %s\n" % (when, what) for when, what in self.dumps])

        inputfile += '\n'
        for cmd, v in self.input:
            inputfile += '%s %s\n' % (cmd, str(v))

        coords = [x * Angstrom2Bohr for x in list(atoms.get_positions())]
        species = atoms.get_chemical_symbols()
        inputfile += '\ncoords-type cartesian\n'
        for i in range(len(coords)):
            inputfile += 'ion %s %f %f %f \t 1\n' % (species[i], coords[i][0], coords[i][1], coords[i][2])

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

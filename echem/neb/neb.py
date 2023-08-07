from ase.neb import NEB
from ase.optimize import FIRE
from ase.io import read
from echem.neb.calculators import JDFTx
from echem.io_data.jdftx import Ionpos, Lattice, Input
from pathlib import Path
import os
import logging
logging.basicConfig(level=logging.INFO, filename="logfile_NEB.log",
                    filemode="w", format="%(asctime)s %(levelname)s %(message)s")


class NEB_JDFTx:
    def __init__(self,
                 init_ionpos: Ionpos,
                 init_lattice: Lattice,
                 final_ionpos: Ionpos,
                 final_lattice: Lattice,
                 nimages: int,
                 jdftx_input: Input,
                 path_jdftx_executable: str | Path,
                 jdftx_prefix: str = 'jdft',
                 output_name: str = 'output.out',
                 cNEB: bool = True,
                 spring_constant: float = 0.1):
        self.init_ionpos = init_ionpos,
        self.init_lattice = init_lattice,
        self.final_ionpos = final_ionpos,
        self.final_lattice = final_lattice,
        self.nimages = nimages,
        self.jdftx_input = jdftx_input,

        if isinstance(path_jdftx_executable, str):
            self.path_jdftx_executable = Path(path_jdftx_executable)
        else:
            self.path_jdftx_executable = path_jdftx_executable

        self.path_rundir = Path(os.getcwd())

        self.jdftx_prefix = jdftx_prefix
        self.output_name = output_name
        self.cNEB = cNEB
        self.spring_constant = spring_constant

        self.optimizer = None

    def prepare(self):
        init_poscar = self.init_ionpos.convert('vasp', self.init_lattice)
        init_poscar.to_file('POSCAR_init.vasp')
        initial = read('POSCAR_init.vasp', format='vasp')
        final_poscar = self.final_ionpos.convert('vasp', self.final_lattice)
        final_poscar.to_file('POSCAR_final.vasp')
        final = read('POSCAR_final.vasp', format='vasp')

        images = [initial]
        images += [initial.copy() for _ in range(self.nimages)]
        images += [final]

        neb = NEB(images,  k=self.spring_constant, climb=self.cNEB)
        neb.interpolate()

        length = len(str(self.nimages - 1))
        for i in range(self.nimages):
            folder = str(i).zfill(length)
            if not os.path.exists(folder):
                os.mkdir(folder)

        for i, image in enumerate(images[1:-1]):
            image.calc = JDFTx(self.path_jdftx_executable,
                               path_rundir=self.path_rundir / str(i).zfill(length),
                               commands=self.jdftx_input.commands)

        self.optimizer = FIRE(neb, trajectory='NEB_trajectory.traj', logfile='logfile_optimizer')

    def run(self):
        self.prepare()
        self.optimizer.run(fmax=0.04)

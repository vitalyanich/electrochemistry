from __future__ import annotations
from ase.neb import NEB, NEBOptimizer
from ase.optimize import FIRE
from ase.io import read, write
from echem.neb.calculators import JDFTx
from echem.io_data.jdftx import Ionpos, Lattice, Input
from pathlib import Path
import logging
from echem.neb.autoneb import AutoNEB
from typing import Literal
logging.basicConfig(level=logging.INFO, filename="logfile_NEB.log",
                    filemode="a", format="%(asctime)s %(levelname)s %(message)s")


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
                 spring_constant: float = 0.1,
                 interpolation_method: Literal['linear', 'idpp'] = 'linear'):

        self.init_ionpos = init_ionpos
        self.init_lattice = init_lattice
        self.final_ionpos = final_ionpos
        self.final_lattice = final_lattice
        self.nimages = nimages
        self.jdftx_input = jdftx_input

        if isinstance(path_jdftx_executable, str):
            self.path_jdftx_executable = Path(path_jdftx_executable)
        else:
            self.path_jdftx_executable = path_jdftx_executable

        self.path_rundir = Path.cwd()

        self.jdftx_prefix = jdftx_prefix
        self.output_name = output_name
        self.cNEB = cNEB
        self.spring_constant = spring_constant
        self.interpolation_method = interpolation_method

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

        neb = NEB(images,
                  k=self.spring_constant,
                  climb=self.cNEB)
        neb.interpolate(method=self.interpolation_method)

        length = len(str(self.nimages - 1))
        for i in range(self.nimages):
            folder = Path(str(i).zfill(length))
            folder.mkdir(exist_ok=True)

        for i, image in enumerate(images[1:-1]):
            image.calc = JDFTx(self.path_jdftx_executable,
                               path_rundir=self.path_rundir / str(i).zfill(length),
                               commands=self.jdftx_input.commands)
        self.optimizer = NEBOptimizer(neb=neb,
                                      method='ODE',
                                      trajectory='NEB_trajectory.traj',
                                      logfile='logfile_NEBOptimizer.log')

    def run(self, fmax=0.05):
        self.prepare()
        self.optimizer.run(fmax=fmax)


class AutoNEB_JDFTx:
    """
    Class for running AutoNEB with JDFTx calculator

    Parameters:

    prefix: string or Path
        path to folder with initial files. Basically could be os.getcwd()
        In this folder required:
            1) init.vasp file with initial configuration
            2) final.vasp file with final configuration
            3) in file with JDFTx calculation parameters
    path_jdftx_executable: string or Path
        path to jdftx executable
    n_start: int
        Starting number of images between starting and final for NEB
    n_max: int
        Maximum number of images, including starting and final
    climb: boolean
        Whether it is necessary to use cNEB or not
    fmax: float or list of floats
        The maximum force along the NEB path
    maxsteps: int
        The maximum number of steps in each NEB relaxation.
        If a list is given the first number of steps is used in the build-up
        and final scan phase;
        the second number of steps is used in the CI step after all images
        have been inserted.
    k: float
        The spring constant along the NEB path
    method: str (see neb.py)
        Choice betweeen three method:
        'aseneb', standard ase NEB implementation
        'improvedtangent', published NEB implementation
        'eb', full spring force implementation (default)
    optimizer: str or object
        Set optimizer for NEB: FIRE, BFGS or NEB
    space_energy_ratio: float
        The preference for new images to be added in a big energy gab
        with a preference around the peak or in the biggest geometric gab.
        A space_energy_ratio set to 1 will only considder geometric gabs
        while one set to 0 will result in only images for energy
        resolution.
    interpolation_method: string
        method for interpolation
    smooth_curve: boolean
    """

    def __init__(self,
                 prefix,
                 path_jdftx_executable,
                 n_start=3,
                 n_simul=3,
                 n_max=10,
                 climb=True,
                 fmax=0.05,
                 maxsteps=100,
                 k=0.1,
                 method='eb',
                 optimizer='FIRE',
                 space_energy_ratio=0.5,
                 interpolation_method='idpp',
                 smooth_curve=False):
        self.path_jdftx_executable = path_jdftx_executable
        self.prefix = Path(prefix)
        self.n_start = n_start
        self.commands = Input.from_file(Path(prefix) / 'in').commands
        self.interpolation_method = interpolation_method
        self.autoneb = AutoNEB(self.attach_calculators,
                               prefix=prefix,
                               n_simul=n_simul,
                               n_max=n_max,
                               climb=climb,
                               fmax=fmax,
                               maxsteps=maxsteps,
                               k=k,
                               method=method,
                               space_energy_ratio=space_energy_ratio,
                               world=None, parallel=False, smooth_curve=smooth_curve,
                               interpolate_method=interpolation_method, optimizer=optimizer)

    def prepare(self):
        initial = read(self.prefix / 'init.vasp', format='vasp')
        final = read(self.prefix / 'final.vasp', format='vasp')
        images = [initial]
        images += [initial.copy() for _ in range(self.n_start)]
        images += [final]
        neb = NEB(images)
        neb.interpolate(method=self.interpolation_method)
        for i, image in enumerate(images):
            write(self.prefix / f'{i:03d}.traj', image, format='traj')

    def attach_calculators(self, images, indexes, iteration):
        for image, index in zip(images, indexes):
            path_rundir = self.autoneb.iter_folder / f'iter_{iteration}' / str(index)
            path_rundir.mkdir(exist_ok=True)
            image.calc = JDFTx(self.path_jdftx_executable,
                               path_rundir=path_rundir,
                               commands=self.commands)

    def run(self):
        self.prepare()
        self.autoneb.run()
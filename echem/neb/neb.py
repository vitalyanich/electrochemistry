from __future__ import annotations
from ase.neb import NEB, NEBOptimizer
from ase.optimize import FIRE
from ase.io import read, write
from echem.neb.calculators import JDFTx
from echem.io_data.jdftx import Ionpos, Lattice, Input
from pathlib import Path
import logging
import os
from ase.io.trajectory import Trajectory
from echem.neb.autoneb import AutoNEB
from typing import Literal
logging.basicConfig(level=logging.INFO, filename="logfile_NEB.log",
                    filemode="a", format="%(asctime)s %(levelname)s %(message)s")


class NEB_JDFTx:
    def __init__(self,
                 path_jdftx_executable: str | Path,
                 nimages: int = 5,
                 output_name: str = 'output.out',
                 cNEB: bool = True,
                 restart: bool = False,
                 from_vasp: bool = False,
                 spring_constant: float = 5.0,
                 input_format: Literal['vasp', 'jdftx'] = 'jdftx',
                 fmax: float = 0.05,
                 method: str = 'ODE',
                 inp_filename: str = 'in',
                 interpolation_method: Literal['linear', 'idpp'] = 'idpp'):

        self.nimages = nimages

        if isinstance(path_jdftx_executable, str):
            self.path_jdftx_executable = Path(path_jdftx_executable)
        else:
            self.path_jdftx_executable = path_jdftx_executable

        self.path_rundir = Path.cwd()
        self.jdftx_input = Input.from_file(inp_filename)
        self.output_name = output_name
        self.cNEB = cNEB
        self.fmax = fmax
        self.from_vasp = from_vasp
        self.restart = restart
        self.method = method
        self.spring_constant = spring_constant
        self.interpolation_method = interpolation_method
        self.input_format = input_format
        self.optimizer = None

    def prepare(self):
        if not self.restart:
            if self.input_format == 'jdftx':
                init_ionpos = Ionpos.from_file('init.ionpos')
                init_lattice = Lattice.from_file('init.lattice')
                final_ionpos = Ionpos.from_file('final.ionpos')
                final_lattice = Lattice.from_file('final.lattice')
                init_poscar = init_ionpos.convert('vasp', init_lattice)
                init_poscar.to_file('init.vasp')
                final_poscar = final_ionpos.convert('vasp', final_lattice)
                final_poscar.to_file('final.vasp')

            initial = read('init.vasp', format='vasp')
            final = read('final.vasp', format='vasp')

            images = [initial]
            images += [initial.copy() for _ in range(self.nimages)]
            images += [final]

            neb = NEB(images,
                      k=self.spring_constant,
                      climb=self.cNEB)
            neb.interpolate(method=self.interpolation_method)
            for i, image in enumerate(images):
                image.write(f'start_img{i:02d}.vasp', format='vasp')
        else:
            if not self.from_vasp:
                trj = Trajectory('NEB_trajectory.traj')
                n_iter = int(len(trj) / (self.nimages + 2))
                images = []
                for i in range(self.nimages + 2):
                    trj[(n_iter - 1) * self.nimages + i].write(f'start_img{i:02d}.vasp', format='vasp')
            img = read(f'start_img{i:02d}.vasp', format='vasp')
            images.append(img)

            neb = NEB(images,
                      k=self.spring_constant,
                      climb=self.cNEB)

        length = len(str(self.nimages - 1))
        for i in range(self.nimages):
            folder = Path(str(i+1).zfill(length))
            folder.mkdir(exist_ok=True)

        for i, image in enumerate(images[1:-1]):
            image.calc = JDFTx(self.path_jdftx_executable,
                               path_rundir=self.path_rundir / str(i+1).zfill(length),
                               commands=self.jdftx_input.commands)
        self.optimizer = NEBOptimizer(neb=neb,
                                      method=self.method,
                                      trajectory='NEB_trajectory.traj',
                                      logfile='logfile_NEBOptimizer.log')

    def run(self):
        self.prepare()
        self.optimizer.run(fmax=self.fmax)


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
                 restart=False,
                 method='eb',
                 optimizer='FIRE',
                 space_energy_ratio=0.5,
                 interpolation_method='idpp',
                 smooth_curve=False):
        self.restart = restart
        self.path_jdftx_executable = path_jdftx_executable
        self.prefix = Path(prefix)
        self.n_start = n_start
        self.n_max = n_max
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
        if not self.restart:
            initial = read(self.prefix / 'init.vasp', format='vasp')
            final = read(self.prefix / 'final.vasp', format='vasp')
            images = [initial]
            if self.n_start != 0:
                images += [initial.copy() for _ in range(self.n_start)]
            images += [final]
            if self.n_start != 0:
                neb = NEB(images)
                neb.interpolate(method=self.interpolation_method)
            for i, image in enumerate(images):
                image.write(self.prefix / f'{i:03d}.traj', format='traj')
                image.write(self.prefix / f'{i:03d}.vasp', format='vasp')
        else:
            index_exists = [i for i in range(self.n_max) if
                            os.path.isfile(self.prefix / f'{i:03d}.traj')]
            for i in index_exists:
                image = Trajectory(self.prefix / f'{i:03d}.traj')
                image[-1].write(self.prefix / f'{i:03d}.vasp', format='vasp')
                img = read(self.prefix / f'{i:03d}.vasp', format='vasp')
                img.write(self.prefix / f'{i:03d}.traj', format='traj')

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
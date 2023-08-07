from ase.neb import NEB
from ase.optimize import FIRE
from ase.io import read
from echem.neb.calculators import JDFTx
from echem.io_data.jdftx import Ionpos, Lattice, Input
from pathlib import Path
from typing import Literal
from ase.autoneb import AutoNEB
import os


class NEB_JDFTx:
    def __init__(self,
                 init_ionpos: Ionpos,
                 init_lattice: Lattice,
                 final_ionpos: Ionpos,
                 final_lattice: Lattice,
                 nimages: int,
                 jdftx_input: Input,
                 path_jdftx_executable: str | Path,
                 folderpath_pseudopots: str | Path | None = None,
                 pseudo_set: Literal['SG15', 'GBRV', 'GBRV-pbe', 'GBRV-lda', 'GBRV-pbesol'] = 'GBRV',
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

        if isinstance(folderpath_pseudopots, str):
            self.folderpath_pseudopots = Path(folderpath_pseudopots)
        else:
            self.folderpath_pseudopots = folderpath_pseudopots

        self.path_rundir = Path(os.getcwd())

        self.pseudo_set = pseudo_set
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

        commands = ...

        for i, image in enumerate(images[1:-1]):
            image.calc = JDFTx(self.path_jdftx_executable,
                               path_rundir=self.path_rundir / i,
                               folderpath_pseudopots=self.folderpath_pseudopots,
                               pseudoSet=self.pseudo_set,
                               commands=commands)

        self.optimizer = FIRE(neb, trajectory='NEB_trajectory.traj', logfile='logfile_optimizer')

    def run(self):
        self.prepare()
        self.optimizer.run(fmax=0.04)


class AutoNEB_JDFTx:
    def __init__(self, commands,
                 prefix,
                 n_max,
                 path_jdftx_executable,
                 climb=True,
                 fmax=0.05,
                 maxsteps=100,
                 k=0.1,
                 method='eb',
                 optimizer=FIRE,
                 space_energy_ratio=0.5,
                 interpolation_method='idpp'):
        self.path_jdftx_executable = path_jdftx_executable
        self.commands = commands
        self.autoneb = AutoNEB(self.attach_calculators,
                               prefix=prefix,
                               n_simul=1,
                               n_max=n_max,
                               climb=climb,
                               fmax=fmax,
                               maxsteps=maxsteps,
                               k=k,
                               method=method,
                               optimizer=optimizer,
                               space_energy_ratio=space_energy_ratio,
                               world=None, parallel=True, smooth_curve=smooth_curve,
                               interpolate_method=interpolation_method)

    def attach_calculators(self, images):
        for i, image in enumerate(images[1:-1]):
            path_rundir = self.autoneb.iter_folder / i
            path_rundir.mkdir(exist_ok=True)
            image.calc = JDFTx(self.path_jdftx_executable,
                               path_rundir=path_rundir,
                               commands=self.commands)
    def run(self):
        self.autoneb.run()
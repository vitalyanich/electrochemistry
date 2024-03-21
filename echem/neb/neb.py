from __future__ import annotations
from ase.neb import NEB
from ase.optimize.sciopt import OptimizerConvergenceError
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.io import read
from echem.neb.calculators import JDFTx
from echem.neb.autoneb import AutoNEB
from echem.io_data.jdftx import Ionpos, Lattice, Input
from pathlib import Path
import numpy as np
import logging
import os
from typing import Literal
logging.basicConfig(level=logging.INFO, filename="logfile_NEB.log",
                    filemode="a", format="%(asctime)s %(levelname)s %(message)s")


class NEBOptimizer:
    def __init__(self,
                 neb: NEB,
                 trajectory_filepath: str | Path | None = None,
                 append_trajectory: bool = False):

        self.neb = neb
        if trajectory_filepath is not None:
            if append_trajectory:
                self.trj_writer = TrajectoryWriter(trajectory_filepath, mode='a')
            else:
                self.trj_writer = TrajectoryWriter(trajectory_filepath, mode='w')
        else:
            self.trj_writer = None

    def converged(self, fmax):
        return self.neb.get_residual() <= fmax

    def update_positions(self, X):
        positions = X.reshape((self.neb.nimages - 2) * self.neb.natoms, 3)
        self.neb.set_positions(positions)

    def get_forces(self):
        return self.neb.get_forces().reshape(-1)

    def dump_trajectory(self):
        if self.trj_writer is not None:
            for image in self.neb.nimages:
                self.trj_writer.write(image)

    def dump_positions_vasp(self):
        length = len(str(self.neb.nimages + 1))
        for i, image in enumerate(self.neb.nimages):
            image.write(f'last_img{str(i).zfill(length)}.vasp', format='vasp')

    def run_static(self,
                   fmax: float = 0.1,
                   max_steps: int = 100,
                   alpha: float = 0.1):

        if max_steps < 1:
            raise ValueError('max_steps must be greater or equal than one')

        X = self.neb.get_positions().reshape(-1)

        for i in range(max_steps):
            self.dump_trajectory()
            self.dump_positions_vasp()

            F = self.get_forces()

            energies = []
            for image in self.neb.images:
                energies.append(np.round(image.calc.E, 4))
            logging.info(f'Step: {i}. Energies: {energies}')

            R = self.neb.get_residual()
            if R <= fmax:
                logging.info(f"static: terminates successfully after {i} iterations. "
                             f"Residual R = {R:.3f}")
                return True
            else:
                logging.info(f'Step: {i}. Residual: {R:.3f}')
            X += alpha * F
            self.update_positions(X)

        logging.info(f'static: convergence was not achieved after {max_steps} iterations. '
                     f'Residual R = {R:.3f} > {fmax}')
        return False

    def run_ode(self,
                fmax: float = 0.1,
                max_steps: int = 100,
                C1: float = 1e-2,
                C2: float = 2.0,
                extrapolation_scheme: Literal[1, 2, 3] = 3,
                h: float | None = None,
                h_min: float = 1e-10,
                R_max: float = 1e3,
                rtol: float = 0.1):
        """
            fmax : float
               convergence tolerance for residual force
            max_steps : int
                maximum number of steps
            C1 : float
               sufficient contraction parameter
            C2 : float
               residual growth control (Inf means there is no control)
            extrapolation_scheme : int
               extrapolation style (3 seems the most robust)
            h : float
               initial step size, if None an estimate is used based on ODE12
            h_min : float
               minimal allowed step size
            R_max: float
               terminate if residual exceeds this value
            rtol : float
               relative tolerance
        """

        F = self.get_forces()

        energies = []
        for image in self.neb.images:
            energies.append(image.calc.E)
        logging.info(f'Step: 0. Energies: {energies}')

        R = self.neb.get_residual()  # pick the biggest force

        if R >= R_max:
            logging.info(f"ODE12r: Residual {R:.3f} >= R_max {R_max} at iteration 0")
            raise OptimizerConvergenceError(f"ODE12r: Residual {R:.3f} >= R_max {R_max} at iteration 0")
        else:
            logging.info(f'Step: 0. Residual: {R:.3f}')

        if h is None:
            h = 0.5 * rtol ** 0.5 / R  # Chose a step size based on that force
            h = max(h, h_min)  # Make sure the step size is not too big

        X = self.neb.get_positions().reshape(-1)

        for step in range(1, max_steps):
            X_new = X + h * F  # Pick a new position
            self.update_positions(X_new)
            F_new = self.get_forces()  # Calculate the new forces at this position

            energies = []
            for image in self.neb.images:
                energies.append(np.round(image.calc.E, 4))
            logging.info(f'Step: {step}. Energies: {energies}')

            R_new = self.neb.get_residual()
            logging.info(f'Step: {step}. Residual: {R:.3f}')

            e = 0.5 * h * (F_new - F)  # Estimate the area under the forces curve
            err = np.linalg.norm(e, np.inf)  # Error estimate

            # Accept step if residual decreases sufficiently and/or error acceptable
            accept = ((R_new <= R * (1 - C1 * h)) or ((R_new <= R * C2) and err <= rtol))

            # Pick an extrapolation scheme for the system & find new increment
            y = F - F_new
            if extrapolation_scheme == 1:  # F(xn + h Fp)
                h_ls = h * (F @ y) / (y @ y)
            elif extrapolation_scheme == 2:  # F(Xn + h Fp)
                h_ls = h * (F @ F_new) / (F @ y + 1e-10)
            elif extrapolation_scheme == 3:  # min | F(Xn + h Fp) |
                h_ls = h * (F @ y) / (y @ y + 1e-10)
            else:
                raise ValueError(f'Invalid extrapolation_scheme: {extrapolation_scheme}. Must be 1, 2 or 3')

            if np.isnan(h_ls) or h_ls < h_min:  # Rejects if increment is too small
                h_ls = np.inf

            h_err = h * 0.5 * np.sqrt(rtol / err)

            # Accept the step and do the update
            if accept:
                logging.info(f'The step {step} is accepted')

                X = X_new
                R = R_new
                F = F_new

                self.dump_trajectory()
                self.dump_positions_vasp()

                # We check the residuals again
                if self.converged(fmax):
                    logging.info(f"ODE12r: terminates successfully after {step} iterations. "
                                 f"Residual {R:.3f}")
                    return True

                if R >= R_max:
                    logging.info(f"ODE12r: Residual {R:.3f} is too large")
                    return False

                # Compute a new step size.
                # Based on the extrapolation and some other heuristics
                h = max(0.25 * h, min(4 * h, h_err, h_ls))  # Log steep-size analytic results

                logging.info(f"ODE12r: new step size h = {h}")
                logging.info(f"ODE12r: residual {R=}")
                logging.info(f"ODE12r: {h_ls=}")
                logging.info(f"ODE12r: {h_err=}")

            else:
                logging.info(f'The step {step} is rejected')

                # Compute a new step size.
                h = max(0.1 * h, min(0.25 * h, h_err, h_ls))
                logging.info(f"ODE12r: new step size h = {h}")
                logging.info(f"ODE12r: R_new = {R_new}")
                logging.info(f"ODE12r: R = {R}")

            if abs(h) <= h_min:  # abort if step size is too small
                logging.info(f'ODE12r terminates unsuccessfully. Step size {h=} is too small')

        logging.info(f'ODE12r terminates unsuccessfully after {max_steps} iterations')
        return True


class NEB_JDFTx:
    def __init__(self,
                 path_jdftx_executable: str | Path,
                 nimages: int = 5,
                 input_filepath: str | Path = 'input.in',
                 output_name: str = 'output.out',
                 input_format: Literal['jdftx', 'vasp'] = 'jdftx',
                 cNEB: bool = True,
                 spring_constant: float = 5.0,
                 interpolation_method: Literal['linear', 'idpp'] = 'idpp',
                 restart: Literal[False, 'from_traj', 'from_vasp'] = False):

        if isinstance(path_jdftx_executable, str):
            self.path_jdftx_executable = Path(path_jdftx_executable)
        else:
            self.path_jdftx_executable = path_jdftx_executable

        if isinstance(input_filepath, str):
            input_filepath = Path(input_filepath)
        self.jdftx_input = Input.from_file(input_filepath)

        self.nimages = nimages
        self.path_rundir = Path.cwd()
        self.output_name = output_name
        self.input_format = input_format.lower()
        self.cNEB = cNEB
        self.restart = restart
        self.spring_constant = spring_constant
        self.interpolation_method = interpolation_method.lower()
        self.optimizer = None

    def prepare(self):
        length = len(str(self.nimages + 1))

        if self.restart is False:
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
                image.write(f'start_img{str(i).zfill(length)}.vasp', format='vasp')

        else:
            images = []
            if self.restart == 'from_traj':
                trj = Trajectory('NEB_trajectory.traj')
                n_iter = int(len(trj) / (self.nimages + 2))
                for i in range(self.nimages + 2):
                    trj[(n_iter - 1) * (self.nimages + 2) + i].write(f'start_img{str(i).zfill(length)}.vasp',
                                                                     format='vasp')
                trj.close()

            elif self.restart == 'from_traj' or self.restart == 'from_vasp':
                for i in range(self.nimages + 2):
                    img = read(f'start_img{str(i).zfill(length)}.vasp', format='vasp')
                    images.append(img)

            else:
                raise ValueError(f'restart must be False or \'from_traj\', '
                                 f'or \'from_vasp\' but you set {self.restart=}')

            neb = NEB(images,
                      k=self.spring_constant,
                      climb=self.cNEB)

        for i in range(self.nimages):
            folder = Path(str(i+1).zfill(length))
            folder.mkdir(exist_ok=True)

        for i, image in enumerate(images[1:-1]):
            image.calc = JDFTx(self.path_jdftx_executable,
                               path_rundir=self.path_rundir / str(i+1).zfill(length),
                               commands=self.jdftx_input.commands)
        self.optimizer = NEBOptimizer(neb=neb, trajectory_filepath='NEB_trajectory.traj')

    def run(self,
            fmax: float = 0.1,
            method: Literal['ode', 'static'] = 'ode',
            max_steps: int = 100,
            **kwargs):

        self.prepare()

        if method == 'ode':
            self.optimizer.run_ode(fmax, max_steps)
        elif method == 'static':
            self.optimizer.run_static(fmax, max_steps)
        else:
            raise ValueError(f'Method must be ode or static but you set {method=}')


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

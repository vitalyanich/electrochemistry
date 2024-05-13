from __future__ import annotations
from ase.neb import NEB
from ase.optimize.sciopt import OptimizerConvergenceError
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.io import read
from echem.neb.calculators import JDFTx
from echem.neb.autoneb import AutoNEB
from echem.io_data.jdftx import Ionpos, Lattice, Input
from echem.core.useful_funcs import shell
from pathlib import Path
import numpy as np
import logging
import os
from typing import Literal, Callable
logging.basicConfig(filename='logfile_NEB.log', filemode='a', level=logging.INFO,
                    format="%(asctime)s %(levelname)8s %(name)14s %(message)s",
                    datefmt='%d/%m/%Y %H:%M:%S')


class NEBOptimizer:
    def __init__(self,
                 neb: NEB,
                 trajectory_filepath: str | Path | None = None,
                 append_trajectory: bool = True):

        self.neb = neb
        self.logger = logging.getLogger(self.__class__.__name__ + ':')
        self.E_image_first = None
        self.E_image_last = None

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

    def get_energies(self, first: bool = False, last: bool = False):
        if not first and not last:
            return [image.calc.E for image in self.neb.images[1:-1]]
        elif first and not last:
            return [image.calc.E for image in self.neb.images[:-1]]
        elif not first and last:
            return [image.calc.E for image in self.neb.images[1:]]
        elif first and last:
            return [image.calc.E for image in self.neb.images]

    def dump_trajectory(self):
        if self.trj_writer is not None:
            for image in self.neb.images:
                self.trj_writer.write(image)

    def dump_positions_vasp(self, prefix='last_img'):
        length = len(str(self.neb.nimages + 1))
        for i, image in enumerate(self.neb.images):
            image.write(f'{prefix}_{str(i).zfill(length)}.vasp', format='vasp')

    def set_step_in_calculators(self, step, first: bool = False, last: bool = False):
        if not first and not last:
            for image in self.neb.images[1:-1]:
                image.calc.global_step = step
        elif first and not last:
            for image in self.neb.images[:-1]:
                image.calc.global_step = step
        elif not first and last:
            for image in self.neb.images[1:]:
                image.calc.global_step = step
        elif first and last:
            for image in self.neb.images:
                image.calc.global_step = step

    def run_static(self,
                   fmax: float = 0.1,
                   max_steps: int = 100,
                   alpha: float = 0.02,
                   dE_max: float = None,
                   construct_calc_fn: Callable = None):
        self.logger.info('Static method of optimization was chosen')
        max_new_images_at_step = 1
        min_steps_after_insertion = 3
        steps_after_insertion = 0

        if dE_max is not None:
            self.logger.info(f'AutoNEB with max {dE_max} eV difference between images was set')
            self.logger.info(f'Initial number of images is {self.neb.nimages}, '
                             f'including initial and final images')

        if max_steps < 1:
            raise ValueError('max_steps must be greater or equal than one')

        if dE_max is not None:
            self.set_step_in_calculators(0, first=True, last=True)
            self.E_image_first = self.neb.images[0].get_potential_energy()
            self.E_image_last = self.neb.images[-1].get_potential_energy()

        length_step = len(str(max_steps))
        #X = self.neb.get_positions().reshape(-1)
        for step in range(max_steps):
            self.dump_trajectory()
            self.dump_positions_vasp(prefix=f'Step-{step}-1-')
            if dE_max is not None:
                self.set_step_in_calculators(step, first=True, last=True)
            else:
                self.set_step_in_calculators(step)

            F = self.get_forces()
            self.logger.info(f'Step: {step:{length_step}}. Energies = '
                             f'{[np.round(en, 4) for en in self.get_energies()]}')

            R = self.neb.get_residual()
            if R <= fmax:
                self.logger.info(f'Step: {step:{length_step}}. Optimization terminates successfully. Residual R = {R:.4f}')
                return True
            else:
                self.logger.info(f'Step: {step:{length_step}}. Residual R = {R:.4f}')

            X = self.neb.get_positions().reshape(-1)
            X += alpha * F
            self.update_positions(X)
            self.dump_positions_vasp(prefix=f'Step:-{step}-2-')

            if dE_max is not None:
                energies = self.get_energies(first=True, last=True)
                self.logger.debug(f'Energies raw: {energies}')
                self.logger.info(f'Step: {step:{length_step}}. Energies = '
                                 f'{[np.round(en, 4) for en in energies]}')
                diff = np.abs(np.diff(energies))
                self.logger.debug(f'diff: {diff}')
                idxs = np.where(diff > dE_max)[0]
                self.logger.debug(f'Idxs where diff > dE_max: {idxs}')

                if len(idxs) > max_new_images_at_step:
                    idxs = np.flip(np.argsort(diff))[:max_new_images_at_step]
                    self.logger.debug(f'Images will be added for idxs {idxs} since more than {max_new_images_at_step} '
                                      f'diffs were large than {dE_max} eV')

                if (len(idxs) > 0) and (steps_after_insertion > min_steps_after_insertion):
                    steps_after_insertion = -1
                    for idx in reversed(idxs):
                        self.logger.debug(f'Start working with idx: {idx}')
                        length_prev = len(str(len(self.neb.images) - 1))
                        length_new = len(str(len(self.neb.images)))
                        self.logger.debug(f'{length_prev=} {length_new=}')
                        tmp_images = [self.neb.images[idx].copy(),
                                      self.neb.images[idx].copy(),
                                      self.neb.images[idx + 1].copy()]
                        tmp_neb = NEB(tmp_images)
                        tmp_neb.interpolate()
                        images_new = self.neb.images.copy()
                        images_new.insert(idx + 1, tmp_neb.images[1])

                        energies = self.get_energies(first=True, last=True)
                        energies.insert(idx + 1, None)
                        self.neb = NEB(images_new,
                                       k=self.neb.k[0],
                                       climb=self.neb.climb)
                        self.dump_positions_vasp(prefix=f'Step: {step} 3 ')

                        zfill_length = len(str(len(self.neb.images)))
                        for k, image in enumerate(self.neb.images):
                            self.logger.debug(f'Trying to attach the calc to {k} image '
                                              f'with the length: {len(str(len(self.neb.images)))}')
                            image.calc = construct_calc_fn(str(k).zfill(zfill_length))
                            image.calc.E = energies[k]

                        if length_prev != length_new:
                            self.logger.debug('Trying to rename due to the change in length')
                            for i in range(0, self.neb.nimages):
                                shell(f'mv {str(i).zfill(length_prev)} {str(i).zfill(length_new)}')
                                self.logger.debug(f'Trying to execute the following command: '
                                                  f'mv {str(i).zfill(length_prev)} {str(i + 1).zfill(length_new)}')

                        self.logger.debug('Trying to rename due to the insertion')
                        for i in range(len(self.neb.images) - 2, idx, -1):
                            self.logger.debug(f'{i=}')
                            self.logger.debug(f'Trying to execute the following command: '
                                              f'mv {str(i).zfill(length_new)} {str(i + 1).zfill(length_new)}')
                            shell(f'mv {str(i).zfill(length_new)} {str(i + 1).zfill(length_new)}')

                        self.logger.debug(f'Trying to create the new folder: {str(idx + 1).zfill(length_new)}')
                        folder = Path(str(idx + 1).zfill(length_new))
                        folder.mkdir()
                        self.dump_positions_vasp(prefix=f'Step: {step} 4 ')

                steps_after_insertion += 1
                self.logger.debug(f'Step after insertion: {steps_after_insertion}')

        self.logger.warning(f'convergence was not achieved after max iterations = {max_steps}, '
                            f'residual R = {R:.4f} > {fmax}')
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

        if max_steps < 2:
            raise ValueError('max_steps must be greater or equal than two')
        length = len(str(max_steps))

        self.set_step_in_calculators(0)

        F = self.get_forces()
        self.logger.info(f'Step: {0:{length}}. Energies = {[np.round(en, 4) for en in self.get_energies()]}')

        R = self.neb.get_residual()  # pick the biggest force

        if R >= R_max:
            self.logger.info(f'Step: {0:{length}}. Residual {R:.4f} >= R_max {R_max}')
            raise OptimizerConvergenceError(f'Step: 0. Residual {R:.4f} >= R_max {R_max}')
        else:
            self.logger.info(f'Step: {0:{length}}. Residual R = {R:.4f}')

        if h is None:
            h = 0.5 * rtol ** 0.5 / R  # Chose a step size based on that force
            h = max(h, h_min)  # Make sure the step size is not too big
        self.logger.info(f'Step: {0:{length}}. Step size h = {h}')

        X = self.neb.get_positions().reshape(-1)

        for step in range(1, max_steps):
            X_new = X + h * F  # Pick a new position
            self.update_positions(X_new)

            self.set_step_in_calculators(step)
            F_new = self.get_forces()  # Calculate the new forces at this position
            self.logger.info(f'Step: {step:{length}}. Energies = {[np.round(en, 4) for en in self.get_energies()]}')

            R_new = self.neb.get_residual()
            self.logger.info(f'Step: {step:{length}}. At new coordinates R = {R:.4f} -> R_new = {R_new:.4f}')

            e = 0.5 * h * (F_new - F)  # Estimate the area under the forces curve
            err = np.linalg.norm(e, np.inf)  # Error estimate

            # Accept step if residual decreases sufficiently and/or error acceptable
            condition_1 = R_new <= R * (1 - C1 * h)
            condition_2 = R_new <= R * C2
            condition_3 = err <= rtol
            accept = condition_1 or (condition_2 and condition_3)
            self.logger.info(f'Step: {step:{length}}. {"R_new <= R * (1 - C1 * h)":26} \t is {condition_1}')
            self.logger.info(f'Step: {step:{length}}. {"R_new <= R * C2":26} is {condition_2}')
            self.logger.info(f'Step: {step:{length}}. {"err <= rtol":26} is {condition_3}')

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

            if accept:
                self.logger.info(f'Step: {step:{length}}. The displacement is accepted')

                X = X_new
                R = R_new
                F = F_new

                self.dump_trajectory()
                self.dump_positions_vasp()

                # We check the residuals again
                if self.converged(fmax):
                    self.logger.info(f"Step: {step:{length}}. Optimization terminates successfully")
                    return True

                if R > R_max:
                    self.logger.info(f"Step: {step:{length}}. Optimization fails, R = {R:.4f} > R_max = {R_max}")
                    return False

                # Compute a new step size.
                # Based on the extrapolation and some other heuristics
                h = max(0.25 * h, min(4 * h, h_err, h_ls))  # Log steep-size analytic results
                self.logger.info(f'Step: {step:{length}}. New step size h = {h}')

            else:
                self.logger.info(f'Step: {step:{length}}. The displacement is rejected')
                h = max(0.1 * h, min(0.25 * h, h_err, h_ls))
                self.logger.info(f'Step: {step:{length}}. New step size h = {h}')

            if abs(h) < h_min:  # abort if step size is too small
                self.logger.info(f'Step: {step:{length}}. Stop optimization since step size h = {h} < h_min = {h_min}')
                return True

        self.logger.warning(f'Step: {step:{length}}. Convergence was not achieved after max iterations = {max_steps}')
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
                 restart: Literal[False, 'from_traj', 'from_vasp'] = False,
                 dE_max: float = None):

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
        self.dE_max = dE_max
        self.optimizer = None
        self.logger = logging.getLogger(self.__class__.__name__ + ':')

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
                image.write(f'start_img_{str(i).zfill(length)}.vasp', format='vasp')

        else:
            images = []
            if self.restart == 'from_traj':
                trj = Trajectory('NEB_trajectory.traj')
                n_iter = int(len(trj) / (self.nimages + 2))
                for i in range(self.nimages + 2):
                    trj[(n_iter - 1) * (self.nimages + 2) + i].write(f'start_img_{str(i).zfill(length)}.vasp',
                                                                     format='vasp')
                trj.close()

            if self.restart == 'from_traj' or self.restart == 'from_vasp':
                for i in range(self.nimages + 2):
                    img = read(f'start_img_{str(i).zfill(length)}.vasp', format='vasp')
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
        if self.dE_max is not None:
            self.logger.debug(f'Trying to create the folder {str(0).zfill(length)}')
            folder = Path(str(0).zfill(length))
            folder.mkdir(exist_ok=True)
            self.logger.debug(f'Trying to create the folder {str(self.nimages + 1).zfill(length)}')
            folder = Path(str(self.nimages + 1).zfill(length))
            folder.mkdir(exist_ok=True)

        for i, image in enumerate(images[1:-1]):
            image.calc = JDFTx(self.path_jdftx_executable,
                               path_rundir=self.path_rundir / str(i+1).zfill(length),
                               commands=self.jdftx_input.commands)

        if self.dE_max is not None:
            images[0].calc = JDFTx(self.path_jdftx_executable,
                                   path_rundir=self.path_rundir / str(0).zfill(length),
                                   commands=self.jdftx_input.commands)
            images[-1].calc = JDFTx(self.path_jdftx_executable,
                                    path_rundir=self.path_rundir / str(self.nimages + 1).zfill(length),
                                    commands=self.jdftx_input.commands)

        self.optimizer = NEBOptimizer(neb=neb,
                                      trajectory_filepath='NEB_trajectory.traj')

    def run(self,
            fmax: float = 0.1,
            method: Literal['ode', 'static'] = 'ode',
            max_steps: int = 100,
            **kwargs):

        self.prepare()

        if self.dE_max is not None:
            def calc_fn(folder_name) -> JDFTx:
                return JDFTx(self.path_jdftx_executable,
                             path_rundir=self.path_rundir / folder_name,
                             commands=self.jdftx_input.commands)
        else:
            calc_fn = None
        if method == 'ode':
            self.optimizer.run_ode(fmax, max_steps)
        elif method == 'static':
            self.optimizer.run_static(fmax, max_steps, dE_max=self.dE_max, construct_calc_fn=calc_fn)
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

from __future__ import annotations
import numpy as np
from nptyping import NDArray, Shape, Number


class IonicDynamics:
    def __init__(self,
                 forces_hist: NDArray[Shape['Nsteps, Natoms, 3'], Number] | None,
                 coords_hist: NDArray[Shape['Nsteps, Natoms, 3'], Number] | None,
                 lattice: NDArray[Shape['3, 3'], Number] | None,
                 coords_are_cartesian: bool | None):
        self.forces_hist = forces_hist
        self.coords_hist = coords_hist
        self.lattice = lattice
        self.coords_are_cartesian = coords_are_cartesian

    @property
    def forces(self):
        if self.forces_hist is not None:
            return self.forces_hist[-1]
        else:
            raise ValueError('Forces_hist is None')

    def get_forces(self,
                   mod: str = 'mean',
                   diff: bool = False):
        """
        Args:
            mod (str, optional):
                norm - (N_steps, N_atoms) returns the norm of forces along the ionic trajectory
                mean - (N_steps, ) returns the mean value of forces' norm in simulation cell along the ionic trajectory
                max - (N_steps, ) returns the max value of forces' norm in simulation cell along the ionic trajectory
            diff (bool, optional): if True returns absolute value of forces differences between i and i+1 steps.
            If False returns just forces values at each step

        Returns:

        """
        if self.forces_hist is not None:
            if mod == 'norm':
                forces = np.linalg.norm(self.forces_hist, axis=2)
            elif mod == 'mean':
                forces = np.mean(np.linalg.norm(self.forces_hist, axis=2), axis=1)
            elif mod == 'max':
                forces = np.max(np.linalg.norm(self.forces_hist, axis=2), axis=1)
            else:
                raise ValueError(f'mod should be norm/mean/max. You set {mod}')

            if diff:
                return np.abs(forces[1:] - forces[:-1])
            else:
                return forces
        else:
            raise ValueError('Forces_hist is None')

    def get_displacements(self,
                          i: int | None = None,
                          j: int | None = None,
                          scalar: bool = True) -> (NDArray[Shape['Natoms, 3']] |
                                                   NDArray[Shape['Natoms, 1']] |
                                                   NDArray[Shape['Nsteps, Natoms, 3']] |
                                                   NDArray[Shape['Nsteps, Natoms, 1']]):
        if self.coords_hist is not None and \
                self.lattice is not None and \
                self.coords_are_cartesian is not None:

            if isinstance(i, int) and isinstance(j, int):

                if self.coords_are_cartesian:
                    transform = np.linalg.inv(self.lattice)
                    r1 = np.matmul(self.coords_hist[i], transform)
                    r2 = np.matmul(self.coords_hist[j], transform)
                else:
                    r1 = self.coords_hist[i]
                    r2 = self.coords_hist[j]

                R = r2 - r1
                R = (R + 0.5) % 1 - 0.5
                assert np.all(R >= - 0.5) and np.all(R <= 0.5)

                if scalar:
                    return np.linalg.norm(np.matmul(R, self.lattice), axis=1)
                else:
                    return np.matmul(R, self.lattice)

            elif i is None or j is None:
                if self.coords_are_cartesian:
                    transform = np.linalg.inv(self.lattice)
                    R = np.matmul(self.coords_hist, transform)
                else:
                    R = self.coords_hist
                R = np.diff(R, axis=0)
                R = (R + 0.5) % 1 - 0.5

                if scalar:
                    return np.linalg.norm(np.matmul(R, self.lattice), axis=2)
                else:
                    return np.matmul(R, self.lattice)

        else:
            raise ValueError('Method get_distance_matrix can only be called '
                             'if coords_hist, lattice and coords_are_cartesian are not None')

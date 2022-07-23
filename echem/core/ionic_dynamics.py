import numpy as np


class IonicDynamics:
    def __init__(self,
                 forces_hist: np.ndarray):
        self.forces_hist = forces_hist

    @property
    def forces(self):
        return self.forces_hist[-1]

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

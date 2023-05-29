import numpy as np
from pathlib import Path
from nptyping import NDArray, Shape, Number


class ACF:
    def __init__(self,
                 coords: NDArray[Shape['Natoms, 3'], Number],
                 nelec_per_atom: NDArray[Shape['Natoms'], Number]):
        self.coords = coords
        self.nelec_per_atom = nelec_per_atom
        self.nelec_per_isolated_atom = None

    @staticmethod
    def from_file(filepath: str | Path):
        if isinstance(filepath, str):
            filepath = Path(filepath)

        file = open(filepath)
        file.readline()
        line = file.readline()

        if '+' in line:
            data = np.genfromtxt(filepath, skip_header=2, skip_footer=5, delimiter='|')
        else:
            data = np.genfromtxt(filepath, skip_header=2, skip_footer=4)

        return ACF(data[:, 1:4], data[:, 4])

    def get_charge(self,
                   nelec_per_isolated_atom: NDArray[Shape['Natoms'], Number] | None = None):
        if nelec_per_isolated_atom is not None:
            return nelec_per_isolated_atom - self.nelec_per_atom
        else:
            if self.nelec_per_isolated_atom is not None:
                return self.nelec_per_isolated_atom - self.nelec_per_atom
            else:
                raise ValueError('nelec_per_isolated_atom should be defined either as argument of '
                                 'this function or as self.nelec_per_isolated_atom')

    def get_delta_elec(self,
                       nelec_per_isolated_atom: NDArray[Shape['Natoms'], Number]):
        return self.nelec_per_atom - nelec_per_isolated_atom

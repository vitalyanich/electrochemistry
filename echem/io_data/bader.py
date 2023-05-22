import numpy as np
from pathlib import Path
from nptyping import NDArray, Shape, Number


class ACF:
    def __init__(self,
                 coords: NDArray[Shape['Natoms, 3'], Number],
                 nelec_per_atom: NDArray[Shape['Natoms'], Number]):
        self.coords = coords
        self.nelec_per_atom = nelec_per_atom

    @staticmethod
    def from_file(filepath: str | Path):
        if isinstance(filepath, str):
            filepath = Path(filepath)

        data = np.genfromtxt(filepath, skip_header=2, skip_footer=4)

        return ACF(data[:, 1:4], data[:, 4])

    def get_charge(self,
                   nele_per_isolated_atom: NDArray[Shape['Natoms'], Number]):
        return nele_per_isolated_atom - self.nelec_per_atom

    def get_delta_elec(self,
                       nele_per_isolated_atom: NDArray[Shape['Natoms'], Number]):
        return self.nelec_per_atom - nele_per_isolated_atom

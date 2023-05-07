import os
import re
import shutil
from typing import Callable
from echem.io_data.jdftx import Ionpos, Lattice
from echem.io_data.vasp import Poscar
from echem.core.structure import Structure
from echem.core.constants import THz2eV
from echem.core.thermal_properties import ThermalProperties
from InterPhon.core import PreProcess, PostProcess
from nptyping import NDArray, Shape, Number
from typing import Union
from pathlib import Path


class InterPhonInterface(ThermalProperties):
    def __init__(self,
                 folder_to_jdftx_files: Union[str, Path],
                 folder_files_to_copy: Union[str, Path] = None,
                 select_fun: Callable[[Structure], list[list[str]]] = None,
                 user_args: dict = None,
                 sym_flag: bool = True):

        if isinstance(folder_to_jdftx_files, str):
            folder_to_jdftx_files = Path(folder_to_jdftx_files)
        if isinstance(folder_files_to_copy, str):
            folder_files_to_copy = Path(folder_files_to_copy)

        self.folder_to_jdftx_files = folder_to_jdftx_files
        self.folder_files_to_copy = folder_files_to_copy
        self.select_fun = select_fun
        self.user_args = user_args
        self.sym_flag = sym_flag

        self.post_process = None
        self.eigen_freq = None
        self.weights = None

    def _create_poscar_for_interphon(self,
                                     folder_to_jdftx_files: Path,
                                     select_fun: Callable[[Structure], list[list[str]]] = None) -> None:
        """
        Function creates POSCAR with unitcell for InterPhon adding selective dynamics data

        Args:
            folder_to_jdftx_files (str): path to folder with jdft.ionpos and jdft.lattice files
            select_fun (Callable, optional): function that take Structure as input and provides list with
                selective dynamics data for POSCAR class. All atoms are allowed to move in default.
        """

        ionpos = Ionpos.from_file(folder_to_jdftx_files / 'jdft.ionpos')
        lattice = Lattice.from_file(folder_to_jdftx_files / 'jdft.lattice')
        poscar = ionpos.convert('vasp', lattice)

        if select_fun is not None:
            sd_data = select_fun(poscar.structure)
        else:
            sd_data = [['T', 'T', 'T'] for _ in range(poscar.structure.natoms)]

        poscar.sdynamics_data = sd_data

        poscar.to_file(folder_to_jdftx_files / 'POSCAR_unitcell_InterPhon')

    def _make_preprocess(self,
                         poscar_unitcell: Path,
                         folder_to_disps: Path,
                         folder_files_to_copy: Path = None,
                         user_args: dict = None,
                         sym_flag: bool = True) -> None:
        """
        Function creates folders with POSCARs with displaced atoms and all other necessary files for calculation

        Args:
            poscar_unitcell (str): path to the POSCAR file that contains the unitcell for IterPhon
                with defined sd dynamics
            folder_to_disps (str): path to a folder where all new folders with corresponding POSCARs with
                displaced atoms will be created
            folder_files_to_copy (str, optional): path to a folder from which all files will be copied to each
                new folder with new POSCARs
            user_args (dict, optional): dist with all necessary information for the InterPhon PreProcess class.
                Only 2D periodicity is supported. If you want switch off symmetries, you have to define 'periodicity'
                in user_args and set sym_flag=False
                Example and default value: user_args = {'dft_code': 'vasp', 'displacement': 0.05,
                'enlargement': "1 1 1", 'periodicity': "1 1 0"}
            sym_flag (bool, optional): if True the symmetry will be applied. Only 2D symmetries are supported

        """
        if user_args is None:
            user_args = {'dft_code': 'vasp',
                         'displacement': 0.05,
                         'enlargement': '1 1 1',
                         'periodicity': '1 1 0'}

        if poscar_unitcell != folder_to_disps / 'POSCAR_unitcell_InterPhon':
            shutil.copyfile(poscar_unitcell, folder_to_disps / 'POSCAR_unitcell_InterPhon')

        pre_process = PreProcess()
        pre_process.set_user_arg(user_args)

        pre_process.set_unit_cell(in_file=str(poscar_unitcell),
                                  code_name='vasp')
        pre_process.set_super_cell(out_file=str(folder_to_disps / 'POSCAR_supercell_InterPhon'),
                                   code_name='vasp')
        pre_process.write_displace_cell(out_file=str(folder_to_disps / 'POSCAR'),
                                        code_name='vasp',
                                        sym_flag=sym_flag)

        poscars_disp = [f for f in folder_to_disps.iterdir() if f.is_file() and bool(re.search(r'POSCAR-\d{4}$',
                                                                                               f.name))]

        for poscar_disp in poscars_disp:
            poscar = Poscar.from_file(poscar_disp)
            ionpos, lattice = poscar.convert('jdftx')

            subfolder_to_disp = folder_to_disps / poscar_disp.name[-4:]
            if not os.path.isdir(subfolder_to_disp):
                os.mkdir(subfolder_to_disp)
            ionpos.to_file(subfolder_to_disp / 'jdft.ionpos')
            lattice.to_file(subfolder_to_disp / 'jdft.lattice')

            shutil.copyfile(folder_to_disps / poscar_disp, subfolder_to_disp / 'POSCAR')

            if folder_files_to_copy is not None:
                files_to_copy = [f for f in folder_files_to_copy.iterdir() if f.is_file()]
                for file in files_to_copy:
                    shutil.copyfile(file, subfolder_to_disp / file.name)

        with open(folder_to_disps / 'user_args_InterPhon', 'w') as file:
            for key, value in user_args.items():
                file.write(f'{key}: {value}\n')

    def _make_postprocess(self,
                          folder_to_disps: Path,
                          filepath_unitcell: Path,
                          filepath_supercell: Path,
                          filepath_kpoints: Path,
                          user_args: dict = None,
                          sym_flag: bool = True) -> None:
        """
        Function process the output files after all calculations with displaced atoms are finished

        Args:
            folder_to_disps (str): path to the folder contains all folders with performed calculations with
                atom displacements
            filepath_unitcell (str): path to the POSCAR file that contains the unitcell for IterPhon
                with defined sd dynamics
            filepath_supercell (str): path to the POSCAR file produced by InterPhon with proper enlargement
            filepath_kpoints (str): path to the KPOINTS file. The phonons will be assessed in the given k-points
            user_args (dict, optional): dist with all necessary information for the InterPhon PreProcess class.
                Example and default value: user_args = {'dft_code': 'vasp', 'displacement': 0.05,
                'enlargement': "1 1 1", 'periodicity': "1 1 0"}
            sym_flag (bool, optional): if True the symmetry will be applied. Only 2D symmetries are supported
        """
        if user_args is None:
            user_args = {'dft_code': 'vasp',
                         'displacement': 0.05,
                         'enlargement': '1 1 1',
                         'periodicity': '1 1 0'}

        output_paths = [f / 'output.out' for f in folder_to_disps.iterdir()
                        if f.is_dir() and bool(re.search(r'\d{4}$', f.name))]

        post_process = PostProcess(in_file_unit_cell=str(filepath_unitcell),
                                   in_file_super_cell=str(filepath_supercell),
                                   code_name='vasp')
        post_process.set_user_arg(user_args)
        post_process.set_reciprocal_lattice()
        post_process.set_force_constant(force_files=[str(f) for f in output_paths],
                                        code_name='jdftx',
                                        sym_flag=sym_flag)
        post_process.set_k_points(k_file=str(filepath_kpoints))
        post_process.eval_phonon()

        self.post_process = post_process
        ThermalProperties.__init__(self, self.post_process.w_q * THz2eV)

    def create_displacements_jdftx(self):
        self._create_poscar_for_interphon(folder_to_jdftx_files=self.folder_to_jdftx_files,
                                          select_fun=self.select_fun)
        self._make_preprocess(poscar_unitcell=self.folder_to_jdftx_files / 'POSCAR_unitcell_InterPhon',
                              folder_to_disps=self.folder_to_jdftx_files,
                              folder_files_to_copy=self.folder_files_to_copy,
                              user_args=self.user_args,
                              sym_flag=self.sym_flag)

    def get_phonons(self) -> NDArray[Shape['Nkpts, Nfreq'], Number]:
        if self.post_process is None:
            self._make_postprocess(folder_to_disps=self.folder_to_jdftx_files,
                                   filepath_unitcell=self.folder_to_jdftx_files / 'POSCAR_unitcell_InterPhon',
                                   filepath_supercell=self.folder_to_jdftx_files / 'POSCAR_supercell_InterPhon',
                                   filepath_kpoints=self.folder_to_jdftx_files / 'KPOINTS',
                                   user_args=self.user_args,
                                   sym_flag=self.sym_flag)

        return self.eigen_freq

    def get_E_zpe(self) -> float:
        if self.eigen_freq is None:
            self._make_postprocess(folder_to_disps=self.folder_to_jdftx_files,
                                   filepath_unitcell=self.folder_to_jdftx_files / 'POSCAR_unitcell_InterPhon',
                                   filepath_supercell=self.folder_to_jdftx_files / 'POSCAR_supercell_InterPhon',
                                   filepath_kpoints=self.folder_to_jdftx_files / 'KPOINTS',
                                   user_args=self.user_args,
                                   sym_flag=self.sym_flag)
        return ThermalProperties.get_E_zpe(self)

    def get_E_temp(self,
                   T: float) -> float:
        if self.eigen_freq is None:
            self._make_postprocess(folder_to_disps=self.folder_to_jdftx_files,
                                   filepath_unitcell=self.folder_to_jdftx_files / 'POSCAR_unitcell_InterPhon',
                                   filepath_supercell=self.folder_to_jdftx_files / 'POSCAR_supercell_InterPhon',
                                   filepath_kpoints=self.folder_to_jdftx_files / 'KPOINTS',
                                   user_args=self.user_args,
                                   sym_flag=self.sym_flag)
        return ThermalProperties.get_E_temp(self, T)

    def get_TS(self,
               T: float) -> float:
        if self.eigen_freq is None:
            self._make_postprocess(folder_to_disps=self.folder_to_jdftx_files,
                                   filepath_unitcell=self.folder_to_jdftx_files / 'POSCAR_unitcell_InterPhon',
                                   filepath_supercell=self.folder_to_jdftx_files / 'POSCAR_supercell_InterPhon',
                                   filepath_kpoints=self.folder_to_jdftx_files / 'KPOINTS',
                                   user_args=self.user_args,
                                   sym_flag=self.sym_flag)
        return ThermalProperties.get_TS(self, T)

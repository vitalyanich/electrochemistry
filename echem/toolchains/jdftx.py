import numpy as np
import re
from monty.re import regrep
from pathlib import Path
from typing import Union
from typing import TypedDict
from subprocess import Popen, PIPE
from timeit import default_timer as timer
from datetime import timedelta
from echem.io_data.jdftx import VolumetricData
from echem.io_data.jdftx import Output
from echem.io_data.ddec import AtomicNetCharges


class System(TypedDict):
    substrate: str
    adsorbate: str
    idx: int
    is_pzc: bool
    output: Output
    ddec_nac: AtomicNetCharges


text = r'''</net charge>

<input filename>
valence_density.cube
</input filename>

<periodicity along A, B, and C vectors>
.true.
.true.
.true.
</periodicity along A, B, and C vectors>

<atomic densities directory complete path>
G:\My Drive\Software\chargemol_09_26_2017\atomic_densities\
</atomic densities directory complete path>

<charge type>
DDEC6
</charge type>

<compute BOs>
.true.
</compute BOs>

<number of core electrons>
1 0
6 2
8 2
29 10
50 36
</number of core electrons>'''


class InfoExtractor:
    def __init__(self,
                 path_ddec_executable: Union[str, Path],
                 systems: list[System] = None,
                 output_name: str = 'output.out',
                 jdftx_prefix: str = 'jdft'):

        self.path_ddec_executable = path_ddec_executable

        if systems is None:
            self.systems = []

        self.output_name = output_name
        self.jdftx_prefix = jdftx_prefix

    def get_info(self,
                 path_root_folder: Union[str, Path]):

        if isinstance(path_root_folder, str):
            path_root_folder = Path(path_root_folder)

        paths_subfolders = [path for path in path_root_folder.iterdir() if path.is_dir()]

        for path_subfolder in paths_subfolders:
            files = [file.name for file in path_subfolder.iterdir() if file.is_file()]

            substrate, adsorbate, idx = path_subfolder.name.split('_')
            if '+' in substrate or '-' in substrate:
                is_pzc = False
            else:
                is_pzc = True
            output = Output.from_file(path_subfolder / self.output_name)

            # Check whether all nessesary files have been already created
            if not all(i in files for i in ['valence_density.cube', 'POSCAR', 'CONTCAR', 'XDATCAR', 'job_control.txt']):
                print(f'Create necessary files for {path_subfolder}')

                fft_box_size = output.fft_box_size

                poscar = output.get_poscar()
                poscar.to_file(path_subfolder / 'POSCAR')

                contcar = output.get_contcar()
                contcar.to_file(path_subfolder / 'CONTCAR')

                xdatcar = output.get_xdatcar()
                xdatcar.to_file(path_subfolder / 'XDATCAR')

                if 'output_fft.out' in files:
                    files.remove('output_fft.out')
                    patterns = {'fft_box_size': r'Chosen fftbox size, S = \[(\s+\d+\s+\d+\s+\d+\s+)\]'}
                    matches = regrep(path_subfolder / 'output_fft.out', patterns)
                    fft_box_size = np.array([int(i) for i in matches['fft_box_size'][0][0][0].split()])

                n_up = VolumetricData.from_file(path_subfolder / f'{self.jdftx_prefix}.n_up',
                                                fft_box_size,
                                                output.structure).convert_to_cube()
                n_dn = VolumetricData.from_file(path_subfolder / f'{self.jdftx_prefix}.n_dn',
                                                fft_box_size,
                                                output.structure).convert_to_cube()
                n = n_up + n_dn
                n.to_file(path_subfolder / 'valence_density.cube')

                if is_pzc:
                    charge = 0.0
                    job_control = open(path_subfolder / 'job_control.txt', 'w')
                    job_control.write('<net charge>\n')
                    job_control.write(f'{charge}\n')
                    job_control.write(text)
                    job_control.close()
                else:
                    substrate_pure = re.split('[-+]', substrate)[0]
                    systems_pzc_ref = [system for system in self.systems if system['substrate'] == substrate_pure and system['adsorbate'] == adsorbate]
                    if len(systems_pzc_ref) == 0:
                        raise ValueError(f'There is no PZC reference for {path_subfolder}')
                    nelec_arr = [system['output'].nelec for system in systems_pzc_ref]

                    if not all(nelec == nelec_arr[0] for nelec in nelec_arr):
                        raise ValueError('There are different number of electrons for PZC systems')

                    charge = - (output.nelec_hist[-1] - nelec_arr[0])
                    job_control = open(path_subfolder / 'job_control.txt', 'w')
                    job_control.write('<net charge>\n')
                    job_control.write(f'{charge}\n')
                    job_control.write(text)
                    job_control.close()

            if 'nbound.cube' not in files and f'{self.jdftx_prefix}.nbound' in files:
                    nbound = VolumetricData.from_file(path_subfolder / f'{self.jdftx_prefix}.nbound', fft_box_size, output.structure).convert_to_cube()
                    nbound.to_file(path_subfolder / 'nbound.cube')

            if 'DDEC6_even_tempered_net_atomic_charges.xyz' in files:
                ddec_nac = AtomicNetCharges.from_file(path_subfolder / 'DDEC6_even_tempered_net_atomic_charges.xyz')
            else:
                print(f'DDEC NACs have not been found in folder {path_subfolder}')
                print('Run DDEC')
                start = timer()

                p = Popen(self.path_ddec_executable, stdin=PIPE, bufsize=0)
                p.communicate(str(path_subfolder).encode('ascii'))
                end = timer()
                print(f'Done! Elapsed time: {timedelta(seconds=end-start)}')
                ddec_nac = AtomicNetCharges.from_file(path_subfolder / 'DDEC6_even_tempered_net_atomic_charges.xyz')

            system: System = {'substrate': substrate,
                              'adsorbate': adsorbate,
                              'idx': idx,
                              'is_pzc': is_pzc,
                              'output': output,
                              'ddec_nac': ddec_nac}

            self.systems.append(system)
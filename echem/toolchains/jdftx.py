from pathlib import Path
from typing_extensions import Required, NotRequired, TypedDict
from echem.io_data.jdftx import VolumetricData
from echem.io_data.jdftx import Output
from echem.io_data.ddec import AtomicNetCharges
from echem.core.constants import Hartree2eV, Bohr2Angstrom
from monty.re import regrep
from subprocess import Popen, PIPE
from timeit import default_timer as timer
from datetime import timedelta
import matplotlib.pyplot as plt
import re
import numpy as np


class System(TypedDict):
    substrate: str
    adsorbate: str
    idx: int
    is_pzc: bool
    output: Output
    ddec_nac: AtomicNetCharges

class DDEC_params(TypedDict):
    path_atomic_densities: Required[str]
    path_ddec_executable: NotRequired[str]
    input_filename: NotRequired[str]
    periodicity: NotRequired[tuple[bool]]
    charge_type: NotRequired[str]
    compute_BOs: NotRequired[bool]
    number_of_core_electrons: NotRequired[list[list[int]]]


class InfoExtractor:
    def __init__(self,
                 ddec_params: DDEC_params,
                 systems: list[System] = None,
                 output_name: str = 'output.out',
                 jdftx_prefix: str = 'jdft',
                 do_ddec: bool = False):

        if 'path_ddec_executable' in ddec_params:
            self.path_ddec_executable = ddec_params['path_ddec_executable']
        elif do_ddec:
            raise ValueError('"path_ddec_executable" ust be specified in ddec_params if do_ddec=True')

        if systems is None:
            self.systems = []

        self.output_name = output_name
        self.jdftx_prefix = jdftx_prefix
        self.do_ddec = do_ddec
        self.ddec_params = ddec_params

    def create_job_control(self,
                           filepath: str | Path,
                           charge: float,
                           ddec_params: DDEC_params):
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if 'path_atomic_densities' in ddec_params:
            path_atomic_densities = ddec_params['path_atomic_densities']
        else:
            raise ValueError('"path_atomic_densities" must be specified in ddec_params dict')

        if 'input_filename' in ddec_params:
            input_filename = ddec_params['input_filename']
        else:
            input_filename = None

        if 'periodicity' in ddec_params:
            periodicity = ddec_params['periodicity']
        else:
            periodicity = (True, True, True)

        if 'charge_type' in ddec_params:
            charge_type = ddec_params['charge_type']
        else:
            charge_type = 'DDEC6'

        if 'compute_BOs' in ddec_params:
            compute_BOs = ddec_params['compute_BOs']
        else:
            compute_BOs = True

        if 'number_of_core_electrons' in ddec_params:
            number_of_core_electrons = ddec_params['number_of_core_electrons']
        else:
            number_of_core_electrons = None

        job_control = open(filepath, 'w')

        job_control.write('<net charge>\n')
        job_control.write(f'{charge}\n')
        job_control.write('</net charge>\n\n')

        job_control.write('<atomic densities directory complete path>\n')
        job_control.write(path_atomic_densities + '\n')
        job_control.write('</atomic densities directory complete path>\n\n')

        if input_filename is not None:
            job_control.write('<input filename>\n')
            job_control.write(input_filename + '\n')
            job_control.write('</input filename>\n\n')

        job_control.write('<periodicity along A, B, and C vectors>\n')
        for p in periodicity:
            if p:
                job_control.write('.true.\n')
            else:
                job_control.write('.false.\n')
        job_control.write('</periodicity along A, B, and C vectors>\n\n')

        job_control.write('<charge type>\n')
        job_control.write(charge_type + '\n')
        job_control.write('</charge type>\n\n')

        job_control.write('<compute BOs>\n')
        if compute_BOs:
            job_control.write('.true.\n')
        else:
            job_control.write('.false.\n')
        job_control.write('</compute BOs>\n')

        if number_of_core_electrons is not None:
            job_control.write('<number of core electrons>\n')
            for i in number_of_core_electrons:
                job_control.write(f'{i[0]} {i[1]}\n')
            job_control.write('</number of core electrons>\n')

        job_control.close()

    def get_info_multiple(self,
                          path_root_folder: str | Path,
                          recreate_files: bool = False):
        if isinstance(path_root_folder, str):
            path_root_folder = Path(path_root_folder)

        subfolders = [f for f in path_root_folder.rglob('*') if f.is_dir()]
        depth = max([len(f.parents) for f in subfolders])
        subfolders = [f for f in subfolders if len(f.parents) == depth]

        for folder in subfolders:
            self.get_info(folder, recreate_files)

    def get_info(self,
                 path_root_folder: str | Path,
                 recreate_files: bool = False):

        if isinstance(path_root_folder, str):
            path_root_folder = Path(path_root_folder)

        files = [file.name for file in path_root_folder.iterdir() if file.is_file()]

        substrate, adsorbate, idx, *_ = path_root_folder.name.split('_')
        if 'vib' in _:
            return None
        if '+' in substrate or '-' in substrate:
            is_pzc = False
        else:
            is_pzc = True
        output = Output.from_file(path_root_folder / self.output_name)

        # Check whether all nessesary files have been already created
        if not all(i in files for i in ['valence_density.cube', 'POSCAR', 'CONTCAR', 'XDATCAR', 'job_control.txt']) or recreate_files:
            print(f'Create necessary files for {path_root_folder}')

            fft_box_size = output.fft_box_size

            poscar = output.get_poscar()
            poscar.to_file(path_root_folder / 'POSCAR')

            contcar = output.get_contcar()
            contcar.to_file(path_root_folder / 'CONTCAR')

            xdatcar = output.get_xdatcar()
            xdatcar.to_file(path_root_folder / 'XDATCAR')

            if 'output_fft.out' in files:
                files.remove('output_fft.out')
                patterns = {'fft_box_size': r'Chosen fftbox size, S = \[(\s+\d+\s+\d+\s+\d+\s+)\]'}
                matches = regrep(str(path_root_folder / 'output_fft.out'), patterns)
                fft_box_size = np.array([int(i) for i in matches['fft_box_size'][0][0][0].split()])

            if f'{self.jdftx_prefix}.n_up' in files and f'{self.jdftx_prefix}.n_dn' in files:
                n_up = VolumetricData.from_file(path_root_folder / f'{self.jdftx_prefix}.n_up',
                                                fft_box_size,
                                                output.structure).convert_to_cube()
                n_dn = VolumetricData.from_file(path_root_folder / f'{self.jdftx_prefix}.n_dn',
                                                fft_box_size,
                                                output.structure).convert_to_cube()
                n = n_up + n_dn
                n.to_file(path_root_folder / 'valence_density.cube')

                if output.get_magnetization_abs > 1e-2:
                    n = n_up - n_dn
                    n.to_file(path_root_folder / 'spin_density.cube')

            elif f'{self.jdftx_prefix}.n' in files:
                n = VolumetricData.from_file(path_root_folder / f'{self.jdftx_prefix}.n',
                                             fft_box_size,
                                             output.structure).convert_to_cube()
                n.to_file(path_root_folder / 'valence_density.cube')

            for file in files:
                if file.startswith(f'{self.jdftx_prefix}.fluidN_'):
                    fluidN = VolumetricData.from_file(path_root_folder / file,
                                                      fft_box_size,
                                                      output.structure).convert_to_cube()
                    fluidN.to_file(path_root_folder / (file.removeprefix(self.jdftx_prefix + '.') + '.cube'))

            if 'nbound.cube' not in files and f'{self.jdftx_prefix}.nbound' in files:
                nbound = VolumetricData.from_file(path_root_folder / f'{self.jdftx_prefix}.nbound', fft_box_size, output.structure).convert_to_cube()
                nbound.to_file(path_root_folder / 'nbound.cube')

            if is_pzc:
                self.create_job_control(filepath=path_root_folder / 'job_control.txt',
                                        charge=0.0,
                                        ddec_params=self.ddec_params)
            else:
                substrate_pure = re.split('[-+]', substrate)[0]
                systems_pzc_ref = [system for system in self.systems if system['substrate'] == substrate_pure and system['adsorbate'] == adsorbate]
                if len(systems_pzc_ref) == 0:
                    raise ValueError(f'There is no PZC reference for {path_root_folder}')
                nelec_arr = [system['output'].nelec for system in systems_pzc_ref]

                if not all(nelec == nelec_arr[0] for nelec in nelec_arr):
                    raise ValueError('There are different number of electrons for PZC systems')

                charge = - (output.nelec_hist[-1] - nelec_arr[0])
                self.create_job_control(filepath=path_root_folder / 'job_control.txt',
                                        charge=charge,
                                        ddec_params=self.ddec_params)

        if 'DDEC6_even_tempered_net_atomic_charges.xyz' in files:
            ddec_nac = AtomicNetCharges.from_file(path_root_folder / 'DDEC6_even_tempered_net_atomic_charges.xyz')
        elif self.do_ddec:
            print(f'DDEC NACs have not been found in folder {path_root_folder}')
            print('Run DDEC')
            start = timer()

            p = Popen(str(self.path_ddec_executable), stdin=PIPE, bufsize=0)
            p.communicate(str(path_root_folder).encode('ascii'))
            end = timer()
            print(f'Done! Elapsed time: {timedelta(seconds=end-start)}')
            ddec_nac = AtomicNetCharges.from_file(path_root_folder / 'DDEC6_even_tempered_net_atomic_charges.xyz')
        else:
            ddec_nac = None

        system: System = {'substrate': substrate,
                          'adsorbate': adsorbate,
                          'idx': int(idx),
                          'is_pzc': is_pzc,
                          'output': output,
                          'ddec_nac': ddec_nac}

        self.systems.append(system)

    def get_system(self, substrate: str, adsorbate: str, idx: int = None):
        if idx is None:
            return [system for system in self.systems if
                    system['substrate'] == substrate and system['adsorbate'] == adsorbate]
        else:
            return [system for system in self.systems if
                    system['substrate'] == substrate and system['adsorbate'] == adsorbate and system['idx'] == idx]

    def get_F(self, substrate: str, adsorbate: str, idx: int):
        return self.get_system(substrate, adsorbate, idx)[0]['output'].energy_ionic_hist['F'][-1]

    def get_G(self, substrate: str, adsorbate: str, idx: int):
        return self.get_system(substrate, adsorbate, idx)[0]['output'].energy_ionic_hist['G'][-1]

    def get_N(self, substrate: str, adsorbate: str, idx: int):
        return self.get_system(substrate, adsorbate, idx)[0]['output'].nelec

    def get_mu(self, substrate: str, adsorbate: str, idx: int):
        return self.get_system(substrate, adsorbate, idx)[0]['output'].mu

    def plot_energy(self, substrate: str, adsorbate: str):

        systems = self.get_system(substrate, adsorbate)
        energy_min = min(system['output'].energy for system in systems)

        i, j = np.divmod(len(systems), 3)
        i += 1
        if j == 0:
            i -= 1

        fig, axs = plt.subplots(i, 3, figsize=(25, 5 * i), dpi=180)
        fig.subplots_adjust(wspace=0.3, hspace=0.2)

        for system, ax_e in zip(systems, axs.flatten()):
            out = system['output']

            ax_e.plot(range(out.nisteps), (out.energy_ionic_hist['F'] - out.energy_ionic_hist['F'][-1]) * Hartree2eV, color='r', label=r'$\Delta F$', ms=3, marker='o')
            ax_e.set_yscale('log')
            ax_e.set_xlabel(r'$Step$', fontsize=12)
            ax_e.set_ylabel(r'$\Delta F, \ eV$', color='r', fontsize=14)
            ax_e.legend(fontsize=14)

            delta_E = (out.energy - energy_min) * Hartree2eV
            if np.abs(delta_E) < 1e-8:
                ax_e.text(0.5, 0.9, f'$\mathbf{{E_f - E_f^{{min}} = {np.round(delta_E, 2)} \ eV}}$', ha='center', va='center', transform=ax_e.transAxes, fontsize=12)
                ax_e.set_title(f'$\mathbf{{ {substrate} \ {adsorbate} \ {system["idx"]} }}$', fontsize=13, y=1, pad=-15)
            else:
                ax_e.text(0.5, 0.9, f'$E_f - E_f^{{min}} = {np.round(delta_E, 2)} \ eV$', ha='center', va='center', transform=ax_e.transAxes, fontsize=12)
                ax_e.set_title(f'${substrate} \ {adsorbate} \ {system["idx"]}$', fontsize=13, y=1, pad=-15)

            ax_f = ax_e.twinx()
            ax_f.plot(range(len(out.get_forces())), out.get_forces() * Hartree2eV / (Bohr2Angstrom ** 2), color='g', label=r'$\left< |\vec{F}| \right>$', ms=3, marker='o')
            ax_f.set_ylabel(r'$Average \ Force, \ eV / \AA^3$', color='g', fontsize=13)
            ax_f.legend(loc='upper right', bbox_to_anchor=(1, 0.9), fontsize=13)
            ax_f.set_yscale('log')
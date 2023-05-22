from pathlib import Path
from wsl_pathlib.path import WslPath
from typing_extensions import Required, NotRequired, TypedDict
from echem.io_data.jdftx import VolumetricData, Output, Lattice, Ionpos, Eigenvals, Fillings, kPts, DOS
from echem.io_data.ddec import Output_DDEC
from echem.io_data.bader import ACF
from echem.core.constants import Hartree2eV, eV2Hartree, Bohr2Angstrom, Angstrom2Bohr
from echem.core.electronic_structure import EBS
from monty.re import regrep
from subprocess import Popen, PIPE
import subprocess
from timeit import default_timer as timer
from datetime import timedelta
import matplotlib.pyplot as plt
import shutil
import numpy as np
from nptyping import NDArray, Shape, Number
from typing import Literal
from tqdm.autonotebook import tqdm
from termcolor import colored
from threading import Lock
from concurrent.futures import ThreadPoolExecutor


class System(TypedDict):
    substrate: str
    adsorbate: str
    idx: int
    output: Output | None
    nac_ddec: Output_DDEC | None
    output_phonons: Output | None
    dos: EBS | None
    nac_bader: ACF | None


class DDEC_params(TypedDict):
    path_atomic_densities: Required[str]
    path_ddec_executable: NotRequired[str]
    input_filename: NotRequired[str]
    periodicity: NotRequired[tuple[bool]]
    charge_type: NotRequired[str]
    compute_BOs: NotRequired[bool]
    number_of_core_electrons: NotRequired[list[list[int]]]


class SBATCH_phonon(TypedDict):
    main_text: NotRequired[str]
    path_executable: Required[str]


class InfoExtractor:
    def __init__(self,
                 ddec_params: DDEC_params = None,
                 sbatch_phonon: SBATCH_phonon = None,
                 systems: list[System] = None,
                 output_name: str = 'output.out',
                 jdftx_prefix: str = 'jdft',
                 do_ddec: bool = False,
                 do_bader: bool = False):

        if ddec_params is not None:
            if do_ddec and 'path_ddec_executable' not in ddec_params:
                raise ValueError('"path_ddec_executable" must be specified in ddec_params if do_ddec=True')
        elif do_ddec:
            raise ValueError('"ddec_params" mist be specified if do_ddec=True')

        if systems is None:
            self.systems = []

        self.output_name = output_name
        self.jdftx_prefix = jdftx_prefix
        self.do_ddec = do_ddec
        self.ddec_params = ddec_params
        self.sbatch_phonon = sbatch_phonon
        self.do_bader = do_bader

        self.lock = Lock()

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

    def check_out_outvib_sameness(self):
        for system in self.systems:
            if system['output'] is not None and system['output_phonons'] is not None:
                if not system['output'].structure == system['output_phonons'].structure:
                    print(colored('System:', color='red'),
                          colored(' '.join((system['substrate'], system['adsorbate'], str(system['idx']))),
                                  color='red', attrs=['bold']),
                          colored('has output and phonon output for different systems'))

    def get_info_multiple(self,
                          path_root_folder: str | Path,
                          recreate_files: bool = False,
                          num_workers: int = 1) -> None:
        if isinstance(path_root_folder, str):
            path_root_folder = Path(path_root_folder)

        subfolders = [f for f in path_root_folder.rglob('*') if f.is_dir()]
        depth = max([len(f.parents) for f in subfolders])
        subfolders = [f for f in subfolders if len(f.parents) == depth]

        with tqdm(total=len(subfolders)) as pbar:
            with ThreadPoolExecutor(num_workers) as executor:
                for _ in executor.map(self.get_info, subfolders, [recreate_files] * len(subfolders)):
                    pbar.update()

    def get_info(self,
                 path_root_folder: str | Path,
                 recreate_files: bool = False) -> None:

        if isinstance(path_root_folder, str):
            path_root_folder = Path(path_root_folder)

        files = [file.name for file in path_root_folder.iterdir() if file.is_file()]

        substrate, adsorbate, idx, *_ = path_root_folder.name.split('_')
        idx = int(idx)
        if 'vib' in _:
            is_vib_folder = True
        else:
            is_vib_folder = False
        if 'bad' in _:
            return None

        if is_vib_folder:
            output_phonons = Output.from_file(path_root_folder / self.output_name)
            if (output_phonons.phonons['zero'] is not None and any(output_phonons.phonons['zero'] > 1e-5)) or \
                    (output_phonons.phonons['imag'] is not None and any(np.abs(output_phonons.phonons['imag']) > 1e-5)):
                print(colored(str(path_root_folder), color='yellow', attrs=['bold']))
                if output_phonons.phonons['zero'] is not None:
                    print(f'{len(output_phonons.phonons["zero"])} zero modes: {output_phonons.phonons["zero"]}')
                if output_phonons.phonons['imag'] is not None:
                    print(f'{len(output_phonons.phonons["imag"])} imag modes: {output_phonons.phonons["imag"]}')
            output = None
        else:
            output = Output.from_file(path_root_folder / self.output_name)
            output_phonons = None

        if not is_vib_folder:

            if 'POSCAR' not in files or recreate_files:
                print('Create POSCAR for\t', colored(str(path_root_folder), attrs=['bold']))
                poscar = output.get_poscar()
                poscar.to_file(path_root_folder / 'POSCAR')

            if 'CONTCAR' not in files or recreate_files:
                print('Create CONTCAR for\t', colored(str(path_root_folder), attrs=['bold']))
                contcar = output.get_contcar()
                contcar.to_file(path_root_folder / 'CONTCAR')

            if 'XDATCAR' not in files or recreate_files:
                print('Create XDATCAR for\t', colored(str(path_root_folder), attrs=['bold']))
                xdatcar = output.get_xdatcar()
                xdatcar.to_file(path_root_folder / 'XDATCAR')

            fft_box_size = output.fft_box_size
            if 'output_volumetric.out' in files:
                files.remove('output_volumetric.out')
                patterns = {'fft_box_size': r'Chosen fftbox size, S = \[(\s+\d+\s+\d+\s+\d+\s+)\]'}
                matches = regrep(str(path_root_folder / 'output_volumetric.out'), patterns)
                fft_box_size = np.array([int(i) for i in matches['fft_box_size'][0][0][0].split()])

            if 'valence_density.cube' not in files or recreate_files:
                print('Create valence(spin)_density for\t', colored(str(path_root_folder), attrs=['bold']))
                if f'{self.jdftx_prefix}.n_up' in files and f'{self.jdftx_prefix}.n_dn' in files:
                    n_up = VolumetricData.from_file(path_root_folder / f'{self.jdftx_prefix}.n_up',
                                                    fft_box_size,
                                                    output.structure).convert_to_cube()
                    n_dn = VolumetricData.from_file(path_root_folder / f'{self.jdftx_prefix}.n_dn',
                                                    fft_box_size,
                                                    output.structure).convert_to_cube()
                    n = n_up + n_dn
                    n.to_file(path_root_folder / 'valence_density.cube')

                    if output.magnetization_abs > 1e-2:
                        n = n_up - n_dn
                        n.to_file(path_root_folder / 'spin__density.cube')

                elif f'{self.jdftx_prefix}.n' in files:
                    n = VolumetricData.from_file(path_root_folder / f'{self.jdftx_prefix}.n',
                                                 fft_box_size,
                                                 output.structure).convert_to_cube()
                    n.to_file(path_root_folder / 'valence_density.cube')
                else:
                    print(colored('(!) There is no files for valence(spin)_density.cube creation',
                                  color='red', attrs=['bold']))

            if ('nbound.cube' not in files or recreate_files) and f'{self.jdftx_prefix}.nbound' in files:
                print('Create nbound.cube for\t', colored(str(path_root_folder), attrs=['bold']))
                nbound = VolumetricData.from_file(path_root_folder / f'{self.jdftx_prefix}.nbound',
                                                  fft_box_size, output.structure).convert_to_cube()
                nbound.to_file(path_root_folder / 'nbound.cube')

            for file in files:
                if file.startswith(f'{self.jdftx_prefix}.fluidN_'):
                    fluid_type = file.removeprefix(self.jdftx_prefix + '.')
                    if f'{fluid_type}.cube' not in files or recreate_files:
                        print(f'Create {fluid_type}.cube for\t', colored(str(path_root_folder), attrs=['bold']))
                        fluidN = VolumetricData.from_file(path_root_folder / file,
                                                          fft_box_size,
                                                          output.structure).convert_to_cube()
                        fluidN.to_file(path_root_folder / (fluid_type + '.cube'))

            if self.ddec_params is not None and ('job_control.txt' not in files or recreate_files):
                print('Create job_control.txt for', colored(str(path_root_folder), attrs=['bold']))
                charge = - (output.nelec_hist[-1] - output.nelec_pzc)
                self.create_job_control(filepath=path_root_folder / 'job_control.txt',
                                        charge=charge,
                                        ddec_params=self.ddec_params)

            if 'ACF.dat' in files:
                nac_bader = ACF.from_file(path_root_folder / 'ACF.dat')
            elif self.do_bader:
                print('Run Bader for', colored(str(path_root_folder), attrs=['bold']))

                subprocess.run(["wsl", "~", "-e", "mkdir", path_root_folder.name])
                sp = subprocess.run(["wsl", "../bader", "-c", "bader", "-i", "cube",
                                     f"/{WslPath(path_root_folder).wsl_path}/valence_density.cube"],
                                    capture_output=True,
                                    cwd=rf'\\wsl.localhost\Ubuntu-22.04\home\v_linux\{path_root_folder.name}')

                if 'WRITING BADER ATOMIC CHARGES TO ACF.dat' not in sp.stdout.decode('UTF-8'):
                    print(colored('Can not do Bader charge analysis for', color='red', attrs=['bold']),
                          path_root_folder)
                    nac_bader = None
                else:
                    subprocess.run(["wsl", "~", "-e", "cp", f"{path_root_folder.name}/ACF.dat",
                                    f"/{WslPath(path_root_folder).wsl_path}/ACF.dat"])
                    subprocess.run(["wsl", "~", "-e", "cp", f"{path_root_folder.name}/AVF.dat",
                                    f"/{WslPath(path_root_folder).wsl_path}/AVF.dat"])
                    subprocess.run(["wsl", "~", "-e", "cp", f"{path_root_folder.name}/BCF.dat",
                                    f"/{WslPath(path_root_folder).wsl_path}/BCF.dat"])
                    subprocess.run(["wsl", "~", "-e", "rm", "-r", path_root_folder.name])
                    nac_bader = ACF.from_file(path_root_folder / 'ACF.dat')
            else:
                nac_bader = None

            if not recreate_files and 'valence_cube_DDEC_analysis.output' in files:
                nac_ddec = Output_DDEC.from_file(path_root_folder / 'valence_cube_DDEC_analysis.output')
            elif self.ddec_params is not None and self.do_ddec:
                print('Run DDEC for', colored(str(path_root_folder), attrs=['bold']))
                start = timer()

                p = Popen(str(self.ddec_params['path_ddec_executable']), stdin=PIPE, bufsize=0)
                p.communicate(str(path_root_folder).encode('ascii'))
                end = timer()
                print(f'Done! Elapsed time: {timedelta(seconds=end-start)}')
                nac_ddec = Output_DDEC.from_file(path_root_folder / 'valence_cube_DDEC_analysis.output')
            else:
                nac_ddec = None

            if f'{self.jdftx_prefix}.eigenvals' in files and f'{self.jdftx_prefix}.kPts' in files:
                eigs = Eigenvals.from_file(path_root_folder / f'{self.jdftx_prefix}.eigenvals',
                                           output)
                kpts = kPts.from_file(path_root_folder / f'{self.jdftx_prefix}.kPts')
                if f'{self.jdftx_prefix}.fillings' in files:
                    occs = Fillings.from_file(path_root_folder / f'{self.jdftx_prefix}.fillings',
                                              output).occupations
                else:
                    occs = None
                dos = DOS(eigenvalues=eigs.eigenvalues * Hartree2eV,
                          weights=kpts.weights,
                          efermi=output.mu * Hartree2eV,
                          occupations=occs)
            else:
                dos = None

            if f'output_phonon.out' in files:
                output_phonons = Output.from_file(path_root_folder / 'output_phonon.out')
                if (output_phonons.phonons['zero'] is not None and any(output_phonons.phonons['zero'] > 1e-5)) or \
                        (output_phonons.phonons['imag'] is not None and any(
                            np.abs(output_phonons.phonons['imag']) > 1e-5)):
                    print(colored(str(path_root_folder), color='yellow', attrs=['bold']))
                    if output_phonons.phonons['zero'] is not None:
                        print(f'{len(output_phonons.phonons["zero"])} zero modes: {output_phonons.phonons["zero"]}')
                    if output_phonons.phonons['imag'] is not None:
                        print(f'{len(output_phonons.phonons["imag"])} imag modes: {output_phonons.phonons["imag"]}')

        else:
            nac_ddec = None
            nac_bader = None
            dos = None

        self.lock.acquire()

        system_proccessed = self.get_system(substrate, adsorbate, idx)
        if len(system_proccessed) == 1:
            if is_vib_folder:
                system_proccessed[0]['output_phonons'] = output_phonons
            elif output_phonons is not None:
                system_proccessed[0]['output_phonons'] = output_phonons
            else:
                system_proccessed[0]['output'] = output
                system_proccessed[0]['nac_ddec'] = nac_ddec
                system_proccessed[0]['dos'] = dos
                system_proccessed[0]['nac_bader'] = nac_bader

            self.lock.release()

        elif len(system_proccessed) == 0:
            system: System = {'substrate': substrate,
                              'adsorbate': adsorbate,
                              'idx': idx,
                              'output': output,
                              'nac_ddec': nac_ddec,
                              'output_phonons': output_phonons,
                              'dos': dos,
                              'nac_bader': nac_bader}

            self.systems.append(system)
            self.lock.release()

        else:
            self.lock.release()
            raise ValueError(f'There should be 0 ot 1 copy of the system in the InfoExtractor.'
                             f'However there are {len(system_proccessed)} systems copies of following system: '
                             f'{substrate=}, {adsorbate=}, {idx=}')

    def get_system(self, substrate: str, adsorbate: str, idx: int = None) -> list[System]:
        if idx is None:
            return [system for system in self.systems if
                    system['substrate'] == substrate and system['adsorbate'] == adsorbate]
        else:
            return [system for system in self.systems if
                    system['substrate'] == substrate and system['adsorbate'] == adsorbate and system['idx'] == idx]

    def get_F(self, substrate: str, adsorbate: str, idx: int,
              units: Literal['eV', 'Ha'] = 'eV',
              T: float | int = None) -> float:

        if T is None:
            E = self.get_system(substrate, adsorbate, idx)[0]['output'].energy_ionic_hist['F'][-1]
            if units == 'Ha':
                return E
            elif units == 'eV':
                return E * Hartree2eV
            else:
                raise ValueError(f'units should be "Ha" or "eV" however "{units}" was given')
        elif isinstance(T, float | int):
            E = self.get_system(substrate, adsorbate, idx)[0]['output'].energy_ionic_hist['F'][-1]
            E_vib = self.get_E_vib_tot(substrate, adsorbate, idx, T)
            if units == 'Ha':
                return E + E_vib * eV2Hartree
            elif units == 'eV':
                return E * Hartree2eV + E_vib
            else:
                raise ValueError(f'units should be "Ha" or "eV" however "{units}" was given')
        else:
            raise ValueError(f'T should be None, float or int, but {type(T)} was given')

    def get_G(self, substrate: str, adsorbate: str, idx: int,
              units: Literal['eV', 'Ha'] = 'eV',
              T: float | int = None) -> float:
        if T is None:
            E = self.get_system(substrate, adsorbate, idx)[0]['output'].energy_ionic_hist['G'][-1]
            if units == 'Ha':
                return E
            elif units == 'eV':
                return E * Hartree2eV
            else:
                raise ValueError(f'units should be "Ha" or "eV" however "{units}" was given')
        elif isinstance(T, float | int):
            E = self.get_system(substrate, adsorbate, idx)[0]['output'].energy_ionic_hist['G'][-1]
            E_vib = self.get_E_vib_tot(substrate, adsorbate, idx, T)
            if units == 'Ha':
                return E + E_vib * eV2Hartree
            elif units == 'eV':
                return E * Hartree2eV + E_vib
            else:
                raise ValueError(f'units should be "Ha" or "eV" however "{units}" was given')
        else:
            raise ValueError(f'T should be None, float or int, but {type(T)} was given')

    def get_N(self, substrate: str, adsorbate: str, idx: int) -> float:
        return self.get_system(substrate, adsorbate, idx)[0]['output'].nelec

    def get_mu(self, substrate: str, adsorbate: str, idx: int) -> float:
        return self.get_system(substrate, adsorbate, idx)[0]['output'].mu

    def get_E_vib_tot(self, substrate: str, adsorbate: str, idx: int, T: float) -> float:
        return self.get_system(substrate, adsorbate, idx)[0]['output_phonons'].thermal_props.get_E_tot(T)

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

            delta_ionic_energy = (out.energy_ionic_hist['F'] - out.energy_ionic_hist['F'][-1]) * Hartree2eV
            if (delta_ionic_energy < 0).any():
                energy_modulus_F = True
                delta_ionic_energy = np.abs(delta_ionic_energy)
            else:
                energy_modulus_F = False

            ax_e.plot(range(out.nisteps), delta_ionic_energy, color='r', label=r'$\Delta F$', ms=3, marker='o')

            if 'G' in out.energy_ionic_hist.keys():
                delta_ionic_energy = (out.energy_ionic_hist['G'] - out.energy_ionic_hist['G'][-1]) * Hartree2eV
                if (delta_ionic_energy < 0).any():
                    energy_modulus_G = True
                    delta_ionic_energy = np.abs(delta_ionic_energy)
                else:
                    energy_modulus_G = False
            else:
                energy_modulus_G = None

            ax_e.plot(range(out.nisteps), delta_ionic_energy, color='orange', label=r'$\Delta G$', ms=3, marker='o')

            ax_e.set_yscale('log')
            ax_e.set_xlabel(r'$Step$', fontsize=12)
            if energy_modulus_F:
                ylabel = r'$|\Delta F|, \ $'
            else:
                ylabel = r'$\Delta F, \ $'
            if energy_modulus_G is not None:
                if energy_modulus_G:
                    ylabel += r'$|\Delta G|, \ $'
                else:
                    ylabel += r'$\Delta G, \ $'
            ylabel += r'$eV$'

            ax_e.set_ylabel(ylabel, color='r', fontsize=14)
            ax_e.legend(loc='upper right', fontsize=14)

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
            ax_f.legend(loc='upper right', bbox_to_anchor=(1, 0.8), fontsize=13)
            ax_f.set_yscale('log')


def create_z_displacements(folder_source: str | Path,
                           folder_result: str | Path,
                           n_atoms_mol: int,
                           scan_range: NDArray[Shape['Nsteps'], Number] | list[float],
                           create_flat_surface: bool = False,
                           folder_files_to_copy: str | Path = None) -> None:
    """
    Create folder with all necessary files for displacing the selected atoms along z-axis
    Args:
        folder_source: path for the folder with .lattice and .ionpos JDFTx files that will be initial files for configurations
        folder_result: path for the folder where all final files will be saved
        n_atoms_mol: number of atoms that should be displaced. All atoms must be in the end atom list in .ionpos
        scan_range: array with displacement (in angstroms) for the selected atoms
        create_flat_surface: if True all atoms will be projected into graphene surface; if False all atoms except molecules remain at initial positions
        folder_files_to_copy: path for the folder with input.in and run.sh files to copy into each folder with final configurations
    """
    if isinstance(folder_source, str):
        folder_source = Path(folder_source)
    if isinstance(folder_result, str):
        folder_result = Path(folder_result)
    if isinstance(folder_files_to_copy, str):
        folder_files_to_copy = Path(folder_files_to_copy)

    substrate, adsorbate, idx, *_ = folder_source.name.split('_')
    lattice = Lattice.from_file(folder_source / 'jdft.lattice')

    for d_ang in scan_range:
        d_ang = np.round(d_ang, 2)
        d_bohr = d_ang * Angstrom2Bohr

        ionpos = Ionpos.from_file(folder_source / 'jdft.ionpos')

        Path(folder_result / f'{substrate}_{adsorbate}_{idx}/{d_ang}').mkdir(parents=True, exist_ok=True)
        ionpos.coords[-n_atoms_mol:, 2] += d_bohr

        idx_surf = [i for i, coord in enumerate(ionpos.coords) if np.abs(coord[0]) < 1 or np.abs(coord[1]) < 1]
        z_carbon = np.mean(ionpos.coords[idx_surf], axis=0)[2]

        if create_flat_surface:
            ionpos.coords[:-n_atoms_mol, 2] = z_carbon
        else:
            ionpos.coords[idx_surf, 2] = z_carbon

        ionpos.move_scale[-n_atoms_mol:] = 0
        ionpos.move_scale[idx_surf] = 0

        ionpos.to_file(folder_result / f'{substrate}_{adsorbate}_{idx}/{d_ang}/jdft.ionpos')
        lattice.to_file(folder_result / f'{substrate}_{adsorbate}_{idx}/{d_ang}/jdft.lattice')
        poscar = ionpos.convert('vasp', lattice)
        poscar.to_file(folder_result / f'{substrate}_{adsorbate}_{idx}/{d_ang}/POSCAR')

        shutil.copyfile(folder_files_to_copy / 'input.in', folder_result / f'{substrate}_{adsorbate}_{idx}/{d_ang}/input.in')
        shutil.copyfile(folder_files_to_copy / 'run.sh', folder_result / f'{substrate}_{adsorbate}_{idx}/{d_ang}/run.sh')
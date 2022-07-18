from ..io_data import jdftx
from ..io_data import vasp
from ..core.constants import Bohr2Angstrom
import os
from monty.re import regrep
import numpy as np


def analise_all(folder: str,
                nelec_PZC: float = None):

    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    slurm = [file for file in files if file.startswith('slurm')]
    for item in slurm:
        files.remove(item)
    files_processed = []
    class_instances = []

    for file in files:
        if file == 'output.out':
            output = jdftx.Output.from_file(os.path.join(folder, 'output.out'))
            class_instances.append(output)
            files_processed.append('Output')
            print(folder, output.nelec_hist[-1])
            fft_box_size = output.fft_box_size
        elif file.endswith('.ionpos'):
            ionpos = jdftx.Ionpos.from_file(os.path.join(folder, file))
            class_instances.append(ionpos)
            files_processed.append('Ionpos')
        elif file.endswith('.ionpos_start'):
            ionpos = jdftx.Ionpos.from_file(os.path.join(folder, file))
            class_instances.append(ionpos)
            files_processed.append('Ionpos_start')
        elif file.endswith('.lattice'):
            lattice = jdftx.Lattice.from_file(os.path.join(folder, file))
            class_instances.append(lattice)
            files_processed.append('Lattice')

    if 'output_fft.out' in files:
        files.remove('output_fft.out')
        patterns = {'fft_box_size': r'Chosen fftbox size, S = \[(\s+\d+\s+\d+\s+\d+\s+)\]'}
        matches = regrep(os.path.join(folder, 'output_fft.out'), patterns)
        fft_box_size = np.array([int(i) for i in matches['fft_box_size'][0][0][0].split()])

    for file in files:

        if file.endswith('.n_up'):
            n_up = jdftx.VolumetricData.from_file(os.path.join(folder, file),
                                                  fft_box_size,
                                                  output.structure).convert_to_cube()
            class_instances.append(n_up)
            files_processed.append('n_up')
        elif file.endswith('.n_dn'):
            n_dn = jdftx.VolumetricData.from_file(os.path.join(folder, file),
                                                  fft_box_size,
                                                  output.structure).convert_to_cube()
            class_instances.append(n_dn)
            files_processed.append('n_dn')
        elif file.endswith('.nbound'):
            nbound = jdftx.VolumetricData.from_file(os.path.join(folder, file),
                                                    fft_box_size,
                                                    output.structure).convert_to_cube()
            class_instances.append(nbound)
            files_processed.append('nbound')

    if 'Output' in files_processed:
        poscar = class_instances[files_processed.index('Output')].get_poscar()
        poscar.to_file(os.path.join(folder, 'POSCAR'))
        poscar = class_instances[files_processed.index('Output')].get_contcar()
        poscar.to_file(os.path.join(folder, 'CONTCAR'))
        poscar = class_instances[files_processed.index('Output')].get_xdatcar()
        poscar.to_file(os.path.join(folder, 'XDATCAR'))

    if 'n_up' in files_processed and 'n_dn' in files_processed:
        n = class_instances[files_processed.index('n_up')] + class_instances[files_processed.index('n_dn')]
        n.to_file(os.path.join(folder, 'valence_density.cube'))
        #n = class_instances[files_processed.index('n_up')] - class_instances[files_processed.index('n_dn')]
        #n.to_file(os.path.join(folder, 'spin_density.cube'))
    else:
        if 'n_up' in files_processed:
            class_instances[files_processed.index('n_up')].to_file(os.path.join(folder, 'n_up.cube'))
        if 'n_dn' in files_processed:
            class_instances[files_processed.index('n_dn')].to_file(os.path.join(folder, 'n_dn.cube'))

    if 'nbound' in files_processed:
        class_instances[files_processed.index('nbound')].to_file(os.path.join(folder, 'nbound.cube'))
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

    if nelec_PZC is not None:
        charge = - (class_instances[files_processed.index('Output')].nelec_hist[-1] - nelec_PZC)

        job_control = open(os.path.join(folder, 'job_control.txt'), 'w')

        job_control.write('<net charge>\n')
        job_control.write(f'{charge}\n')
        job_control.write(text)
        job_control.close()
    else:
        charge = 0.0
        job_control = open(os.path.join(folder, 'job_control.txt'), 'w')

        job_control.write('<net charge>\n')
        job_control.write(f'{charge}\n')
        job_control.write(text)
        job_control.close()

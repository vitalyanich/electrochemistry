from ..io_data import jdftx
import os


def analise_all(folder):
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
            files.remove(file)
        elif file.endswith('.ionpos'):
            ionpos = jdftx.Ionpos.from_file(os.path.join(folder, file))
            class_instances.append(ionpos)
            files_processed.append('Ionpos')
            files.remove(file)
        elif file.endswith('.lattice'):
            lattice = jdftx.Lattice.from_file(os.path.join(folder, file))
            class_instances.append(lattice)
            files_processed.append('Lattice')
            files.remove(file)

    for file in files:

        if file.endswith('.n_up'):
            n_up = jdftx.VolumetricData.from_file(os.path.join(folder, file),
                                                  output.fft_box_size,
                                                  output.structure).convert_to_cube()
            class_instances.append(n_up)
            files_processed.append('n_up')
        elif file.endswith('.n_dn'):
            n_dn = jdftx.VolumetricData.from_file(os.path.join(folder, file),
                                                  output.fft_box_size,
                                                  output.structure).convert_to_cube()
            class_instances.append(n_dn)
            files_processed.append('n_dn')
        elif file.endswith('.nbound'):
            nbound = jdftx.VolumetricData.from_file(os.path.join(folder, file),
                                                  output.fft_box_size,
                                                  output.structure).convert_to_cube()
            class_instances.append(nbound)
            files_processed.append('nbound')

    if 'Ionpos' in files_processed and 'Lattice' in files_processed:
        poscar = class_instances[files_processed.index('Ionpos')].convert('vasp', class_instances[files_processed.index('Lattice')])
        poscar.to_file(os.path.join(folder, 'CONTCAR'))

    if 'n_up' in files_processed and 'n_dn' in files_processed:
        n = class_instances[files_processed.index('n_up')] + class_instances[files_processed.index('n_dn')]
        n.to_file(os.path.join(folder, 'valence_density.cube'))
        n = class_instances[files_processed.index('n_up')] - class_instances[files_processed.index('n_dn')]
        n.to_file(os.path.join(folder, 'spin_density.cube'))
    else:
        if 'n_up' in files_processed:
            class_instances[files_processed.index('n_up')].to_file(os.path.join(folder, 'n_up.cube'))
        if 'n_dn' in files_processed:
            class_instances[files_processed.index('n_dn')].to_file(os.path.join(folder, 'n_dn.cube'))

    if 'nbound' in files_processed:
        class_instances[files_processed.index('nbound')].to_file(os.path.join(folder, 'nbound.cube'))

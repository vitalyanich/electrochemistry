from ..io_data import jdftx
import os


def analise_all(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    slurm = [file for file in files if file.startswith('slurm')]
    for item in slurm:
        files.remove(item)
    files_processed = []
    class_instances = []

    if 'output.out' in files:
        output = jdftx.Output.from_file(os.path.join(folder, 'output.out'))
        class_instances.append(output)
        files_processed.append('Output')

    for file in files:
        if 'Output' in files_processed:
            if file.endswith('.n_up'):
                n_up = jdftx.VolumetricData.from_file(os.path.join(folder, file), output).convert_to_cube()
                class_instances.append(n_up)
                files_processed.append('n_up')
            elif file.endswith('.n_dn'):
                n_dn = jdftx.VolumetricData.from_file(os.path.join(folder, file), output).convert_to_cube()
                class_instances.append(n_dn)
                files_processed.append('n_dn')
            elif file.endswith('.nbound'):
                nbound = jdftx.VolumetricData.from_file(os.path.join(folder, file), output).convert_to_cube()
                class_instances.append(nbound)
                files_processed.append('nbound')

        if file.endswith('.ionpos'):
            ionpos = jdftx.Ionpos.from_file(os.path.join(folder, file))
            class_instances.append(ionpos)
            files_processed.append('Ionpos')
        elif file.endswith('.lattice'):
            lattice = jdftx.Lattice.from_file(os.path.join(folder, file))
            class_instances.append(lattice)
            files_processed.append('Lattice')
        #elif file.endswith('.eigenvals'):
        #    eigenvals = jdftx.Eigenvals.from_file(os.path.join(folder, file), output)
        #    class_instances.append(eigenvals)
        #    files_processed.append('Eigenvals')

    if 'Ionpos' in files_processed and 'Lattice' in files_processed:
        poscar = class_instances[files_processed.index('Ionpos')].convert('vasp', class_instances[files_processed.index('Lattice')])
        poscar.to_file(os.path.join(folder, 'CONTCAR'))
    if 'n_up' in files_processed:
        class_instances[files_processed.index('n_up')].to_file(os.path.join(folder, 'n_up.cube'))
    if 'n_dn' in files_processed:
        class_instances[files_processed.index('n_dn')].to_file(os.path.join(folder, 'n_dn.cube'))
        #if 'n_up' in files_processed:
        #    n_spin = class_instances[files_processed.index('n_up')] - class_instances[files_processed.index('n_dn')]
        #    n_spin.to_file(os.path.join(folder, 'n_spin.cube'))
    if 'nbound' in files_processed:
        class_instances[files_processed.index('nbound')].to_file(os.path.join(folder, 'nbound.cube'))

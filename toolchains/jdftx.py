from ..io_data.jdftx import Output, Eigenvals, Ionpos
import os


def analise_all(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    files_processed = []
    for file in files:
        if file.endswith('.out'):
            output = Output.from_file(os.path.join(folder, file))
            files_processed.append(output)
        if file.endswith('.ionpos'):
            ionpos = Ionpos.from_file(os.path.join(folder, file))
            files_processed.append(ionpos)
        if file.endswith('.eigenvals'):
            eigenvals = Eigenvals.from_file(os.path.join(folder, file))
            files_processed.append(eigenvals)

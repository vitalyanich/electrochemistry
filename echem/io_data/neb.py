from monty.re import regrep
import matplotlib.pyplot as plt
from pathlib import Path
from echem.io_data.jdftx import Output
from echem.core.constants import Hartree2eV
import numpy as np

def get_energies_from_logs(folderpath, plot=False, dpi=200):
    patterns = {'en': r'(\d+).(\d+)\s+Current Energy:\s+(.\d+\.\d+)', 'iter': r'Now starting iteration (\d+) on\s+\[(.+)\]'}
    NEBlogpath = Path(folderpath) / 'logfile_NEB.log'
    pylogpath = Path(folderpath) / 'py.log'
    matches_neb = regrep(str(NEBlogpath), patterns)
    matches_py = regrep(str(pylogpath), patterns)
    iterations_number = len(matches_py['iter'])
    energies = []
    n_images_list = []
    for i in range(iterations_number):
        energies.append([])
        images_list = matches_py['iter'][i][0][1].split(', ')
        energies[i] = {key: [] for key in images_list}
        n_images = len(images_list)
        n_images_list.append(n_images)
    for i in range(len(matches_neb['en'])):
        iteration = int(matches_neb['en'][i][0][0])
        image = matches_neb['en'][i][0][1]
        energies[iteration - 1][image].append(float(matches_neb['en'][i][0][2]))
    if plot:
        max_i = 0
        for i in range(len(energies)):
            plt.figure(dpi=dpi)
            barrier = []
            all_images = []
            for image in energies[i].keys():
                if int(image) > max_i:
                    max_i = int(image)
                plt.scatter([int(image) for _ in range(len(energies[i][image]))], energies[i][image], c=f'C{int(image)}')
                if len(energies[i][image]) != 0:
                    plt.scatter(int(image), energies[i][image][-1], c=f'C{int(image)}')
                    barrier.append(energies[i][image][-1])
                    all_images.append(int(image))
                plt.plot(all_images, barrier, c='black')
        return plt, energies
    else:
        return energies

def get_energies_from_outs(folderpath, opt_history=False, plot=False, dpi=200):
    folderpath /= 'iterations'
    neb_barriers_hist = []
    neb_barriers = []
    for iter, iter_path in enumerate(folderpath.glob('iter_*')):
        neb_barriers.append([])
        neb_barriers_hist.append([])
        for f_path in iter_path.glob('[0-9]'):
            out = Output.from_file(f_path / 'output.out')
            if opt_history:
                neb_barriers_hist[iter].append(out.energy_ionic_hist['G'] * Hartree2eV)
            neb_barriers[iter].append(out.energy_ionic_hist['G'][-1] * Hartree2eV)
    if plot:
        if opt_history:
            for i, barrier in enumerate(neb_barriers_hist):
                plt.figure(dpi=dpi)
                plt.title(f'Iteration {i}')
                for i, traj in enumerate(barrier):
                    plt.plot(traj, label=i)
                plt.legend(frameon=False)
            plt.figure(dpi=dpi)
            for i, barrier in enumerate(neb_barriers):
                plt.plot(barrier, label=i)
            plt.legend(frameon=False)
            return plt, neb_barriers, neb_barriers_hist
        else:
            plt.figure(dpi=dpi)
            for i, barrier in enumerate(neb_barriers):
                plt.plot(barrier, label=i)
            plt.legend(frameon=False)
            return plt, neb_barriers
    else:
        return neb_barriers

def get_energies_from_pylog(filepath, plot=False, dpi=200):
    energies = []
    with open(filepath) as f:
        data = f.readlines()
    for line in data:
        if 'Energies after iteration' in line:
            energies.append(list(map(float, line.strip().split('[')[1][:-1].split(', '))))
    if plot:
        plt.figure(dpi=dpi)
        for i, e in enumerate(energies):
            plt.plot(e, label=i)
        plt.legend(frameon=False)
        return plt, energies
    else:
        return energies

def get_energies_from_NEBlog(folderpath, plot=False, dpi=200):
    patterns = {'en': r'(\d+)\s+Current Energy:\s+(.\d+\.\d+)', \
                'images': r'Successfully initialized JDFTx calculator(.+)/(\d+)'}
    NEBlogpath = Path(folderpath) / 'logfile_NEB.log'
    matches_neb = regrep(str(NEBlogpath), patterns)
    nimages = len(matches_neb['images'])
    energies = [[] for i in range(nimages)]
    for i in range(len(matches_neb['en'])):
        image = int(matches_neb['en'][i][0][0])
        energies[image-1].append(float(matches_neb['en'][i][0][1]))
    if plot:
        plt.figure(dpi=dpi)
        barrier = []
        all_images = []
        for image in range(len(energies)):
            plt.scatter([image for _ in range(len(energies[image]))], energies[image], c=f'C{image}')
            barrier.append(energies[image][-1])
            all_images.append(int(image))
            plt.plot(all_images, barrier, c='black')
        return plt, energies
    else:
        return energies
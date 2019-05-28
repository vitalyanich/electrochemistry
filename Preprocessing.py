from monty.re import regrep
import itertools
import numpy as np
from pymatgen.io.vasp.outputs import Locpot
#from vaspwfc import vaspwfc
import os, sys, time
import multiprocessing as mp
import matplotlib.pyplot as plt


class Preprocessing():
    """
    This class is for basic preprocessing VASP output files (WAVECAR, OUTCAR, LOCPOT)
    Main goal for this class is to extract necessary date from VASP files and save data to .npy files:
    efermi - Fermi level
    eigenvalues - eigenvalues for each kpoint and band
    ndands - number of bands
    nkpts - number of kpoints
    occupations - occupations for each kpoint and band
    weights - weights for each kpoint
    WF_ijke - "effective" wavefunction 3D grids for each small energy range from E to E+dE
    """

    def __init__(self):
        """
        Initialization of variables
        """
        self.efermi = None
        self.nkpts = None
        self.nbands = None
        self.weights = None
        self.eigenvalues = None
        self.occupations = None
        self.vacuum_lvl = None

    def get_wavefunction(self, mesh_shape, energies, kb_array, weights, arr_real, arr_imag, done_number, wavecar_path='WAVECAR'):
        """
        This function is inner function for processes.
        It extract wavefunction from WAVECAR using vaspwfc library (see. https://github.com/QijingZheng/VaspBandUnfolding)
        :param mesh_shape: shape of wavefucntion mesh (tuple of 3 int)
        :param energies: list of energies for current process to be done (list of int)
        :param kb_array: array of kpoints and bands which lay in current energy range (np.array of tuples (k,b) depends on energies)
        :param weights: np.array of float64 containes weights of different k-points (np.array of float depends on energies)
        :param arr_real: shared memory float64 array to put real part of wavefunction
        :param arr_imag: shared memory float64 array to put imaginary part of wavefunction
        :param done_number: shared memory int to controll number of done energy ranges
        :param wavecar_path: path to WAVECAR file
        :return:
        """
        t_p = time.time()
        wavecar = vaspwfc(wavecar_path)
        print("Reading WAVECAR takes", time.time()-t_p, " sec")
        sys.stdout.flush()
        for e in energies:
            t = time.time()
            n = mesh_shape[0]*mesh_shape[1]*mesh_shape[2]
            start = e * n
            kb_array_for_e = kb_array[e]
            WF_for_current_Energy_real = np.zeros(mesh_shape)
            WF_for_current_Energy_imag = np.zeros(mesh_shape)
            for kb in kb_array_for_e:
                kpoint = kb[0]
                band = kb[1]
                wf = wavecar.wfc_r(ikpt=kpoint, iband=band, ngrid=wavecar._ngrid * 1.5)
                phi_real = np.real(wf)
                phi_imag = np.imag(wf)
                print ("Wavefunction done from process:", e, "; kpoint = ", kpoint, "; band = ", band)
                sys.stdout.flush()
                WF_for_current_Energy_real += phi_real * weights[kpoint - 1]
                WF_for_current_Energy_imag += phi_imag * weights[kpoint - 1]
            WF_for_current_Energy_1D_real = np.reshape(WF_for_current_Energy_real, n)
            WF_for_current_Energy_1D_imag = np.reshape(WF_for_current_Energy_imag, n)
            for ii in range(n):
                arr_real[ii + start] = float(WF_for_current_Energy_1D_real[ii])
                arr_imag[ii + start] = float(WF_for_current_Energy_1D_imag[ii])
            print ("Energy ", e, " finished, it takes: ", time.time()-t, " sec")
            sys.stdout.flush()
            done_number.value += 1
            print(done_number.value, " energies are done")
            sys.stdout.flush()

    def process_WAVECAR(self, Erange_from_fermi, dE, wavecar_path='WAVECAR', dir_to_save='Saved_data'):
        """
        This function is based on vaspwfc class https://github.com/QijingZheng/VaspBandUnfolding
        This function process WAVECAR file obtained with VASP. It saves 4D np.array WF_ijke.npy
        which containes spacially resolved _complex128 effective wavefunctions for each energy range [E, E+dE]
        'Effective' means that we sum wavefunctions with close energies.
        :param Erange_from_fermi: Energy range for wich we extract wavefunction (tuple or list of float)
        :param dE: Energy step (float)
        :param wavecar_path: path to WAVECAR file (string)
        :param dir_to_save: directory to save WF_ijke.npy file (string)
        :return: nothing
        """
        for var in ['efermi', 'nkpts', 'nbands', 'eigenvalues', 'weights']:
            self.check_existance(var)
        Erange = [Erange_from_fermi[0] + self.efermi, Erange_from_fermi[1] + self.efermi]
        energies_number = int((Erange[1] - Erange[0]) / dE)
        kb_array = [[] for i in range(energies_number)]
        wavecar = vaspwfc(wavecar_path)
        wf = wavecar.wfc_r(ikpt=1, iband=1, ngrid=wavecar._ngrid * 1.5)
        mesh_shape = np.shape(wf)
        n = mesh_shape[0]*mesh_shape[1]*mesh_shape[2]
        wave_functions = np.zeros((energies_number,) + mesh_shape, dtype=np.complex_)
        for band in range(1, self.nbands+1):
            for kpoint in range(1, self.nkpts+1):
                if self.eigenvalues[kpoint - 1][band - 1] > Erange[0] and self.eigenvalues[kpoint - 1][band - 1] < Erange[1]:
                    energy = self.eigenvalues[kpoint - 1][band - 1]
                    e = int((energy - Erange[0]) / dE)
                    kb_array[e].append([kpoint, band])
        t_arr = time.time()
        arr_real = mp.Array('f', [0.0] * n * energies_number)
        arr_imag = mp.Array('f', [0.0] * n * energies_number)
        done_number = mp.Value('i', 0)
        print("Array Created, it takes: ", time.time() - t_arr, " sec")
        processes = []
        print("Total Number Of Energies = ", energies_number)
        numCPU = mp.cpu_count()
        Energies_per_CPU = energies_number // numCPU + 1
        for i in range(numCPU):
            if (i + 1) * Energies_per_CPU > energies_number:
                energies = np.arange(i * Energies_per_CPU, energies_number)
            else:
                energies = np.arange(i * Energies_per_CPU, (i + 1) * Energies_per_CPU)
            p = mp.Process(target=self.get_wavefunction, args=(mesh_shape, energies, kb_array, self.weights, arr_real, arr_imag, done_number, wavecar_path))
            processes.append(p)
        print(numCPU, " CPU Available")
        for step in range(len(processes) // numCPU + 1):
            for i in range(step * numCPU, (step + 1) * numCPU):
                try:
                    processes[i].start()
                except:
                    pass
            for i in range(step * numCPU, (step + 1) * numCPU):
                try:
                    processes[i].join()
                except:
                    pass
        print("All energies DONE! Start wavefunction reshaping")
        arr = np.zeros(n * energies_number, dtype=np.complex_)
        arr.real = arr_real
        arr.imag = arr_imag
        for e in range(energies_number):
            wave_functions[e] = np.reshape(arr[e * n:(e + 1) * n], mesh_shape)

        WF_ijke = np.zeros(mesh_shape + (energies_number,), dtype=np.complex_)
        for i in range(mesh_shape[0]):
            for j in range(mesh_shape[1]):
                for k in range(mesh_shape[2]):
                    for e in range(energies_number):
                        WF_ijke[i][j][k][e] = wave_functions[e][i][j][k]
        np.save(dir_to_save+'/WF_ijke', WF_ijke)
        print(dir_to_save+'/WF_ijke.npy Saved')


    def process_OUTCAR(self, outcar_path='OUTCAR', dir_to_save='Saved_data'):
        """
        process OUTCAR file obtained from VASP
        get following variables:
        self.nkpts - number of k-points (int)
        self.efermi - Fermi level (float)
        self.nbands - number of bands (int)
        self.eigenvalues - 2D np.array, eigenvalues[i][j] contains energy for i k-point and j band
        self.occupations - 2D np.array, occupations[i][j] contains occupation for i k-point and j band
        :param file_path: path to OUTCAR file
        :return: nothing
        """
        patterns = {'nkpts': r'Found\s+(\d+)\s+irreducible\sk-points',
                    'weights':'k-points in units of 2pi/SCALE and weight: K-Points',
                    'efermi': 'E-fermi\s:\s+([-.\d]+)',
                    'kpoints': r'k-point\s+(\d+)\s:\s+[-.\d]+\s+[-.\d]+\s+[-.\d]+\n'}
        matches = regrep(outcar_path, patterns)

        self.nkpts = int(matches['nkpts'][0][0][0])
        self.efermi = float(matches['efermi'][0][0][0])
        self.nbands = int(matches['kpoints'][1][1] - matches['kpoints'][0][1]-3)
        self.eigenvalues = []
        self.occupations = []
        self.weights = []

        with open(outcar_path) as file:
            lines = file.readlines()
            for i in range(self.nkpts):
                self.weights.append(float(lines[matches['weights'][0][1]+i+1].split()[3]))
            for kpoint in range(self.nkpts):
                self.eigenvalues.append([])
                self.occupations.append([])
                startline = matches['kpoints'][kpoint][1]+2
                for i in range(startline, startline + self.nbands):
                    self.eigenvalues[kpoint].append(float(lines[i].split()[1]))
                    self.occupations[kpoint].append(float(lines[i].split()[2]))
        self.eigenvalues = np.array(self.eigenvalues)
        self.occupations = np.array(self.occupations)
        for var in ['efermi', 'nkpts', 'nbands', 'weights', 'eigenvalues', 'occupations']:
            self.save(var, dir_to_save)

    def process_LOCPOT(self, file_path='LOCPOT', dir_to_save='Saved_data'):
        """
        This function process LOCPOT file obtained by VASP
        :param file_path: path to LOCPOT file
        :param dir_to_save: path to directory to save vacuum_lvl
        :return: nothing
        """
        locpot = Locpot.from_file(file_path)
        avr = locpot.get_average_along_axis(2)
        self.vacuum_lvl = np.max(avr)
        self.save('vacuum_lvl', dir_to_save)

    def save(self, variable='all', dir='Saved_data'):
        """
        This function saves variables in .npy files
        :param variable: desired variable
        :param dir: directory name to save file
        :return: nothing
        """
        if variable == 'all':
            for var in ['efermi', 'nkpts', 'nbands', 'weights', 'eigenvalues', 'occupations', 'vacuum_lvl']:
                self.save(var)
        else:
            if not os.path.exists(dir):
                print('Directory ', dir, ' does not exist. Creating directory')
                os.mkdir(dir)
            np.save(dir+'/'+variable+'.npy', getattr(self, variable))
            print('Variable ', variable, ' saved to directory ', dir)

    def load(self, variable='all', dir='Saved_data'):
        """
        This function loads variables from .npy files to class variables
        :param variable: desired variable
        :param dir: directory from which load files
        :return: nothing
        """
        if variable == 'all':
            for var in ['efermi', 'nkpts', 'nbands', 'weights', 'eigenvalues', 'occupations', 'vacuum_lvl']:
                self.load(var)
        else:
            setattr(self, variable, np.load(dir+'/'+str(variable)+'.npy'))
            print('Variable ', variable, ' loaded')

    def check_existance(self, variable='all', dir='Saved_data'):
        """
        This function checks whether desires variable is not None and if necessary load it from file or process VASP data
        :param variable: desired variable
        :param dir: directory in which check .npy saved data
        :return: nothing
        """
        if variable == 'all':
            for var in ['efermi', 'nkpts', 'nbands', 'weights', 'eigenvalues', 'occupations', 'vacuum_lvl']:
                self.check_existance(var)
        else:
            if getattr(self, variable) is None:
                try:
                    print('Try load variable ',variable, ' from dir ', dir)
                    self.load(variable, dir)
                except:
                    if variable == 'vacuum_lvl':
                        print('Loading ', variable, ' failed! Start processing LOCPOT')
                        self.process_LOCPOT()
                    else:
                        print('Loading ', variable, ' failed! Start processing OUTCAR')
                        self.process_OUTCAR()
            else:
                print('Variable ', variable, 'exists')

    def get_band_eigs(self, band, outcar_path='OUTCAR'):
        """
        This function get eigenvalues for desired band
        :param band: desired band
        :param outcar_path: path to OUTCAR file
        :return:
        """
        if self.eigenvalues is None:
            try:
                print("Variable is not define. Try load from file")
                self.load('eigenvalues')
            except:
                print("Loading failed. Start processing OUTCAR")
                self.process_OUTCAR(outcar_path)
            return self.eigenvalues[:,band]
        else:
            return self.eigenvalues[:,band]

    def get_band_occ(self, band, outcar_path='OUTCAR'):
        """
        This function get occupations for desired band
        :param band: desired band
        :param outcar_path: path to OUTCAR file
        :return:
        """
        if self.occupations is None:
            try:
                print("Variable is not define. Try load from file")
                self.load('occupations')
            except:
                print("Loading failed. Start processing OUTCAR")
                self.process_OUTCAR(outcar_path)
            return self.occupations[:,band]
        else:
            return self.occupations[:,band]

    def get_DOS(self, Erange_from_fermi, dE):
        """
        This fuction calculate Density of States for desired energy range and energy step
        :param Erange_from_fermi: Energy range
        :param dE: energy step
        :return:
        E_arr: float64 np.array with energies relative to Fermi level
        DOS_arr: float64 np.array with Density of States
        """
        for var in ['efermi', 'nkpts', 'nbands', 'eigenvalues', 'weights']:
            self.check_existance(var)
        Erange = [Erange_from_fermi[0] + self.efermi, Erange_from_fermi[1] + self.efermi]
        energies_number = int((Erange[1] - Erange[0]) / dE)
        DOS_arr = np.zeros(energies_number)
        E_arr = np.arange(Erange_from_fermi[0], Erange_from_fermi[1], dE)
        for k in range(self.nkpts):
            for b in range(self.nbands):
                energy = self.eigenvalues[k][b]
                if energy > Erange[0] and energy < Erange[1]:
                    e = int((energy - Erange[0]) / dE)
                    DOS_arr[e]+=self.weights[k]/dE
        return E_arr, DOS_arr


class Espresso:
    def __init__(self, filepath):
        self.filepath = filepath
        self.patterns = {"nkpts": r"number of k points=\s+([\d]+)",
                         'kpts_coord': r'k\s*=\s*(-?\d.[\d]+)\s*(-?\d.[\d]+)\s*(-?\d.[\d]+)\s*\([\d]+ PWs\)',
                         'occupations': 'occupation numbers',
                         'efermi': r'the Fermi energy is\s*(-?[\d]+.[\d]+) ev'}
        self.eigenvalues = None
        self.efermi = None

    def read_data(self):
        matches = regrep(self.filepath, self.patterns)
        self.matches = matches

        if len(self.matches['kpts_coord']) != 0:
            with open(self.filepath, 'r') as file:
                file_data = file.readlines()
                self.eigenvalues = []
                for start, end in zip(self.matches['kpts_coord'], self.matches['occupations']):
                    data = file_data[start[1] + 2:end[1] - 1]
                    for i, line in enumerate(data):
                        data[i] = line.split()
                    data = [float(i) for i in itertools.chain.from_iterable(data)]
                    self.eigenvalues.append(data)
                self.eigenvalues = np.array(self.eigenvalues)

                self.occupations = []
                n_strings_occups = self.matches['occupations'][0][1] - self.matches['kpts_coord'][0][1] - 1
                for start in self.matches['occupations']:
                    data = file_data[start[1] + 1: start[1] + n_strings_occups]
                    for i, line in enumerate(data):
                        data[i] = line.split()
                    data = [float(i) for i in itertools.chain.from_iterable(data)]
                    self.occupations.append(data)
                self.occupations = np.array(self.occupations)

        self.efermi = float(self.matches['efermi'][0][0][0])
        self.nkpt = int(self.matches['nkpts'][0][0][0])

    def get_band_eigs(self, bands):
        if type(bands) is int:
            return np.array([eig for eig in self.eigenvalues[:, bands]])
        if (type(bands) is list) or (type(bands) is np.ndarray):
            return np.array([[eig for eig in self.eigenvalues[:, band]] for band in bands])

    def get_band_occ(self, band):
        if type(band) is int:
            return [occ for occ in self.occupations[:, band]]

if __name__=='__main__':
    outcar_path = 'OUTCAR'
    #wavecar_path = 'WAVECAR'
    p = Preprocessing()
    p.process_OUTCAR('OUTCAR_2', 'Saved_data_2')
    E_arr, DOS_arr = p.get_DOS((-10.0, 5.0), 0.1)
    plt.plot(E_arr, DOS_arr)
    plt.show()
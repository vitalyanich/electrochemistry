from monty.re import regrep
from pymatgen.io.vasp.outputs import Locpot
from pymatgen.io.vasp.outputs import Procar
from electrochemistry.core.vaspwfc_p3 import vaspwfc
import numpy as np
import multiprocessing as mp
import os
import sys
import time
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


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

    def __init__(self, working_folder='', print_out=False, dir_to_save=None):
        """
        Initialization of variables
        """
        self.print_out = print_out
        self.working_folder = working_folder
        if dir_to_save == None:
            self.dir_to_save = working_folder+'/Saved_data'
        else:
            self.dir_to_save = dir_to_save
        self.efermi = None
        self.nkpts = None
        self.nbands = None
        self.weights = None
        self.eigenvalues = None
        self.occupations = None
        self.vacuum_lvl = None
        self.procar_data = None
        self.ion_names = None
        self.orbital_names = None
        self.wavecar_path = working_folder+'/WAVECAR'
        self.outcar_path = working_folder+'/OUTCAR'
        self.poscar_path = working_folder+'/POSCAR'
        self.procar_path = working_folder+'/PROCAR'
        self.locpot_path = working_folder+'/LOCPOT'

    def get_wavefunction(self, mesh_shape, energies, kb_array, weights, arr_real, arr_imag, arr_cd, done_number, n_grid, wavecar_path=None):
        """
        This function is inner function for processes.
        It extracts wavefunction from WAVECAR using vaspwfc library (see. https://github.com/QijingZheng/VaspBandUnfolding)
        :param mesh_shape: shape of wavefucntion mesh (tuple of 3 int)
        :param energies: list of energies for current process to be done (list of int)
        :param kb_array: array of kpoints and bands which lay in current energy range (np.array of tuples (k,b) depends on energies)
        :param weights: np.array of float64 containes weights of different k-points (np.array of float depends on energies)
        :param arr_real: shared memory float64 array to put real part of wavefunction
        :param arr_imag: shared memory float64 array to put imaginary part of wavefunction
        :param done_number: shared memory int to controll number of done energy ranges
        :param wavecar_path: path to WAVECAR file
        :param n_grid: float parameter (multipicator) to increase mesh density (1.0 - standart grid, higher values e.g. 1.5, 2.0 - to inrease quality of grid)
        :return:
        """
        if wavecar_path == None:
            wavecar_path = self.wavecar_path
        t_p = time.time()
        wavecar = vaspwfc(wavecar_path)
        #print("Reading WAVECAR takes", time.time()-t_p, " sec")
        #sys.stdout.flush()
        for e in energies:
            #t = time.time()
            n = mesh_shape[0]*mesh_shape[1]*mesh_shape[2]
            start = e * n
            kb_array_for_e = kb_array[e]
            WF_for_current_Energy_real = np.zeros(mesh_shape)
            WF_for_current_Energy_imag = np.zeros(mesh_shape)
            CD_for_current_Energy = np.zeros(mesh_shape)
            for kb in kb_array_for_e:
                kpoint = kb[0]
                band = kb[1]
                wf = wavecar.wfc_r(ikpt=kpoint, iband=band, ngrid=wavecar._ngrid * n_grid)
                phi_real = np.real(wf)
                phi_imag = np.imag(wf)
                phi_squared = np.abs(wf)**2
                #print ("Wavefunction done from process:", e, "; kpoint = ", kpoint, "; band = ", band)
                #sys.stdout.flush()
                WF_for_current_Energy_real += phi_real * np.sqrt(weights[kpoint - 1])
                WF_for_current_Energy_imag += phi_imag * np.sqrt(weights[kpoint - 1])
                CD_for_current_Energy += phi_squared * weights[kpoint - 1]
            WF_for_current_Energy_1D_real = np.reshape(WF_for_current_Energy_real, n)
            WF_for_current_Energy_1D_imag = np.reshape(WF_for_current_Energy_imag, n)
            CD_for_current_Energy_1D = np.reshape(CD_for_current_Energy, n)
            for ii in range(n):
                arr_real[ii + start] = float(WF_for_current_Energy_1D_real[ii])
                arr_imag[ii + start] = float(WF_for_current_Energy_1D_imag[ii])
                arr_cd[ii + start] = float(CD_for_current_Energy_1D[ii])
            #print ("Energy ", e, " finished, it takes: ", time.time()-t, " sec")
            #sys.stdout.flush()
            done_number.value += 1
            if self.print_out == True:
                print(done_number.value, " energies are done")
            sys.stdout.flush()

    def process_WAVECAR(self, Erange_from_fermi, dE, wavecar_path=None, dir_to_save=None, n_grid=1.0):
        """
        This function is based on vaspwfc class https://github.com/QijingZheng/VaspBandUnfolding
        This function process WAVECAR file obtained with VASP. It saves 4D np.array WF_ijke.npy
        which containes spacially resolved _complex128 effective wavefunctions for each energy range [E, E+dE]
        'Effective' means that we sum wavefunctions with close energies.
        UPDATE: The function saves also dictionary WF_data.npy, which contains two keys: 'energy_range','dE'
        :param Erange_from_fermi: Energy range for wich we extract wavefunction (tuple or list of float)
        :param dE: Energy step (float)
        :param wavecar_path: path to WAVECAR file (string)
        :param dir_to_save: directory to save WF_ijke.npy file (string)
        :param n_grid: float parameter (multipicator) to increase mesh density (1.0 - standart grid, higher values e.g. 1.5, 2.0 - to inrease quality of grid)
        :return: nothing
        """
        if wavecar_path == None:
            wavecar_path = self.wavecar_path
        if dir_to_save == None:
            dir_to_save = self.dir_to_save
        for var in ['efermi', 'nkpts', 'nbands', 'eigenvalues', 'weights']:
            self.check_existance(var)
        Erange = [round(Erange_from_fermi[0] + self.efermi, 5), round(Erange_from_fermi[1] + self.efermi, 5)]
        energies_number = int((Erange[1] - Erange[0]) / dE)
        kb_array = [[] for i in range(energies_number)]
        wavecar = vaspwfc(wavecar_path)
        wf = wavecar.wfc_r(ikpt=1, iband=1, ngrid=wavecar._ngrid * n_grid)
        mesh_shape = np.shape(wf)
        n = mesh_shape[0]*mesh_shape[1]*mesh_shape[2]
        for band in range(1, self.nbands+1):
            for kpoint in range(1, self.nkpts+1):
                energy = self.eigenvalues[kpoint - 1][band - 1]
                if energy >= Erange[0] and energy < Erange[1]:
                    e = int((energy - Erange[0]) / dE)
                    kb_array[e].append([kpoint, band])
        t_arr = time.time()
        arr_real = mp.Array('f', [0.0] * n * energies_number)
        arr_imag = mp.Array('f', [0.0] * n * energies_number)
        arr_cd = mp.Array('f', [0.0] * n * energies_number)
        done_number = mp.Value('i', 0)
        if self.print_out == True:
            print("Array Created, it takes: ", time.time() - t_arr, " sec")
        processes = []
        if self.print_out == True:
            print("Total Number Of Energies = ", energies_number)
        numCPU = mp.cpu_count()
        Energies_per_CPU = energies_number // numCPU + 1
        for i in range(numCPU):
            if (i + 1) * Energies_per_CPU > energies_number:
                energies = np.arange(i * Energies_per_CPU, energies_number)
            else:
                energies = np.arange(i * Energies_per_CPU, (i + 1) * Energies_per_CPU)
            p = mp.Process(target=self.get_wavefunction, args=(mesh_shape, energies, kb_array, self.weights, arr_real, arr_imag, arr_cd, done_number, n_grid, wavecar_path))
            processes.append(p)
        if self.print_out == True:
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
        if self.print_out == True:
            print("All energies DONE! Start wavefunction reshaping")
        arr = np.zeros(n * energies_number, dtype=np.complex_)
        arr.real = arr_real
        arr.imag = arr_imag
        wave_functions = np.reshape(arr, (energies_number,) + mesh_shape)
        wave_functions = np.moveaxis(wave_functions, 0, -1)
        
        charge_densities = np.reshape(arr_cd, (energies_number,) + mesh_shape)
        charge_densities = np.moveaxis(charge_densities, 0, -1)
        
        WF_data = {'energy_range':Erange_from_fermi, 'dE':dE}
        np.save(dir_to_save+'/WF_ijke', wave_functions)
        np.save(dir_to_save+'/CD_ijke', charge_densities)
        if self.print_out == True:
            print(dir_to_save+'/WF_ijke.npy Saved')
            print(dir_to_save+'/CD_ijke.npy Saved')
        np.save(dir_to_save+'/WF_data', WF_data)
        if self.print_out == True:
            print(dir_to_save+'/WF_data.npy Saved')


    def process_OUTCAR(self, outcar_path=None, dir_to_save=None, optimization=False):
        """
        process OUTCAR file obtained from VASP
        get following variables:
        self.nkpts - number of k-points (int)
        self.efermi - Fermi level (float)
        self.nbands - number of bands (int)
        self.eigenvalues - 2D np.array, eigenvalues[i][j] contains energy for i k-point and j band
        self.occupations - 2D np.array, occupations[i][j] contains occupation for i k-point and j band
        :param outcar_path: path to OUTCAR file
        :return: nothing
        """
        if outcar_path == None:
            outcar_path = self.outcar_path
        if dir_to_save == None:
            dir_to_save = self.dir_to_save
        patterns = {'nkpts': r'Found\s+(\d+)\s+irreducible\sk-points',
                    'weights':'Following reciprocal coordinates:',
                    'efermi': 'E-fermi\s:\s+([-.\d]+)',
                    'kpoints': r'k-point\s+(\d+)\s:\s+[-.\d]+\s+[-.\d]+\s+[-.\d]+\n'}
        matches = regrep(outcar_path, patterns)

        self.nkpts = int(matches['nkpts'][0][0][0])
        if optimization:
            self.efermi = []
            efermi_data = np.array(matches['efermi'])[...,0]
            number_of_ionic_steps = len(efermi_data)
            for i in range(number_of_ionic_steps):
                self.efermi.append(float(efermi_data[i][0]))
        else:
            self.efermi = float(matches['efermi'][0][0][0])
        self.nbands = int(matches['kpoints'][1][1] - matches['kpoints'][0][1]-3)
        self.eigenvalues = []
        self.occupations = []
        self.weights = []

        with open(outcar_path) as file:
            lines = file.readlines()
            for i in range(self.nkpts):
                self.weights.append(float(lines[matches['weights'][0][1]+i+2].split()[3]))
            if optimization:
                for step in range(number_of_ionic_steps):
                    self.eigenvalues.append([])
                    self.occupations.append([])
                    for kpoint in range(self.nkpts):
                        self.eigenvalues[step].append([])
                        self.occupations[step].append([])
                        startline = matches['kpoints'][kpoint+(step*self.nkpts)][1] + 2
                        for i in range(startline, startline + self.nbands):
                            self.eigenvalues[step][kpoint].append(float(lines[i].split()[1]))
                            self.occupations[step][kpoint].append(float(lines[i].split()[2]))
            else:
                for kpoint in range(self.nkpts):
                    self.eigenvalues.append([])
                    self.occupations.append([])
                    startline = matches['kpoints'][kpoint][1]+2
                    for i in range(startline, startline + self.nbands):
                        self.eigenvalues[kpoint].append(float(lines[i].split()[1]))
                        self.occupations[kpoint].append(float(lines[i].split()[2]))
        self.eigenvalues = np.array(self.eigenvalues)
        self.occupations = np.array(self.occupations)
        self.weights = np.array(self.weights)
        self.weights /= np.sum(self.weights)
        for var in ['efermi', 'nkpts', 'nbands', 'weights', 'eigenvalues', 'occupations']:
            self.save(var, dir_to_save)

    def process_PROCAR(self, procar_path=None, poscar_path=None, dir_to_save=None):
        """
        This function process PROCAR file obtained by VASP using pymatgen.io_data.vasp.outputs.Procar class
        and saves ion_names, orbital_names and data array
        data array contains projections in the following form: data[kpoint][band][ion_number][orbital_number]
        All numberings start from 0
        :param procar_path: path to PROCAR file
        :param poscar_path: path to POSCAR file
        :param dir_to_save: directory to save data
        :return:
        """
        if procar_path == None:
            procar_path = self.procar_path
        if poscar_path == None:
            poscar_path = self.poscar_path
        if dir_to_save == None:
            dir_to_save = self.dir_to_save
        procar=Procar(procar_path)
        for key in procar.data.keys():
            self.procar_data = procar.data[key]
        self.orbital_names = procar.orbitals
        self.ion_names = self._get_atom_types(poscar_path=poscar_path)
        for var in ['procar_data', 'orbital_names', 'ion_names']:
            self.save(var, dir_to_save)

    def _get_atom_types(self, poscar_path=None):
        """
        Inner function to obtain list of atom types from POSCAR
        :param file_path: path to POSCAR file
        :return: atom_types 1D array with atom_types. Index from 0
        """
        if poscar_path == None:
            poscar_path = self.poscar_path
        with open(poscar_path) as inf:
            lines = inf.readlines()
            ion_types = lines[5].strip().split()
            nions = map(int, lines[6].strip().split())
        atom_types=[]
        for i, number in enumerate(nions):
            for j in range(number):
                    atom_types.append(ion_types[i])
        return atom_types

    def process_LOCPOT(self, locpot_path=None, dir_to_save=None):
        """
        This function process LOCPOT file obtained by VASP
        :param file_path: path to LOCPOT file
        :param dir_to_save: path to directory to save vacuum_lvl
        :return: nothing
        """
        if locpot_path == None:
            locpot_path = self.locpot_path
        if dir_to_save == None:
            dir_to_save = self.dir_to_save
        locpot = Locpot.from_file(locpot_path)
        avr = locpot.get_average_along_axis(2)
        self.vacuum_lvl = np.max(avr)
        self.save('vacuum_lvl', dir_to_save)

    def save(self, variable='all', dir=None):
        """
        This function saves variables in .npy files
        :param variable: desired variable
        :param dir: directory name to save file
        :return: nothing
        """
        if dir == None:
            dir = self.dir_to_save
        if variable == 'all':
            for var in ['efermi', 'nkpts', 'nbands', 'weights', 'eigenvalues', 'occupations', 'vacuum_lvl']:
                self.save(var)
        else:
            if not os.path.exists(dir):
                if self.print_out == True:
                    print('Directory ', dir, ' does not exist. Creating directory')
                os.mkdir(dir)
            np.save(dir+'/'+variable+'.npy', getattr(self, variable))
            if self.print_out == True:
                print('Variable ', variable, ' saved to directory ', dir)

    def load(self, variable='all', dir=None):
        """
        This function loads variables from .npy files to class variables
        :param variable: desired variable
        :param dir: directory from which load files
        :return: nothing
        """
        if dir == None:
            dir = self.dir_to_save
        if variable == 'all':
            for var in ['efermi', 'nkpts', 'nbands', 'weights', 'eigenvalues', 'occupations', 'vacuum_lvl']:
                self.load(var)
        else:
            setattr(self, variable, np.load(dir+'/'+str(variable)+'.npy'))
            if self.print_out == True:
                print('Variable ', variable, ' loaded')

    def check_existance(self, variable='all', dir=None):
        """
        This function checks whether desires variable is not None and if necessary load it from file or process VASP data
        :param variable: desired variable
        :param dir: directory in which check .npy saved data
        :return: nothing
        """
        if dir == None:
            dir = self.dir_to_save
        if variable == 'all':
            for var in ['efermi', 'nkpts', 'nbands', 'weights', 'eigenvalues', 'occupations', 'vacuum_lvl']:
                self.check_existance(var)
        else:
            if getattr(self, variable) is None:
                try:
                    if self.print_out == True:
                        print('Try load variable ',variable, ' from dir ', dir)
                    self.load(variable, dir)
                except:
                    if variable == 'vacuum_lvl':
                        if self.print_out == True:
                            print('Loading ', variable, ' failed! Start processing LOCPOT')
                        self.process_LOCPOT()
                    elif variable == 'procar_data' or variable == 'orbital_names' \
                            or variable == 'ion_names':
                        if self.print_out == True:
                            print('Loading ', variable, ' failed! Start processing PROCAR and POSCAR')
                        self.process_PROCAR()
                    else:
                        if self.print_out == True:
                            print('Loading ', variable, ' failed! Start processing OUTCAR')
                        self.process_OUTCAR()
            else:
                if self.print_out == True:
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
                if self.print_out == True:
                    print("Variable is not define. Try load from file")
                self.load('eigenvalues')
            except:
                if self.print_out == True:
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
                if self.print_out == True:
                    print("Variable is not define. Try load from file")
                self.load('occupations')
            except:
                if self.print_out == True:
                    print("Loading failed. Start processing OUTCAR")
                self.process_OUTCAR(outcar_path)
            return self.occupations[:, band]
        else:
            return self.occupations[:, band]

    def get_DOS(self, Erange_from_fermi, dE, optimization=False):
        """
        This function calculates Density of States for desired energy range and energy step
        :param Erange_from_fermi: Energy range
        :param dE: energy step
        :param dE_new : recalculated energy step to enshure integer number of steps
        :return:
        E_arr: float64 np.array with energies relative to Fermi level
        DOS_arr: float64 np.array with Density of States
        """

        for var in ['efermi', 'nkpts', 'nbands', 'eigenvalues', 'weights']:
            self.check_existance(var)
        if optimization:
            DOS_arr = []
            E_arr = []
            for step in range(len(self.efermi)):
                Erange = [Erange_from_fermi[0] + self.efermi[step], Erange_from_fermi[1] + self.efermi[step]]
                energies_number = int((Erange[1] - Erange[0]) / dE)
                dE_new = (Erange[1] - Erange[0]) / energies_number
                DOS_arr.append(np.zeros(energies_number))
                E_arr.append(np.arange(Erange_from_fermi[0], Erange_from_fermi[1], dE_new))
                for k in range(self.nkpts):
                    for b in range(self.nbands):
                        energy = self.eigenvalues[step][k][b]
                        if energy > Erange[0] and energy < Erange[1]:
                            e = int((energy - Erange[0]) / dE_new)
                            DOS_arr[step][e] += self.weights[k] / dE_new
            return E_arr, DOS_arr * 2
        else:
            Erange = [Erange_from_fermi[0] + self.efermi, Erange_from_fermi[1] + self.efermi]
            energies_number = int((Erange[1] - Erange[0]) / dE)
            dE_new = (Erange[1] - Erange[0])/energies_number
            DOS_arr = np.zeros(energies_number)
            E_arr = np.arange(Erange_from_fermi[0], Erange_from_fermi[1], dE_new)
            for k in range(self.nkpts):
                for b in range(self.nbands):
                    energy = self.eigenvalues[k][b]
                    if energy > Erange[0] and energy < Erange[1]:
                        e = int((energy - Erange[0]) / dE_new)
                        DOS_arr[e]+=self.weights[k] / dE_new
            return E_arr, DOS_arr * 2, dE_new

    def get_pdos(self, Erange_from_fermi, dE, ions='all', orbitals='all', dir_to_data=None):
        if dir_to_data == None:
            dir_to_data = self.dir_to_save
        for var in ['procar_data' , 'orbital_names', 'ion_names', 'efermi', 'eigenvalues', 'weights']:
            self.check_existance(var, dir_to_data)

        Erange = [Erange_from_fermi[0] + self.efermi, Erange_from_fermi[1] + self.efermi]
        energies_number = int((Erange[1] - Erange[0]) / dE)
        dE_new = (Erange[1] - Erange[0]) / energies_number
        DOS_arr = np.zeros(energies_number)
        E_arr = np.arange(Erange_from_fermi[0], Erange_from_fermi[1], dE_new)
        nkpts = np.shape(self.procar_data)[0]
        nbands = np.shape(self.procar_data)[1]
        for k in range(nkpts):
            if self.print_out == True:
                print('kpoint = ', k)
            for b in range(nbands):
                energy = self.eigenvalues[k][b]
                if energy > Erange[0] and energy < Erange[1]:
                    e = int((energy - Erange[0]) / dE_new)
                    if ions == 'all':
                        list_of_ions = [i for i in range(len(self.ion_names))]
                    elif type(ions) == str:
                        list_of_ions = []
                        for i, name in enumerate(self.ion_names):
                            if name == ions:
                                list_of_ions.append(i)
                    elif type(ions) == list:
                        if type(ions[0]) == int:
                            list_of_ions = ions
                        elif type(ions[0]) == str:
                            list_of_ions = []
                            for ion_name in ions:
                                for i, name in enumerate(self.ion_names):
                                    if name == ion_name:
                                        list_of_ions.append(i)
                    if orbitals == 'all':
                        list_of_orbitals = [i for i in range(len(self.orbital_names))]
                    elif type(orbitals) == str:
                        list_of_orbitals = []
                        for i, name in enumerate(self.orbital_names):
                            if name == orbitals:
                                list_of_orbitals.append(i)
                    elif type(orbitals) == list:
                        if type(orbitals[0]) == int:
                            list_of_orbitals = ions
                        elif type(orbitals[0]) == str:
                            list_of_orbitals = []
                            for orb_name in orbitals:
                                for i, name in enumerate(self.orbital_names):
                                    if name == orb_name:
                                        list_of_orbitals.append(i)
                    weight = 0
                    for ion in list_of_ions:
                        for orb in list_of_orbitals:
                            weight+=self.procar_data[k][b][ion][orb]
                    DOS_arr[e] += weight / dE_new * self.weights[k]
        return E_arr, DOS_arr * 2, dE_new

if __name__=='__main__':
    t=time.time()
    """
    outcar_path = 'OUTCAR'
    wavecar_path = 'WAVECAR'
    dir_to_save = "Saved_data"
    p = Preprocessing(dir_to_save)
    p.process_OUTCAR(outcar_path=outcar_path, dir_to_save=dir_to_save)
    print('Processing OUTCAR takes: ', time.time()-t, 'sec')
    t=time.time()
    p.process_WAVECAR((-7.0, 4.0), 0.01, wavecar_path=wavecar_path, dir_to_save=dir_to_save)
    print('Job done! Processing WAVECAR: ', time.time()-t, ' sec')
    """
    dir_to_save = 'Saved_data'
    p = Preprocessing(dir_to_save)
    Erange = [-25.0, 5.0]
    dE = 0.1
    E_arr, DOS_arr, dE_new = p.get_DOS(Erange, dE)
    plt.plot(E_arr, gaussian_filter1d(DOS_arr, sigma=2), label='total')
    E_arr, PDOS_arr, dE_new = p.get_pdos(Erange, dE, ions='C', orbitals='s')
    plt.plot(E_arr, gaussian_filter1d(PDOS_arr, sigma=2), label='C-s')
    E_arr, PDOS_arr, dE_new = p.get_pdos(Erange, dE, ions='C', orbitals=['px', 'py', 'pz'])
    plt.plot(E_arr, gaussian_filter1d(PDOS_arr, sigma=2), label='C-p')
    E_arr, PDOS_arr, dE_new = p.get_pdos(Erange, dE, ions='H', orbitals='s')
    plt.plot(E_arr, gaussian_filter1d(PDOS_arr, sigma=2), label='H-s')
    E_arr, PDOS_arr, dE_new = p.get_pdos(Erange, dE, ions='O', orbitals='s')
    plt.plot(E_arr, gaussian_filter1d(PDOS_arr, sigma=2), label='O-s')
    E_arr, PDOS_arr, dE_new = p.get_pdos(Erange, dE, ions='O', orbitals=['px', 'py', 'pz'])
    plt.plot(E_arr, gaussian_filter1d(PDOS_arr, sigma=2), label='O-p')
    plt.xlabel('E, eV')
    plt.ylabel('DOS, states/eV/cell')
    plt.legend()
    plt.savefig('pdos.png', dpi=300)
    plt.show()
    plt.close()


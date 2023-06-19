# Configuration file for Semeter and Kamalabadi 2005 inversion code
# Written by Riley Troyer Fall 2021

# Write quantities to send to program in this dictionary
# isr_data_path is where the isr data files are stored
# isr_lat, isr_lon are geographic latitude and longitude of isr
# isr_min_alt is the minimum altitude in m to use isr data, below data is usually bad
# isr_sensitivity is the density value below I don't have confidence in the readings
# max_msis_alt is the altitude in meters to approximate infinity when calculating mass distance
# max_iterations defines the max number of iterations of the maximum entropy method
# convergence this is the change in chi2 the program aims to achieve
# test_flux is the number flux to use when constructing the A matrix
# altitude_bins is the altitude values in meters to calculate inversion for
# energy_bins is the energy values in electron-volts to calculate inversion for
# start_file_i, end_file_i index of start and end file to run analysis for

import numpy as np
import os

d = {

    'isr_data_dir' : '../LAMP/data/pfisr-data/barker-code/',
    'save_dir' : '../LAMP/data/pfisr-data/inversions/',
    'event_file' : '../LAMP/reference/lamp-pfisr-database.txt',
    'isr_lat' : 65.117,
    'isr_lon' : 212.540,
    'isr_min_alt' : 60000,
    'isr_sensitivity' : 1e9,
    'max_msis_alt' : 1001e3,
    'max_iterations' : 10000,
    'convergence': 1e-2,
    'test_flux' : 1e11,	
    'altitude_bins' : np.linspace(60e3, 144e3, 100),
    'energy_bins' : np.logspace(np.log10(1000), np.log10(500000), 25),
    'alpha_type' : 'stanford',
    'start_file_i' : None,
    'end_file_i' : None

}

run_info = {'config_info' : d,
            'run_files' : sorted([f for f in 
                    os.listdir(d['isr_data_dir']) 
                    if f.endswith('.h5')])[d['start_file_i']:d['end_file_i']]
}

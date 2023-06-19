"""
Script to perform an energy inversion as per Semeter, Kamalabadi 2005 for PFISR data for specified number of days

Written by Riley Troyer
science@rileytroyer.com
"""
####################### Initialize Program #######################
# Libraries
from datetime import datetime
import logging
import numpy as np
import os
import pickle
from scipy.interpolate import interp1d
import subprocess

# Disable divide by zero numpy warnings
np.seterr(divide='ignore')
np.seterr(invalid='ignore')

# Initiate logging
logging.basicConfig(filename = f'logs/run-semeter-inversion-{datetime.today().date()}.log',
                    encoding='utf-8',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

####################### End Initializing #######################


####################### Custom Functions to Import #######################
# Function to read in PFISR data
from src.data.pfisr_functions import get_isr_data

# Functions to convert electron density to ion production rate
from src.models.ne_to_q_functions import (get_msis_density, mass_distance,
                                          barrett_hays_range_energy_func,
                                          recombination_coeff, isr_ion_production_rate)

# Functions to perform the maximum entropy inversion
from src.models.mem_functions import (estimate_initial_number_flux,
                                      maximum_entropy_iteration)

# Functions to plot and save figures
from src.visualization.mem_plotting_functions import (save_inversion_density_plot,
                                                      save_inversion_numflux_plot)

####################### End of Custom Functions #######################


####################### Local Functions #######################

def find_event_indices(utc_time:np.ndarray) -> list:
    """Function to find only indices of times of interest.
    INPUT
    utc_time - utc datetimes of all pfisr data
    OUTPUT
    slices_n - indices of pfisr data that is of interest
    """
    
    # Find the date for the current pfisr file, this is a little tricky as
    #...some pfisr files span multiple days
    pfisr_dates = np.unique(np.array([d.date() for d in utc_time]))

    # Dates that are in both pa database and pfisr file
    pa_pfisr_dates = np.unique(np.array([d for d in pa_dates 
                                         if d in pfisr_dates]))

    # Loop through each of these dates and get correct indices
    indices = []
    for date in pa_pfisr_dates:
            indices.append(np.argwhere(pa_dates == date))

    # Flatten list of indices
    indices = [item[0] for sublist in indices for item in sublist]

    # Loop through each index and get data slices corresponding to the
    #...start and stop times
    slices_n = []
    for index in indices:

        # Get the date and start time of measurements
        date = pa_database[index, 0]
        start_time = date + ' ' + pa_database[index, 1]
        end_time = date + ' ' + pa_database[index, 2]

        # Convert to datetime
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

        # Find which indices in pfisr data correspond
        slices_n.append(np.argwhere((utc_time >= start_time) 
                                    & (utc_time <= end_time)))

    # Flatten pfisr array indices
    slices_n = [item[0] for sublist in slices_n for item in sublist]
    
    return slices_n

####################### End of Local Functions #######################


####################### Read in Configuration #######################

# Read in config file with dictionary of specified inputs
import src.config.config_2023_05_26 as config

config_data = config.run_info['config_info']

# Path to pfisr data directory
pfisr_data_dir = config_data['isr_data_dir']

# File with times for events of interest
reference_file = config_data['event_file']

# Directory to save files to
save_dir = config_data['save_dir']

# Get location of PFISR
pfrr_lat = config_data['isr_lat']
pfrr_lon = config_data['isr_lon']

# Define test flux in m^-2 s^-1
F = config_data['test_flux']

# Don't use PFISR data below this altitude in km
pfisr_min_alt = config_data['isr_min_alt']

# Get sensitivity limit of PFISR
pfisr_sensitivity = config_data['isr_sensitivity']

# Altitude in meters to approximate infinity when calculating
#...mass distance
max_msis_alt = config_data['max_msis_alt']

# Maximum number of iterations to run maximum entropy process on
max_iterations = config_data['max_iterations']

# Reduced chi square to aim for
convergence = config_data['convergence']

# Define arrays for altitude and energy bins

# Altitude in meters
#...number of points should be around the same as pfisr data
altitude_bins = config_data['altitude_bins']

# Energies in eV
#...should probably be less than altitude bins to avoid overfitting
energy_bins = config_data['energy_bins']

# Get which chemistry model to use
alpha_type = config_data['alpha_type']

# Get files to run code for
pfisr_files = config.run_info['run_files']
pfisr_files = pfisr_files[0:1]

####################### End of Config #######################


####################### START OF PROGRAM #######################

# Read in file with energy dissipation function
lambda_filename = 'models/semeter_kamalabadi_lambda_function.txt'
lambda_data = np.loadtxt(lambda_filename, skiprows=5)

# Create an interpolated function from this
#...values outside set to 0
lambda_interp = interp1d(lambda_data[:, 0], lambda_data[:, 1],
                         bounds_error=False, fill_value=0)

# Read in file with pulsating aurora dates, times and types
pa_database = np.loadtxt(reference_file, delimiter='\t', dtype=str)
pa_database = pa_database[1:, :]

# Convert dates to datetimes
pa_dates = np.array([datetime.strptime(d, '%Y-%m-%d').date() for d 
                     in pa_database[:, 0]])

for alpha_type in ['stanford','vickrey','osepian','gledhill']:
    
    logging.info(f'Starting: {alpha_type}')
    
    for pfisr_filename in pfisr_files:

        logging.info(f'Starting {pfisr_filename} for {alpha_type}.')

        # Read in the pfisr data
        (utc_time, unix_time, 
         pfisr_altitude,
         e_density, de_density) = get_isr_data(pfisr_filename, pfisr_data_dir)

        # Find indices of interest
        slices_n = find_event_indices(utc_time)

        # Create a dictionary to store inversion results in
        inversion_results = {}

        # Make a directory to store plots and dictionary if it doesn't 
        #...already exist
        output_dir = (save_dir + alpha_type + '/' 
                      + str(utc_time[0].date()) + '/')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logging.info(str(utc_time[0].date()))

        for slice_n in slices_n:

            run_time = utc_time[slice_n]

            # Get MSIS calculated densities
            try:
                (total_msis_alt,
                 msis_interp_density) = get_msis_density(run_time, altitude_bins,
                                                         max_alt=max_msis_alt,
                                                         glat=pfrr_lat, glon=pfrr_lon)
            except Exception as e:
                logging.warning(f'Issue with MSIS model. Continuing with exception {e}.')
                continue

            # Get density for altitude bins
            total_msis_density = msis_interp_density(total_msis_alt)
            density_rho = msis_interp_density(altitude_bins)


            # Calculate mass distance (s) for each altitude 
            #...by integrating out to 1000 km (~infinity)
            s = np.array([mass_distance(z, total_msis_density, total_msis_alt) for z 
                          in range(len(altitude_bins))])


            # Calculate ion production rate for each energy and store
            #...in dictionary
            ion_prod_rate = {}

            for i, energy in enumerate(energy_bins):

                # Calculate range-energy value
                R = barrett_hays_range_energy_func(energy)

                # Get the (s/R)(z) for the energy
                s_R = s/R

                # Use s/R to get Lambda function values
                lambda_vals = lambda_interp(s_R)

                # Use all of this to calculate ion production rate 
                #...as function of alt
                q = (lambda_vals * density_rho * energy * F) / (35.5 * R)

                # Write to dictionary
                ion_prod_rate[energy] = q

            # Construct the A matrix
            matrix_A = np.zeros([len(altitude_bins),
                                 len(energy_bins)])

            # Loop through each energy value
            for j in range(len(energy_bins)):

                # Get the size of the energy bin
                #...first bin is from zero to energy
                if j == 0:
                    delta_E = energy_bins[j] - 0
                else:
                    delta_E = energy_bins[j] - energy_bins[j-1]

                # Set column of matrix
                matrix_A[:, j] = ion_prod_rate[energy_bins[j]] * (delta_E/F)

            # Get estimated ion production rate and error 
            #...from isr measurements
            try:
                (q_estimate, 
                 dq_estimate,
                 alphas) = isr_ion_production_rate(e_density, de_density,
                                                   pfisr_altitude, altitude_bins,
                                                   slice_n, unix_time, 
                                                   pfrr_lat, pfrr_lon,
                                                   alpha_type=alpha_type,
                                                   base_dir='./')
            except Exception as e:
                logging.warning(f'Issue with ion production rate calculation. '
                                f'Continuing with expection {e}.')
                continue

            # Make an initial guess of the number flux
            initial_num_flux = estimate_initial_number_flux(energy_bins,
                                                            matrix_A)
            try:
                # Perform the maximum entropy iterative process
                (new_num_flux,
                 chi_square,
                 dof,
                 converged) = maximum_entropy_iteration(initial_num_flux, altitude_bins,
                                                        energy_bins, matrix_A,
                                                        q_estimate, dq_estimate,
                                                        max_iterations=max_iterations, 
                                                        convergence=convergence)
            except Exception as e:
                logging.warning(f'Issue with MEM. Continuing with exception {e}.')
                continue

            # Write data to dictionary
            d = {'altitude' : altitude_bins,
                 'initial_density' : np.sqrt(np.dot(matrix_A,
                                                initial_num_flux)/alphas),
                 'modeled_density' : np.sqrt(np.dot(matrix_A,
                                                new_num_flux)/alphas),
                 'measured_density' : np.sqrt(q_estimate/alphas),
                 'measured_error' : np.sqrt(abs(dq_estimate/alphas)),
                 'energy_bins' : energy_bins,
                 'modeled_flux' : new_num_flux,
                 'chi2' : chi_square,
                 'dof' : dof,
                 'converged' : converged,
                 'units' : 'Values given in meters, seconds, electron-volts.'
                }

            inversion_results[run_time] = d

            # Plot the results and save to output directory
            if slice_n%1 == 0:
                save_inversion_density_plot(inversion_results,
                                            run_time, output_dir)
                save_inversion_numflux_plot(inversion_results,
                                            run_time, output_dir)

        # Write the dictionary with inversion data to a pickle file
        with open(output_dir + 'inversion-data-' + str(utc_time[0].date()) 
                  + '.pickle', 'wb') as handle:
            pickle.dump(inversion_results, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        
        # Clear temporary files in /dev/shm directory in Linux
        try:
            os.system('rm /dev/shm/*')
        except Exception as e: 
            logging.info(f'Tried to remove shared memory files {e}.')

        logging.info(f'Finished with {pfisr_filename} for {alpha_type}.')

    logging.info(f'Finished with all files for {alpha_type}')

logging.info(f'Finished with all files for all alpha types. '
             f'Results and plots are saved under {output_dir}.')

####################### END OF PROGRAM #######################




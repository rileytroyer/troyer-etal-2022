"""
Script to compile AE, AL, substorm, pa times, and peak/lower boundary electron density for PFISR pulsating aurora data

@author Riley Troyer
science@rileytroyer.com
"""

####################### Initialize Program #######################
# Libraries
from datetime import datetime
import logging
import numpy as np
import os
import pickle



# Initiate logging
logging.basicConfig(filename = f'logs/combine-data-and-integrate-energy-{datetime.today().date()}.log',
                    encoding='utf-8',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

####################### End Initializing #######################


####################### Local Functions #######################
def get_energy_integrations(energy_bins:np.array,
                            energy_flux:np.array, threshold:float) -> 'float, float':
    """Function to integrate energy flux to get values for statistics.
    INPUT
    energy_bins - energy bins in eV
    energy_flux - energy flux in eV m^-2 s^-1
    threshold - energy threshold in eV to seperate low and high energies
    OUTPUT
    high_total_energy - total energy above threshold in eV
    low_total_energy - total energy below threshold in eV
    """

    # Get closest bin to specified energy value
    threshold_i = np.argmin(abs(energy_bins - threshold))
    
    # Get low and high energy contributions
    high_total_energy = np.sum(energy_flux[threshold_i:])
    low_total_energy = np.sum(energy_flux[:threshold_i])
    
    return high_total_energy, low_total_energy

####################### End of Local Functions #######################


####################### START OF PROGRAM #######################

chemistry_type = 'stanford'

# Get a list of directories for inversion data
inversion_data_dir = (f'data/interim/semeter-troyer-inversions/{chemistry_type}/')
inversion_dirs = os.listdir(inversion_data_dir)
inversion_dirs = [d for d in inversion_dirs if not d.startswith('.')]
inversion_dirs = [inversion_data_dir + d + '/' for d in inversion_dirs]

inversion_dirs = sorted(inversion_dirs)

# Loop through each directory and read inversion results
#...into single file
all_inversion_results = {}

for inversion_dir in inversion_dirs:
    
    # Look for pickle file in directory
    try:
        inversion_results_filename = [f for f in os.listdir(inversion_dir)
                                      if f.endswith('.pickle')][0]
    
        # Read in pickle file
        with open(inversion_dir 
                  + inversion_results_filename, 'rb') as handle:
            inversion_results = pickle.load(handle)
        
        # Set keys of individual day to full dictionary
        for key in inversion_results.keys():
            all_inversion_results[key] = inversion_results[key]
            
    except Exception as e: 
        logging.info(f'Failed to read in inversion data for {inversion_dir} with exception {e}.')

logging.info(f'Read in inversion data from: data/interim/semeter-troyer-inversions/{chemistry_type}/')


# Read in file with substorm delay and MLT for PFISR data
with open('data/interim/statistics/pa-pfisr-data-dict-latest.pickle', 'rb') as handle:
    pa_statistics_data = pickle.load(handle)

logging.info(f'Read in file: data/interim/statistics/pa-pfisr-data-dict-latest.pickle')

# Create a new dictionary to store only PA results
pa_inversion_results = {}

# Make array to store all inversion results times
inversion_times = np.array(list(all_inversion_results.keys()))

# Loop through each time and add MLT, substorm delay, and energy calcs
for n, key in enumerate(pa_statistics_data.keys()):
    
    # Find the closest time in full inversion results
    delay = abs(key - inversion_times)
    
    # If there isn't a close time skip
    if np.min(delay).total_seconds() > 60:
        continue

    closest_index = np.argmin(delay)
    closest_time = inversion_times[closest_index]
    
    # Set PA dictionary to inversion results
    pa_inversion_results[closest_time]=all_inversion_results[closest_time]
    
    # Add on statistics data
    d = pa_statistics_data[key]
    pa_inversion_results[closest_time]['low_altitude'] = d['altitude']
    pa_inversion_results[closest_time]['substorm_delay'] = d['delay']
    pa_inversion_results[closest_time]['substorm'] = d['substorm']
    pa_inversion_results[closest_time]['pa_event'] = d['pa event']
    pa_inversion_results[closest_time]['local_time'] = d['local_time']
    pa_inversion_results[closest_time]['mlt_time'] = d['mlt_time']
    pa_inversion_results[closest_time]['ae_index'] = d['ae_index']
    pa_inversion_results[closest_time]['al_index'] = d['al_index']
    
    # Add energy data calculations
    
    # Calculate energy flux
    modeled_flux = pa_inversion_results[closest_time]['modeled_flux']
    
    # To get differential number flux need to
    #...multiply by energy bin widths
    energy_bins = pa_inversion_results[closest_time]['energy_bins']
    bin_widths = energy_bins - np.roll(energy_bins, shift=1)
    #...fix first value
    bin_widths[0] = energy_bins[0] - 0
    
    # To get differential energy flux multiply diff num flux
    #...by energy bins
    energy_flux = modeled_flux * bin_widths * energy_bins
    
    # Get integrated energy fluxes for 30, 50, and 100 keV
    #...integration removes dependences of diff energy flux on eV^-1
    #...note, flux is for entire energy bin, so just have to sum
    #...each bin value
    #...Also make sure to exclude last energy bin
    #...inversion process can cause this to be large
    total_energy = np.sum(energy_flux[0:-1])
    
    (high_energy_10, 
     low_energy_10) = get_energy_integrations(energy_bins[0:-1],
                                              energy_flux[0:-1],
                                              10000)
    
    (high_energy_30, 
     low_energy_30) = get_energy_integrations(energy_bins[0:-1],
                                              energy_flux[0:-1],
                                              30000)
    (high_energy_50, 
     low_energy_50) = get_energy_integrations(energy_bins[0:-1],
                                              energy_flux[0:-1],
                                              50000)
    (high_energy_100, 
     low_energy_100) = get_energy_integrations(energy_bins[0:-1],
                                               energy_flux[0:-1],
                                               100000)
    
    # Write these to dictionaries as well
    pa_inversion_results[closest_time]['total_energy'] = total_energy
    pa_inversion_results[closest_time]['high_10keV'] = high_energy_10
    pa_inversion_results[closest_time]['low_10keV'] = low_energy_10
    pa_inversion_results[closest_time]['high_30keV'] = high_energy_30
    pa_inversion_results[closest_time]['low_30keV'] = low_energy_30
    pa_inversion_results[closest_time]['high_50keV'] = high_energy_50
    pa_inversion_results[closest_time]['low_50keV'] = low_energy_50
    pa_inversion_results[closest_time]['high_100keV'] = high_energy_100
    pa_inversion_results[closest_time]['low_100keV'] = low_energy_100

    if n%100 == 0:
        logging.info(f'Finished integrating {n} energy spectra of {len(pa_statistics_data.keys())}.')

logging.info(f'Finished all integrations.')

# Save dictionary to file
with open(f'data/interim/statistics/pa_inversion_results_{chemistry_type}.pickle', 'wb') as handle:
    pickle.dump(pa_inversion_results, handle, 
                protocol=pickle.HIGHEST_PROTOCOL)
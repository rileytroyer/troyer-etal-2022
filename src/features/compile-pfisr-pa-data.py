"""
Script to compile AE, AL, substorm, pa times, and peak/lower boundary electron density for PFISR pulsating aurora data

@author Riley Troyer
science@rileytroyer.com
"""

####################### Initialize Program #######################
# Libraries
from datetime import datetime
from datetime import time as dt_time
from datetime import timedelta
import h5py
import logging
import numpy as np
import os
from pathlib import Path
import pickle
import pytz
from scipy.signal import find_peaks
import sys



# Initiate logging
logging.basicConfig(filename = f'logs/compile-pfisr-pa-data-{datetime.today().date()}.log',
                    encoding='utf-8',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

####################### End Initializing #######################


####################### Custom Functions to Import #######################

# Add root to path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

# Function to read in PFISR data
from src.data.pfisr_functions import get_isr_data

####################### End of Custom Functions #######################


####################### START OF PROGRAM #######################

# Find time of day and delay from substorm start for each slice of
#...pfisr data along with mlt and AE index.

# Read in AE index files
ae_files = sorted(os.listdir('data/raw/indices/ae/'))

# Save indices in dictionary
ae_indices = {}

# Loop through each file and write to array
for ae_file in ae_files:
    
    # Read in file
    ae_data = np.loadtxt('data/raw/indices/ae/' + ae_file,
                         dtype='str', skiprows=1)

    # Parse datetime from file
    dates = np.array([datetime.strptime(d, '%Y/%j-%H:%M:%S') 
                      for d in ae_data[:, 0]])

    # Loop through each date and write index to dictionary
    for n, date in enumerate(dates):
        ae_indices[date] = float(ae_data[n, 1])

logging.info('AE files read in.')
        
# Read in AL index files
al_files = sorted(os.listdir('data/raw/indices/al/'))
al_files = sorted([f for f in al_files if not f.startswith('.')])

# Save indices in dictionary
al_indices = {}

# Loop through each file and write to array
for al_file in al_files:
    
    # Read in file
    al_data = np.loadtxt('data/raw/indices/al/' + al_file,
                         dtype='str', skiprows=1)

    # Parse datetime from file
    dates = np.array([datetime.strptime(d, '%Y/%j-%H:%M:%S') 
                      for d in al_data[:, 0]])

    # Loop through each date and write index to dictionary
    for n, date in enumerate(dates):
        al_indices[date] = float(al_data[n, 1])

logging.info('AL files read in.')

# Use a geomagnetic substorm list

#There are a few different options, specify which to use
filenames = ['substorm-list-forsyth.txt',
             'substorm-list-newell.txt',
             'substorm-list-ohtani.txt']
#filenames = ['substorm-list-forsyth.txt']

# Define location of PFISR in GLON, GLAT
pfisr_glon = 360 - 147.47
pfisr_glat = 65.13

# Read in the files with substorm lists and combine
mag_substorm = np.empty((0, 5), float)
for filename in filenames:
    single_mag_substorm = np.loadtxt('data/raw/substorm-lists/' 
                                     + filename,
                                     delimiter=',', dtype=str)
    mag_label = single_mag_substorm[0, :]
    single_mag_substorm = single_mag_substorm[1:, :]
    
    # Append to array
    mag_substorm = np.append(mag_substorm, single_mag_substorm, axis=0)

# Filter substorms to just near pfisr location
glon_filt_low = mag_substorm[:, 3].astype(float) > (pfisr_glon - 15)
glon_filt_high = mag_substorm[:, 3].astype(float) < (pfisr_glon + 15)
glat_filt_low = mag_substorm[:, 4].astype(float) > (pfisr_glat - 8)
glat_filt_high = mag_substorm[:, 4].astype(float) < (pfisr_glat + 8)

mag_substorm = mag_substorm[glon_filt_low & glon_filt_high
                            & glat_filt_low & glat_filt_high]

# Pull out just dates
substorm_start = mag_substorm[:, 0]

# Convert to datetime
substorm_start = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
                  for d in substorm_start]

logging.info('Substorm files read in.')

# Read in file with pulsating aurora dates, times and types
pa_database = np.loadtxt('data/processed/'
                         + 'pa-pfisr-database.txt',
                         delimiter='\t', dtype=str)
pa_database = pa_database[1:, :]

# Filter pulsating aurora database to just the good data
pa_database_1 = pa_database[pa_database[:, 3] == 'good']
pa_database_2 = pa_database[pa_database[:, 3] == 'issue']

# Read in file with discrete aurora dates and times
discrete_database = np.loadtxt('data/processed/'
                               + 'discrete-pfisr-database.txt',
                               delimiter='\t', dtype=str)
discrete_database = discrete_database[1:, :]


# Loop through each pa date and find time from nearest substorm
pa_pfisr_dict = {}

# # Make database by combining good and issue events
# database = np.concatenate((pa_database_1, pa_database_2),
#                           axis=0)
database = pa_database_1

logging.info('Aurora classifications read in.')

auroral_type = 'pa'
low_alt_cutoff = 60
cutoff = 12

# Define alaska timezone
alaska_tzinfo = pytz.timezone("US/Alaska")
mlt_midnight = dt_time(10, 54, 0)

logging.info('Starting to process all events...')

for m in range(0, len(database)):
    
    # Get the date and start time of measurements
    date = database[m, 0]
    start_time = date + ' ' + database[m, 1]
    end_time = date + ' ' + database[m, 2]
    
    # Convert to datetime
    start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    date = start_time.date()
    
    # Read in the PFISR data file
    date_str = (str(date.year).zfill(4) 
                + str(date.month).zfill(2)
                + str(date.day).zfill(2))
    pfisr_files = os.listdir('data/raw/pfisr/')
    
    # Take only h5 files
    pfisr_files = [f for f in pfisr_files if f.endswith('.h5')]
    
    # In case there are multiple files just take first one
    pfisr_filename = [f for f in pfisr_files if date_str in f][0]

    (pfisr_time,
     unix_time,
     pfisr_alt,
     e_density,
     de_density) = get_isr_data(pfisr_filename, 'data/raw/pfisr/')
    
    # Select data over specified altitude range
    low_alt_cutoff = 60000

    # Select only between 60km and 140km
    altitude_slice = (pfisr_alt 
                      < 140000) & (pfisr_alt 
                                   > low_alt_cutoff)
    e_density = e_density[altitude_slice, :]
    de_density = de_density[altitude_slice, :]
    pfisr_alt = pfisr_alt[altitude_slice]
    
    # Filter to only between pulsating aurora times
    e_density = e_density[:, (pfisr_time >= start_time)
                          & (pfisr_time <= end_time)]
    de_density = de_density[:, (pfisr_time >= start_time)
                            & (pfisr_time <= end_time)]
    pfisr_time = pfisr_time[(pfisr_time >= start_time)
                            & (pfisr_time <= end_time)]

    # Loop through each time and select lowest altitude and peak height
    for n in range(0, len(pfisr_time)):
        
        # Select profile for time
        density_profile = e_density[:, n]
        
        # Get location of max peak
        peaks, properties = find_peaks(density_profile,
                                       height=1e10, prominence=(None, 5e9))
        if len(peaks) == 0:
            peak_alt = np.nan
        else:
            peak_alt = pfisr_alt[peaks[np.argmax(properties['peak_heights'])]]
        
        # Also get the profile for the error in this
        error_profile = de_density[:, n]
        
        # And make sure all of the dates are the same
        profile_time = pfisr_time[n]
        
        # Get local time and magnetic local time
        ak_time = profile_time.astimezone(alaska_tzinfo)
        ak_time = datetime.combine(ak_time.date(), ak_time.time()) 
        mlt_delay = profile_time - datetime.combine(profile_time.date(),
                                              mlt_midnight)
        mlt_time = datetime.combine(profile_time.date(), 
                              dt_time(0, 0, 0)) + mlt_delay
        
        # Select only densities > 10^10
        density_slice = density_profile >= 1e10
        
        # Select only data points with reasonable error
        error_slice = error_profile < 5e9
        
        # Select only only data points with continuity with previous
        # Select only densities that connect to points above
        continuity_slice = np.zeros(len(pfisr_alt), dtype=bool)
        
        for j in np.arange(len(pfisr_alt)):

            if j < (len(pfisr_alt) - 20):

                if ((np.min(density_profile[j+1:j+20]) > 0)
                    & (np.median(density_profile[j-2:j+2]) 
                       < np.median(density_profile[j+1:j+20]))):
                    continuity_slice[j] = True

                else:
                    continuity_slice[j] = False

            else: 
                continuity_slice[j] = False
        
        density_profile = density_profile[density_slice 
                                          & error_slice
                                          & continuity_slice]
        
        # Find the closest substorm start, must start before pa
        delays = [pfisr_time[n] - t for t in substorm_start]
        
        # Create an array to store substorm times and delays
        substorm_info = np.array([substorm_start, delays])
        
        # Set any negative values to 24hr delay
        neg_filter = substorm_info[1, :] < timedelta(seconds=0)
        substorm_info[1, neg_filter] = timedelta(seconds = 3600*cutoff)
        
        # Also any above 24hr to 24hr
        day_filter = substorm_info[1, :] > timedelta(seconds = 
                                                     3600*cutoff)
        substorm_info[1, day_filter] = timedelta(seconds = 3600*cutoff)
        
        # Get the nearest substorm array
        delay_index = np.argmin(substorm_info[1, :])
        substorm = substorm_info[0, delay_index]
        delay = (substorm_info[1, delay_index].total_seconds())/60
        
        # Find nearest AE index value
        ae_i = np.argmin([abs((d - profile_time).total_seconds())
                          for d in ae_indices.keys()])
        ae_index = ae_indices[list(ae_indices.keys())[ae_i]]
        
        # Find the nearest AL index
        al_index = al_indices[list(ae_indices.keys())[ae_i]]
        
        # Account for possibly no density above threshold
        try:
            # Get lowest altitude value
            lowest_alt = pfisr_alt[density_slice
                                   & error_slice
                                   & continuity_slice][0]
        
        except:
            lowest_alt = np.nan

        # Write to dictionary
        pa_pfisr_dict[profile_time] = {'altitude' : lowest_alt,
                                       'peak_alt' : peak_alt,
                                      'delay' : delay,
                                      'substorm' : substorm,
                                      'pa event' : m,
                                      'local_time' : ak_time,
                                      'mlt_time' : mlt_time,
                                      'ae_index' : ae_index,
                                      'al_index' : al_index}
            
    logging.info(f'PA Event {m} finished.')


# Save dictionary to file
with open('data/interim/statistics/pa-pfisr-data-dict-latest.pickle',
                  'wb') as handle:
    pickle.dump(pa_pfisr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

logging.info(f'Compiled data written to pickle file at: data/interim/statistics/pa-pfisr-data-dict-latest.pickle')

####################### END OF PROGRAM #######################
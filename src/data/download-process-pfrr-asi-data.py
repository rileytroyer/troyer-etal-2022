"""
Script to download and process Poker Flat ASI data.
Processed data is stored in an hdf5 file (.h5) with all the images for an entire day.
Processed images are downscaled from 16-bit to 8-bit using a custom scaling.
Custom scaling uses contrast limited adaptive histogram equalization

Written by Riley Troyer
science@rileytroyer.com
"""
# Import needed libraries
from datetime import datetime
import pandas as pd
import logging

# Import functions to do downloading and processing
from pfrr_asi_data_functions import download_pfrr_images, pfrr_asi_to_hdf5_8bit_clahe

# Important directories
data_dir = 'data/'
logs_dir = 'logs/'

# Initiate logging
logging.basicConfig(filename=logs_dir + f'download-process-pfrr-asi-{datetime.today().date()}.log',
                    encoding='utf-8',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

#------------------------------Initializing done------------------------------

# Load in the file with days to download and process data for
days_list_file = data_dir + 'processed/pa-pfisr-database-table.xlsx'
days_list = pd.read_excel(days_list_file).loc[:, 'Date']

# Convert to datetimes
days_list = [d.to_pydatetime().date() for d in days_list]

# Loop through each day, download, and create .h5 file
logging.info('Starting download and processing for all days.')

for day in days_list[0:1]:
    
    try: 
        # Download the images to the raw directory
        # You can use significantly more processes than cpu cores to speed this up
        download_pfrr_images(day, save_dir=data_dir+'raw/pfrr-asi/',
                            wavelength='558', processes=25)
    except Exception as e:
        logging.critical(f'Unable to download files for {day}. Stopped with error {e}')
    
    try:
        # Create the hdf5 file
        # Using more processes than cpu cores will slow this down
        pfrr_asi_to_hdf5_8bit_clahe(day, save_base_dir=data_dir+'interim/pfrr-asi-h5/',
                                    img_base_dir=data_dir+'raw/pfrr-asi/',
                                    wavelength='558', del_files=False, processes=4)
    except Exception as e:
        logging.critical(f'Unable to create h5 file for {day}. Stopped with error {e}')

logging.info('Finished downloading and processing all days.')
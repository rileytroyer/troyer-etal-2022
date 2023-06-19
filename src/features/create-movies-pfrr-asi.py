"""
Script to create movies from the Poker Flat ASI data.
Movies get created from hdf5 files created using download-process-pfrr-asi-data.py script

Written by Riley Troyer
science@rileytroyer.com
"""
# Import needed libraries
from datetime import datetime
import pandas as pd
import logging

# Import functions to do downloading and processing
from pfrr_asi_visual_functions import create_timestamped_movie

# Important directories
project_dir = '/data/projects/new-project-structure/pfisr-energy/'
data_dir = project_dir + 'data/'
logs_dir = project_dir + 'logs/'

# Initiate logging
logging.basicConfig(filename=logs_dir + f'create-movie-pfrr-asi-{datetime.today().date()}.log',
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
logging.info('Starting to create movies for all days.')

for day in days_list[0:2]:
    
    try: 
        # Download the images to the raw directory
        create_timestamped_movie(day, img_base_dir=data_dir+'interim/pfrr-asi-h5/', 
                                 save_base_dir=data_dir+'processed/pfrr-asi-movies/')
    except Exception as e:
        logging.critical(f'Unable to create movie for {day.date()}. Stopped with error {e}')

logging.info('Finished creating movies for all days.')
# Read Me File for PFISR Pulsating Aurora Energy Project
see document-struture.md for a layout of the project structure and where files are located

## Setup
1. When running code make sure you are in the base directory /pfisr-energy as this will ensure that all the code runs as expected.
2. I recommend creating a new virtual environment and installing all the dependencies through pip3 with the requirements.txt file.
3. The most challenging install process has to do with the msise00 and iri2016 libraries. There are a few caveats with these that you should review before installing: docs/iri2016-msise00-install.md

## Data
If you are running this for the first time you will want to download the data

### Poker Flat All Sky Camera
1. Run the code in /src/data/download-process-pfrr-asi-data.py
    - This will download the raw images from the UAF Geophysical institute FTP server
    - Process the images by increasing contrast and converting to 8-bit
    - Store the processed images in an hdf5 (.h5) file.
2. Run the code in /src/data/features/create-movies-pfrr-asi.py
    - This will create a timestamped movie using the .h5 files.

### Poker Flat Incoherent Scatter Radar Data (PFISR)
This data is located in the AMISR repository: https://data.amisr.com/database/pfisr/level2/nenotr/
Unfortunately I don't yet have a function that automatically downloads it, but that is on the horizon.

## Inversion Process
The inversion process takes the PFISR electron density and extracts an estimated differential energy flux. The code features/run-semeter-inversion.py will do this. For an explanation of what each part of the code is doing see notebooks/reports/semeter-inversion-example.ipynb.

To run the code you should create a config file, some examples are in src/config/config_*.py. In this file you will specify where the PFISR data is stored, model parameters, and where to store the output figures and model results. You likely won't need to change much if anything in these config files.

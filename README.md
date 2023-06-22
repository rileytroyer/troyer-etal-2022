# Read Me File for PFISR Pulsating Aurora Energy Project
see document-struture.md for a layout of the project structure and where files are located
you will need to create reports/figures/, logs/, and data/ directories as these are included in my .gitignore file.

## Setup
1. When running code make sure you are in the base directory /pfisr-energy as this will ensure that all the code runs as expected.
2. I recommend creating a new virtual environment and installing all the dependencies through pip3 with the requirements.txt file.
3. The most challenging install process has to do with the msise00 and iri2016 libraries. There are a few caveats with these that you should review before installing: docs/iri2016-msise00-install.md

## 1. Data
If you are running this for the first time you will want to download the data

### Poker Flat All Sky Camera
1. Run the code in /src/data/download-process-pfrr-asi-data.py
    - This will download the raw images from the UAF Geophysical institute FTP server
    - Process the images by increasing contrast and converting to 8-bit
    - Store the processed images in an hdf5 (.h5) file.
2. Run the code in /src/data/features/create-movies-pfrr-asi.py
    - This will create a timestamped movie using the .h5 files.
    - Requires FFMPEG to be installed `sudo apt install ffmpeg`

### Poker Flat Incoherent Scatter Radar Data (PFISR)
This data is located in the AMISR repository: https://data.amisr.com/database/pfisr/level2/nenotr/
Unfortunately I don't yet have a function that automatically downloads it, but that is on the horizon.

## 2. Inversion Process
The inversion process takes the PFISR electron density and extracts an estimated differential energy flux. The code features/run-semeter-inversion.py will do this. For an explanation of what each part of the code is doing see notebooks/reports/semeter-inversion-example.ipynb.

To run the code you should create a config file, some examples are in src/config/config_*.py. In this file you will specify where the PFISR data is stored, model parameters, and where to store the output figures and model results. You likely won't need to change much if anything in these config files.

## 3. Compiling Data to Analyze
After the inversions we need to compile additional data to do our analysis. This includes the following
- AE and AL indices downloaded for each day from: https://lasp.colorado.edu/space_weather/dsttemerin/archive/dst_years.html
- Substorm lists (Newell and Gjerloev (2011), Forsyth et al. (2015), and Ohtani and Gjerloev (2020)) from: https://supermag.jhuapl.edu/substorms/?tab=download
- List of pulsating aurora times derived via human classification
- Lower altitude boundary, and peak altitude of PFISR electron density

To compile this data run the script located at src/features/compile-pfisr-pa-data.py

The results of this code will be a .pickle file with all of the data that is stored in data/interim/statistics/

### 3.1 Combine data and integrate energies
After compiling the additional data the next step is to combine this with the inversion results and integrate the inversion results to produce a low and high energy flux for several different set points (10, 30, 50, and 100 keV).

To do this run the script located at src/features/combine-data-and-integrate-energy.py

The results of this will be a .pickle file stored at data/interim/statistics/

## 4. Visualize the results
With all of the data processed we can start to analyze and visualize it. I do most of this in jupyter notebooks located in notebooks/reports/

To reproduce the figures in the paper you will want to run the following notebooks
- notebooks/reports/plotting-raw-inversion-results-substorm.ipynb
- notebooks/reports/plotting-raw-inversion-results-ae.ipynb

Make note of the high and low energy values for each bin in each notebook. I then enter these numbers into data/processed/chemistry-model-results.xlsx

Running the notebook notebooks/reports/inversion-results/paper-plots.ipynb will create the energy inversion figures in the paper.

Running the notebook notebooks/reports/plot-scatter-plots-for-paper.ipynb will create the scatter plots in the paper.

Running the notebook notebooks/reports/plot-high-energy-timescale-paper.ipynb will create the timescale plot from the paper.

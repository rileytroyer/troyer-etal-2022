""" 
Functions to process and read in pfisr D-region data

Written by Riley Troyer
science@rileytroyer.com
"""

import numpy as np
import datetime
import h5py

def get_isr_data(pfisr_filename:str, pfisr_data_dir:str) -> 'np.array, np.array, np.array, np.array, np.array':
    """Function to get relevant data from PFISR datafile.
    Reads in 90 degree (vertical) beam by default.
    INPUT
    pfisr_filename - data file name, should be .h5 file
    pfisr_data_dir - directory where isr data is stored
    OUTPUT
    utc_time - time stamp for the start of each measurement
    unix_time - unix time
    pfisr_altitude - altitude stamp for each measurement in meters
    e_density - electron number density in m^-3
    de_density - error in number density
    """
    
    # Read in the h5 file
    pfisr_file = h5py.File(pfisr_data_dir + pfisr_filename, 'r')

    # Get the different beams and select specified angle
    beam_angle = 90
    beams = np.array(pfisr_file['BeamCodes'])

    # Get the beam with a 90 degree elevation angle
    indexes = np.linspace(0, len(beams)-1, len(beams))
    beam_num = int(indexes[np.abs(beams[:,2] - beam_angle) == 0][0])

    # Get time and convert to utc datetime
    unix_time = np.array(pfisr_file['Time']['UnixTime'])[:,0]
    utc_time = np.array([datetime.datetime.utcfromtimestamp(d) 
                         for d in unix_time])

    # Get the altitude array
    pfisr_altitude = np.array(pfisr_file['NeFromPower']
                              ['Altitude'])[beam_num, :]

    # Get the uncorrected number density array
    e_density = np.array(pfisr_file['NeFromPower']
                         ['Ne_NoTr'])[:, beam_num, :]

    # Take the transpose
    e_density = np.transpose(e_density)
    
    # Find the noise floor by averaging between 55km and 60km
    #...assume this should be zero
    
    # Calculate the power given that power = density/range^2
    pfisr_range = np.array(pfisr_file['NeFromPower']
                           ['Range'])[0, :]

    # Turn 1D array into 2D array for elementwise division
    pfisr_range = np.array([pfisr_range,]*e_density.shape[1])
    pfisr_range = np.transpose(pfisr_range)
    pfisr_power = np.divide(e_density, pfisr_range**2)

    # Get the power bias
    noise_floor = np.nanmean(pfisr_power[(pfisr_altitude > 55000)
                                    & (pfisr_altitude < 60000), :],
                              axis=0)
    
    # Only apply correction to lower altitudes
    low_fade = 85e3
    high_fade = 110e3
    correction_fade = np.ones(len(pfisr_altitude))

    # Fade from 1 to 0 over 85 to 110km
    fade_selector = (pfisr_altitude > low_fade) & (pfisr_altitude < high_fade)
    fade_len = len(fade_selector[fade_selector == True])
    fade = np.linspace(1, 0, fade_len)

    # Set correct fade values
    correction_fade[fade_selector] = fade
    correction_fade[pfisr_altitude > high_fade] = 0
    

    # Loop through each column and subtract off noise floor
    for j in range(pfisr_power.shape[1]):
        pfisr_power[:, j] = pfisr_power[:, j] - noise_floor[j]*correction_fade   

    # Calculate new unbiased density
    e_density = np.multiply(pfisr_power, pfisr_range**2)
        
    
    # Get error values
    try:
        de_density = np.array(pfisr_file['NeFromPower']
                              ['errNe_NoTr'])[:, beam_num, :]
        de_density = np.transpose(de_density)
    except:
        de_density = np.array(pfisr_file['NeFromPower']
                              ['dNeFrac'])[:, beam_num, :]
        de_density = np.transpose(de_density)
        de_density = de_density * e_density

    # Close file
    pfisr_file.close()
    
    return utc_time, unix_time, pfisr_altitude, e_density, de_density
"""
Functions to take ISR electron density and convert to an ion production rate

@author Riley Troyer
science@rileytroyer.com
"""

# Libraries
import datetime
import logging
import msise00
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

# This might set off some warnings, but I think they can be ignored
# See the documents on how to get this setup properly
from src.models.kaeppler_chemistry import Chemistry as chemistry

def barrett_hays_range_energy_func(K:float) -> float:
    """Function to define mass range of electron in air for a specific
    energy K in eV. From Barett & Hays 1976
    INPUT
    K - energy of electron in eV
    OUTPUT
    R - mass range of particle in kg m^-2 
    """
    # Convert energy to keV to match formula
    K = K/1000
    
    # Range function
    R = 4.3e-7 + 5.36e-6 * K**(1.67) - 0.38e-8 * K**(-0.7)
    
    # Convert R from g/cm^2 to kg/m^2
    R = R * 10
    
    return R

def get_msis_density(run_time:datetime.datetime, altitude_bins:np.ndarray, max_alt:float=1001e3,
                     glat:float=65.117, glon:float=212.540) -> 'np.ndarray, scipy.interpolate':
    """Function to get MSIS calculated atmospheric densities.
    DEPENDENCIES
        msise00, numpy, scipy.interpolate.interp1d
    INPUT
    run_time - time to run msis code for
    altitudes - altitudes in meters to run msis code for
    max_alt - maximum altitude in meters to run msis for. 
               Function creates a high altitude log spaced array
               between the max of altitudes and the max_alt value.
               This is primarily for approximating an indefinite integral.
    glat, glon - latitude and longitude to run model for.
    OUTPUT
    total_msis_alt - altitudes values in meters including original array and
               high altitude array
    msis_interp_density - 1d interpolation of msis density spanning entire altitude
               range.
    """
    
    # Run msis for lower altitudes
    logging.disable(logging.INFO) # lots of info logging in this function
    msis_run_low = msise00.run(time=run_time, altkm=altitude_bins/1000,
                               glat=glat, glon=glon)

    # Define a higher altitude array
    msis_alt_high = np.logspace(np.log10(max(altitude_bins)+1),
                                np.log10(max_alt), 20)
    
    # Run msis for these higher altitudes
    msis_run_high = msise00.run(time=run_time, altkm=msis_alt_high/1000,
                               glat=glat, glon=glon)
    # Reenable info logging
    logging.disable(logging.NOTSET)

    # Get total density data
    msis_density_low = msis_run_low['Total'].data[0, :, 0, 0]
    msis_density_high = msis_run_high['Total'].data[0, :, 0, 0]

    # Combine altitude and densities from low and high altitudes
    total_msis_alt = np.concatenate((altitude_bins, msis_alt_high))
    total_msis_density = np.concatenate((msis_density_low,
                                         msis_density_high))

    # Create a scipy interpolation function to define density v. altitude
    msis_interp_density = interp1d(total_msis_alt, total_msis_density)
    
    return total_msis_alt, msis_interp_density

def mass_distance(z_i:int, total_msis_density:np.ndarray,
                  total_msis_alt:np.ndarray, I:float=0) -> float:
    """Function to mass distance of particle traveling some distance
    into the atmosphere. Denoted s in the derivations.
    Using trapezoid rule for this, which seems to be good enough
    INPUT
    z - index of altitude that particle reached to
    total_msis_density - msis density calculated at total_msis_alt
    total_msis_alt - altitudes values in meters including original array and
                     high altitude array
    I - angle of magnetic inclination at measuring site in radians
    OUTPUT
    s - mass distance in kg m^-2
    """
    
    # Calculate mass distance traveled 
    s = (1/np.cos(I)) * trapezoid(total_msis_density[z_i:],
                                  total_msis_alt[z_i:])
    
    return s

def recombination_coeff(z:float, alpha_type:str='vickrey', base_dir:str='./') -> float:
    """Function defining recombination coefficient
    INPUT
    z - altitude in kilometers
    alpha_type  - what recombination coefficient to use
                other option: osepian, gledhill
    base_dir - how to get to the base project directory
    OUTPUT
    alpha - recombination coefficient in m^3/s
    """
    
    if alpha_type == 'vickrey':
        
        alpha = 2.5e-12 * np.exp(-z/51.2)
    
    if alpha_type == 'osepian':
        
        # Read in file with effective recombination coefficient values
        alpha_filename = base_dir + '/models/effective-recombination-coefficient-osepian-etal-2009.txt'
        alpha_data = np.loadtxt(alpha_filename, skiprows=6)

        # Get altitude and coeff from data
        alpha_alt = alpha_data[:, 0]
        alpha_coeff = alpha_data[:, 1]*1e-6

        # Append formula value at 144 km
        alpha_alt = np.append(alpha_alt, 144)
        alpha_coeff = np.append(alpha_coeff, recombination_coeff(144))

        # Create an interpolated function from this
        #...values outside set to 0
        alpha_interp = interp1d(alpha_alt, alpha_coeff,
                                 bounds_error=False, fill_value=0)  
        
        alpha = alpha_interp(z)
    
    if alpha_type == 'gledhill':
        
        alpha = (4.3e-6 * np.exp(-2.42e-2 * z) 
                 + 8.16e12 * np.exp(-0.524 * z))
        
        # Convert to m^3
        alpha = alpha * 1e-6
    
    return alpha

def isr_ion_production_rate(e_density:np.ndarray, de_density:np.ndarray,
                            pfisr_altitude:np.ndarray, altitude_bins:np.ndarray,
                            slice_n:int, unix_time:np.ndarray,
                            pfrr_lat:float=65.117, pfrr_lon:float=212.540,
                            alpha_type:str='vickrey', base_dir:str='./') -> 'np.ndarray, np.ndarray, np.ndarray':
    """Function to estimate the ion production rate from isr measurements.
    There are many ways to do this that use differing chemistry
    assumptions. Vickrey 1982 is a very basic assumption for the 
    E-region and is extended to D-region. Gledhill 1986 is slightly more
    sophisticated using a best fit of many D-region measurements during
    night time aurora. Osepian 2009 is based on measurements during 
    solar proton events. The Stanford model is based on the chemistry
    model of Lehtinen 2007. 
    INPUT
    e_density - electron density from pfisr radar
    de_density - error in electron density
    pfisr_altitude - altitude at which e_density is measured
    altitude_bins - altitude to calculate model for
    slice_n - data slice of isr data to take
    unix_time - unix time for chemistry model code
    pfrr_lat, pfrr_lon - latitude and longitude of instrument
    alpha_type - what recombination coefficient to use
                other option: vickrey, osepian, gledhill, stanford
    base_dir - how to get to the base project directory
    OUTPUT
    q_estimate - estimated ion production rate m^-2 s^-1
    dq_estimate - error in ion production rate
    alphas - recombination coefficients
    """

    if ((alpha_type=='vickrey')
        or (alpha_type=='osepian')
        or (alpha_type=='gledhill')):
        # Read in density and errors in those measurements 
        #...for specific time
        e_density_slice = e_density[:, slice_n]
        de_density_slice = de_density[:, slice_n]

        # Make interpolation model of this data with respect to altitude
        #...but only do this for altitudes > defined minimum value,
        #...below this data can be weird
        pfisr_density_interp = interp1d(pfisr_altitude, e_density_slice)

        # Same interpolation except for error in density
        pfisr_error_interp = interp1d(pfisr_altitude, de_density_slice)

        # Calculate all recombination coeffcients
        alphas = np.array([recombination_coeff(z/1000,
                                               alpha_type=alpha_type,
                                               base_dir=base_dir)
                           for z in altitude_bins])

        # Multiply by pfisr density to get estimate of production rate
        #...keep sign in calculation, so don't bias high
        pfisr_signs = np.sign(pfisr_density_interp(altitude_bins))
        q_estimate = (alphas 
                      * pfisr_density_interp(altitude_bins)**2)

        # Get error dq = 2*alpha*n*dn
        dq_estimate = (2 * alphas * pfisr_density_interp(altitude_bins)
                       * pfisr_error_interp(altitude_bins))
        dq_estimate = abs(dq_estimate)
    
    elif alpha_type=='stanford':
        # Read in the chemistry class
        chem = chemistry(SteadyStateTime = 100., ISRIntegrationTime = 60.)

        # Read in density and errors in those measurements
        #...for specific time
        e_density_slice = e_density[:, slice_n]
        de_density_slice = de_density[:, slice_n]

        # Make interpolation model of this data with respect to altitude
        #...but only do this for altitudes > defined minimum value,
        #...below this data can be weird
        pfisr_density_interp = interp1d(pfisr_altitude, e_density_slice)

        # Same interpolation except for error in density
        pfisr_error_interp = interp1d(pfisr_altitude, de_density_slice)

        # Multiply by pfisr density to get estimate of production rate
        #...keep sign in calculation, so don't bias high
        pfisr_signs = np.sign(pfisr_density_interp(altitude_bins))

        # Initialize ionization in chemistry class
        #...input altitude in km and stepsize of altitude bins required
        alt_step = altitude_bins[1] - altitude_bins[0]
        chem.Set_Inital_Ionization(unix_time[slice_n],
                                   pfrr_lat, pfrr_lon,
                                   min(altitude_bins)/1000,
                                   max(altitude_bins)/1000,
                                   alt_step/1000)

        # Run chemistry code to convert density to ionization rate.
        #...make sure to run initial ionziation code first
        #...input should be in km and 1/cm^3
        #...this will output in units of cgs
        q_estimate = chem.Calculate_Ionization_From_Ne(altitude_bins/1000,
                                pfisr_density_interp(altitude_bins)/1e6,
                                chem.DregionChem)

        # Add back in negatives and convert to SI
        q_estimate = q_estimate * pfisr_signs * 1e6

        # Calculate the extracted effective recombination coefficient
        alphas = q_estimate / pfisr_density_interp(altitude_bins)**2
        
        # Match Gledhill above 90 km
        e_region_cond = altitude_bins >= 90e3
        alphas[e_region_cond] = [recombination_coeff(z/1000,
                                               alpha_type='gledhill')
                                 for z in altitude_bins[e_region_cond]]
        
        # Recalculate ion production rate
        q_estimate = alphas * pfisr_density_interp(altitude_bins)**2

        # Get error dq = 2*alpha*n*dn
        dq_estimate = (2 * alphas
                       * pfisr_density_interp(altitude_bins)
                       * pfisr_error_interp(altitude_bins))
        dq_estimate = abs(dq_estimate)
        
    else:
        logging.warning('Could not complete, please enter good alpha type.')

    
    return q_estimate, dq_estimate, alphas


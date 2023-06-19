"""Functions to run the maximum entropy inversion method fitting for the 
PFISR ion production rate.

@author Riley Troyer
science@rileytroyer.com
"""

import logging
import numpy as np


def estimate_initial_number_flux(energy_bins:np.ndarray, matrix_A:np.ndarray) -> np.ndarray:
    """Function to estimate the intial number flux for each energy bin
    INPUT
    energy_bins - energy values defining energy bins in eV
    matrix_A - inversion matrix
    OUTPUT
    initial_num_flux - estimated number flux in m^-2 s^-1 for each energy bin
    """
    
    # Make an initial guess of the number flux
    initial_num_flux = np.ones(len(energy_bins))*(1e12/len(energy_bins))

    # Divide by energy bin widths
    bin_widths = energy_bins - np.roll(energy_bins, shift=1)

    # Fix first value
    bin_widths[0] = energy_bins[0] - 0

    # Set initial guess
    initial_num_flux = initial_num_flux/bin_widths

    # If full column of A matrix is zero set initial flux to zero
    for j in range(len(energy_bins)):

        if np.sum(matrix_A[:, j]) == 0:
            initial_num_flux[j] = 0
            
    return initial_num_flux

def maximum_entropy_iteration(initial_num_flux:np.ndarray, altitude_bins:np.ndarray,
                              energy_bins:np.ndarray, matrix_A:np.ndarray,
                              q_estimate:np.ndarray, dq_estimate:np.ndarray,
                              max_iterations:int,
                              convergence:float=1e-2) -> 'np.ndarray, float, int':
    """Function to perform the maximum entropy iterative process to
    approximate inversion of matrix A. 
    Process is outlined in Semeter & Kamalabadi 2005 and Troyer et al. (2022).
    INPUT
    initial_num_flux - initial guess of number flux for each energy bin 
               in m^-2 s^-1
    altitude_bins - altitude values in meters defining altitude bins
    energy_bins - energy values in eV defining energy bins
    matrix_A - matrix that iteration is trying to invert
    q_estimate - estimated ion production rate from ISR m^-2 s^-1
    dq_estimate - error in ion production rate of ISR
    max_iterations - don't do more than this if can't converge
    convergence - change in chi squared value to aim for.
    OUTPUT
    new_num_flux - estimated number flux for energy bins in m^-2 s^-1
    reduced_chi_square - error in modeled fit
    good_alt_index - lower than this won't be good data
    """
    
    # Set previous value to initial at start
    old_num_flux = initial_num_flux
    new_num_flux = np.zeros(len(initial_num_flux))  
    
    # Create array to store all minimum j values
    min_js = np.zeros(len(altitude_bins), dtype=int)

    # Find all nonzero indices of A matrix
    nonzero_args = np.argwhere(matrix_A > 0)

    for i in range(len(min_js)):

        non_zeros = nonzero_args[nonzero_args[:, 0] == i]

        # If there are no non zero values in row, then set to 
        #...first instance
        if len(non_zeros) == 0:
            min_js[i] = 0

        # Otherwise find the minimum j
        else:
            min_js[i] = min(non_zeros[:, 1])

    # Initialize values
    old_chi_square = 1e3
    chi_square = 0
    old_chi2_diff = 1e9
    converged = True
    count = 0

    # Run interations until convergence or count is met
    while (old_chi2_diff > convergence):

        # Check count
        if count > max_iterations:
            logging.warning(f'Unable to converge. ' 
                            f'Max iterations reached with chi2 = {chi_square:0.2f}')
            break

        # Construct the t vector
        t = 1/np.dot(matrix_A[:, min_js], old_num_flux[min_js])

        # Adjust for infinite values in regions without a nonzero j
        t[t == np.inf] = 0        

        for j in range(len(energy_bins)):

            # Construct c vector
            frac = np.inner(matrix_A, old_num_flux)/q_estimate
            c = 20 * (1 - frac) * t

            # Account for nan and infinite values
            #...this is why warning is raised
            c[np.isnan(c)] = 0
            c[c == -np.inf] = 0
            c[c == np.inf] = 0

            # Define w constant
            w = np.ones(len(altitude_bins))/len(altitude_bins)

            # Summation of matrix elements
            i_sum = np.sum(w*c*matrix_A[:, j])

            # New guess
            new_num_flux[j] = old_num_flux[j]/(1-old_num_flux[j]*i_sum)

        # Check chi squared, but only on altitudes that A is defined for
        diff=q_estimate-np.dot(matrix_A, new_num_flux)
        chi_square_array = diff**2/dq_estimate**2

        # Set undefined values to zero
        chi_square_array[np.isnan(chi_square_array)] = 0
        chi_square_array[chi_square_array == np.inf] = 0
        chi_square_array[chi_square_array == -np.inf] = 0
        
        # Get the chi squared value
        chi_square = np.sum(chi_square_array)
        
        # Do a convergence test, make sure it isn't blowing up
        if (old_chi2_diff 
            < abs(old_chi_square - chi_square)) & (count > 1000):
            logging.warning(f'Not converging. Stopping. '
                            f'chi2 = {chi_square:0.2f}')
            converged = False
            break 

        # Set old values to new
        old_num_flux = np.copy(new_num_flux)
        old_chi2_diff = abs(old_chi_square - chi_square)
        old_chi_square = chi_square

        # Set count
        count = count + 1
        
    # Get reduced chi square, which should be around 1
    diff=q_estimate-np.dot(matrix_A, new_num_flux)
    dof = len(q_estimate[dq_estimate < q_estimate]) - matrix_A.shape[1]
    
    # Notify of convergence
    if ((count < max_iterations) & (converged == True)):
        logging.info(f'Convergence reached. Iterations: {count-1}. '
                     f'Reduced chi2: {chi_square/dof:0.2f}')
        
    return new_num_flux, chi_square, dof, converged
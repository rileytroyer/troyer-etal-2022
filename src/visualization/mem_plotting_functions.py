"""
Functions to plot results of maximum entropy inversion method fitting.

@author Riley Troyer
science@rileytroyer.com
"""

# Libraries
import datetime
import gc
from matplotlib import pyplot as plt
import numpy as np

def inversion_density_plot(inversion_results:dict, run_time:datetime.datetime) -> None:
    """Function to create a plot of the inversion 
    electron density.
    INPUT
    inversion_results - dictionary of inversion results
    run_time - time to create plot for
    """
    # Get altitude values
    altitude_bins = inversion_results[run_time]['altitude']

    # Get measured density
    pfisr_density_plot = inversion_results[run_time]['measured_density']
    pfisr_density_plot = pfisr_density_plot

    # Get error in density
    pfisr_error_plot = inversion_results[run_time]['measured_error']
    pfisr_error_plot = pfisr_error_plot

    # Initial guess
    initial_guess_plot = inversion_results[run_time]['initial_density']
    initial_guess_plot = initial_guess_plot
    
    # Finally modeled guess
    final_guess_plot = inversion_results[run_time]['modeled_density']
    final_guess_plot = final_guess_plot
    
    # Get reduced chi2
    chi2 = inversion_results[run_time]['chi2']
    dof = inversion_results[run_time]['dof']
    reduced_chi2 = chi2/dof

    # Plot figure of initial guess, real data and fit
    fig, ax = plt.subplots()

    # Titles and axis labels
    ax.set_title(str(run_time) + r' $\chi^2_{red}=$' 
                 + str(round(reduced_chi2, 2)),
                 fontsize=14, fontweight='bold')

    ax.set_ylabel('Altitude [km]', fontsize=14)
    ax.set_xlabel(r'Electron Density [m$^{-3}$]', fontsize=14)

    # Axis
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.tick_params(axis='y', which='major', labelsize=14)

    ax.set_xscale('log')
    #ax.set_xlim(1e10, 1e12)
    #ax.set_ylim(75, 140)

    # Plot PFISR data
    ax.plot(pfisr_density_plot, altitude_bins/1000,
            color='k', linewidth=2, label = 'PFISR')

    # Plot initial guess
    ax.plot(initial_guess_plot, altitude_bins/1000,
            color='C2', linewidth=2, label = 'Initial Guess')

    # Plot final guess
    ax.plot(final_guess_plot, altitude_bins/1000,
            color='C1', linewidth=2, label = 'Final Guess')
    
    ax.plot(pfisr_error_plot, altitude_bins/1000,
            color='red', label = 'PFISR Error')
    
    ax.set_xlim(1e9, 0.5e12)

    plt.legend()

    plt.tight_layout()

def inversion_numflux_plot(inversion_results:dict, run_time:datetime.datetime) -> None:
    """Function to create a plot of the inversion 
    energy spectrum.
    INPUT
    inversion_results - dictionary of inversion results
    run_time - time to create plot for
    """
    # Get energy values
    energy_bins = inversion_results[run_time]['energy_bins']
    
    # Get modeled number flux values
    num_flux = inversion_results[run_time]['modeled_flux']
    
    # Get differential number flux by multiplying by energy bin width
    bin_widths = energy_bins - np.roll(energy_bins, shift=1)
    
    # Fix first value
    bin_widths[0] = energy_bins[0] - 0
    
    num_flux = num_flux*bin_widths
    
    # Get reduced chi2
    chi2 = inversion_results[run_time]['chi2']
    dof = inversion_results[run_time]['dof']
    reduced_chi2 = chi2/dof

    # Plot figure of energy spectrum
    fig, ax = plt.subplots()

    # Titles and axis labels
    ax.set_title(str(run_time) + r' $\chi^2_{red}=$' 
                 + str(round(reduced_chi2, 2)),
                 fontsize=14, fontweight='bold')

    ax.set_ylabel(r'Number Flux [m$^{-2}$ s$^{-1}$ eV$^{-1}$]',
                  fontsize=14)
    ax.set_xlabel('Energy [eV]', fontsize=14)

    # Axis
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.tick_params(axis='y', which='major', labelsize=14)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Plot the energy
    ax.plot(energy_bins, num_flux)

    plt.tight_layout()

def save_inversion_density_plot(inversion_results:dict, run_time:datetime.datetime,
                                output_dir:str) -> None:
    """Function to create and save a plot of the inversion 
    electron density.
    INPUT
    inversion_results - dictionary of inversion results
    run_time - time to create plot for
    output_dir - where to store the images
    """
    # Get altitude values
    altitude_bins = inversion_results[run_time]['altitude']

    # Get measured density
    pfisr_density_plot = inversion_results[run_time]['measured_density']
    pfisr_density_plot = pfisr_density_plot
    
    # Get error in density
    pfisr_error_plot = inversion_results[run_time]['measured_error']
    pfisr_error_plot = pfisr_error_plot

    # Initial guess
    initial_guess_plot = inversion_results[run_time]['initial_density']
    initial_guess_plot = initial_guess_plot
    
    # Finally modeled guess
    final_guess_plot = inversion_results[run_time]['modeled_density']
    final_guess_plot = final_guess_plot
    
    # Get reduced chi2
    chi2 = inversion_results[run_time]['chi2']
    dof = inversion_results[run_time]['dof']
    reduced_chi2 = chi2/dof

    # Plot figure of initial guess, real data and fit
    fig, ax = plt.subplots()

    # Titles and axis labels
    ax.set_title(str(run_time) + r' $\chi^2_{red}=$' 
                 + str(round(reduced_chi2, 2)),
                 fontsize=14, fontweight='bold')

    ax.set_ylabel('Altitude [km]', fontsize=14)
    ax.set_xlabel(r'Electron Density [m$^{-3}$]', fontsize=14)

    # Axis
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.tick_params(axis='y', which='major', labelsize=14)

    ax.set_xscale('log')
    #ax.set_xlim(1e10, 1e12)
    #ax.set_ylim(75, 140)

    # Plot PFISR data
    ax.plot(pfisr_density_plot, altitude_bins/1000,
            color='k', linewidth=2, label = 'PFISR')

    # Plot initial guess
    ax.plot(initial_guess_plot, altitude_bins/1000,
            color='C2', linewidth=2, label = 'Initial Guess')

    # Plot final guess
    ax.plot(final_guess_plot, altitude_bins/1000,
            color='C1', linewidth=2, label = 'Final Guess')
    
    ax.plot(pfisr_error_plot, altitude_bins/1000,
            color='red')
    
    #ax.set_xlim(1e9, 1e12)

    plt.legend()

    plt.tight_layout()

    
    fig_filename = (output_dir + 'e-density-'
                    + str(run_time.date())
                    + '_' + str(run_time.hour).zfill(2)
                    + '-' + str(run_time.minute).zfill(2)
                    + '-' + str(run_time.second).zfill(2)
                    + '.jpg')
    plt.savefig(fig_filename, dpi=150)
    
    # Close the figure
    #...axis
    plt.cla()
    #...figure
    plt.clf()
    #...figure windows
    plt.close('all')
    #...clear memory
    gc.collect()

def save_inversion_numflux_plot(inversion_results:dict, run_time:datetime.datetime,
                                output_dir:str) -> None:
    """Function to create and save a plot of the inversion 
    energy spectrum.
    INPUT
    inversion_results - dictionary of inversion results
    run_time - time to create plot for
    output_dir - where to store the images
    """
    # Get energy values
    energy_bins = inversion_results[run_time]['energy_bins']
    
    # Get modeled number flux values
    num_flux = inversion_results[run_time]['modeled_flux']
    
    # Get differential number flux by multiplying by energy bin width
    bin_widths = energy_bins - np.roll(energy_bins, shift=1)
    
    # Fix first value
    bin_widths[0] = energy_bins[0] - 0
    
    num_flux = num_flux*bin_widths
    
    # Get reduced chi2
    chi2 = inversion_results[run_time]['chi2']
    dof = inversion_results[run_time]['dof']
    reduced_chi2 = chi2/dof

    # Plot figure of energy spectrum
    fig, ax = plt.subplots()

    # Titles and axis labels
    ax.set_title(str(run_time) + r' $\chi^2_{red}=$' 
                 + str(round(reduced_chi2, 2)),
                 fontsize=14, fontweight='bold')

    ax.set_ylabel(r'Number Flux [m$^{-2}$ s$^{-1}$ eV$^{-1}$]',
                  fontsize=14)
    ax.set_xlabel('Energy [eV]', fontsize=14)

    # Axis
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.tick_params(axis='y', which='major', labelsize=14)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Plot the energy
    ax.plot(energy_bins, num_flux)

    plt.tight_layout()
    
    fig_filename = (output_dir + 'number-flux-'
                    + str(run_time.date())
                    + '_' + str(run_time.hour).zfill(2)
                    + '-' + str(run_time.minute).zfill(2)
                    + '-' + str(run_time.second).zfill(2)
                    + '.jpg')
    plt.savefig(fig_filename, dpi=150)
    
    # Close the figure
    #...axis
    plt.cla()
    #...figure
    plt.clf()
    #...figure windows
    plt.close('all')
    #...clear memory
    gc.collect()
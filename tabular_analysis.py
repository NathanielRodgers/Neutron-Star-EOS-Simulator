#!/usr/bin/env python
# coding: utf-8

# In[87]:


import matplotlib.pyplot as plt
import numpy as np


# In[92]:


def read_tabular_Sly(baryon_number_file, thermo_file):
    '''
    Reads tabulated data from Sly9

    Parameters:
    baryon_number_file : file of baryon number densities
    thermo_file : file of pressure, energy density, and more values

    Returns:
    mass-density, pressure, energy-density arrays in cgs units
    '''
    # data from https://compose.obspm.fr/eos/86, headers defined in CompOSE guidebook
    # may not work 1:1 with other datasets, so read documentation on unit labeling and column placement

    # Load ComPOSE files
    nb_all = np.loadtxt(baryon_number_file, skiprows=2)      # baryon number density in fm^-3
    thermo_raw = np.loadtxt(thermo_file, skiprows = 1)  # thermo table

    # Constants
    m_n = 1.67492749804e-24  # g, neutron mass
    fm_to_cm3 = 1e-39         # 1 fm^-3 = 1e39 cm^-3

    # Convert baryon number density to mass density
    rho_all = nb_all * m_n * 1e39  # g/cm^3

    # Mask to keep only rho >= 3e14
    mask = rho_all >= 3e14
    rho = rho_all[mask]

    # Apply same mask to thermo table rows
    thermo_slice = thermo_raw[mask]

    # get pressure and density
    nb_slice = nb_all[mask]
    Q1 = thermo_slice[:, 3]  # p / nb in MeV
    Q7 = thermo_slice[:, 10] # e per baryon (dimensionless)

    # Convert to physical units
    # 1 MeV/fm^3 ~ 1.60218e33 erg/cm^3, but here Q1 is p/nb, so multiply by nb
    p = Q1 * nb_slice * 1.60218e33  # erg/cm^3
    epsilon = nb_slice * m_n * 1e39 * (1 + Q7)  # total energy density in g/cm^3 * c^2

    return rho, p, epsilon


# In[93]:


def graph_tabular(baryon_number_file, thermo_file, boundaries = None):
    '''
    Takes tabular data from Sly9 and graphs it, along with input density boundaries

    Parameters:
    baryon_number_file : file of baryon number densities
    thermo_file : file of pressure, energy density, and more values
    boundaries : array of log densities

    returns:
    log-log density vs pressure graph
    '''

    data = read_tabular_Sly(baryon_number_file, thermo_file)
    density_dat, pressure_dat, epsilon_dat = data

    for density, pressure in zip(density_dat, pressure_dat):
        plt.scatter(np.log10(density), np.log10(pressure), color = 'black')

    if boundaries:
        for i in boundaries:
            plt.axvline(x=i, color = 'grey', linestyle='-.')
        plt.scatter([],[], color='grey', linestyle='-.', label = 'Density Segment Boundaries')


    plt.xlabel('log density (g/cm^3)')
    plt.ylabel('log pressure (erg/cm^3)')
    plt.title('SLy9 Tabular EOS Density vs Pressure (logspace)')
    ticks = np.arange(14.4, 15.5, 0.1)
    plt.xticks(ticks)
    plt.legend()
    plt.show()
    return


def get_identity(input_matrix):
    n,m = input_matrix.shape
    iFlat = np.ones(n)
    i = np.diag(iFlat)
    return i
def initialize_equation(x, y, yerr = None):
    n,m = x.shape
    ones = np.ones((n,m))
    A = np.append(ones, x, axis=0).T

    Y = y.T

    if yerr is None:
        return A, Y, None

    C = (np.diag(yerr))**2

    return A, Y, C
def solve_equation(x, y, yerr = None):

    A, Y, C = initialize_equation(x, y, yerr)

    if yerr is not None:
        # define C inverse
        i_C = get_identity(C)
        Cinv = np.linalg.solve(C,i_C)
        # Solve for AT*Cinv
        ATCinv = np.matmul(A.T, Cinv)

        # Solve for [AT*Cinv*A]inv
        ATCinvA = np.matmul(ATCinv, A)
        i_ATCinvA = get_identity(ATCinvA)
        ATCinvA_inv = np.linalg.solve(ATCinvA, i_ATCinvA)

        # Solve for AT*Cinv*Y
        ATCinvY = np.matmul(ATCinv, Y)

        X = np.matmul(ATCinvA_inv, ATCinvY)

        return X, ATCinvA_inv

    # so then yerr is none:
    # same equation, just not weighing everything
    ATA = np.matmul(A.T, A)

    # RHS:
    i_ATA = get_identity(ATA)
    ATA_inv = np.linalg.solve(ATA, i_ATA)

    # LHS:
    ATY = np.matmul(A.T, Y)

    # Solution:
    X = np.matmul(ATA_inv, ATY)

    return X, ATA_inv
def best_fit(x, y, yerr = None):

    X, cov_st_err = solve_equation(x, y, yerr)
    b = X[0]
    b_err = np.sqrt(cov_st_err[0,0])
    m = X[1]
    m_err = np.sqrt(cov_st_err[1,1])

    return m, m_err, b, b_err


# In[129]:


def fit_tabular(densities, pressures, boundaries = None):
    '''
    Computes best fit lines of density and pressure across user-input boundaries

    Parameters:
    densities, pressure : array of densities and pressures in cgs
    boundaries : array of log densities

    returns:
    log-log density vs pressure graph, best-fit lines of each region
    '''


    densities = np.log10(densities)
    pressures = np.log10(pressures)

    if boundaries is not None:
        density_segments = []
        pressure_segments = []
        num_bounds = len(boundaries) 
        for index, i in enumerate(boundaries):

            # first boundary case
            if index == 0:
                mask = densities < boundaries[index]
                density_segments.append(densities[mask])
                pressure_segments.append(pressures[mask])

            # every other case
            else:
                mask = (densities >= boundaries[index - 1]) & (densities < boundaries[index])
                density_segments.append(densities[mask])
                pressure_segments.append(pressures[mask])


        # final boundary case
        mask = densities >= boundaries[-1]
        density_segments.append(densities[mask])
        pressure_segments.append(pressures[mask])

        # fit a line to each of these segments
        lstsq_slopes = []
        lstsq_intercepts = []
        for index, i in enumerate(density_segments):
            lstsq_densities = np.array([i])
            lstsq_pressures = np.array(pressure_segments[index])
            m, m_err, b, b_err = best_fit(lstsq_densities, lstsq_pressures)
            lstsq_slopes.append(m)
            lstsq_intercepts.append(b)
        lstsq_slopes = np.array(lstsq_slopes)
        lstsq_intercepts = np.array(lstsq_intercepts)

        # plot the segments and best fit lines
        for index, i in enumerate(lstsq_slopes):
            plt.scatter(density_segments[index], pressure_segments[index], alpha = 0.1)
            plt.plot(density_segments[index], lstsq_slopes[index] * density_segments[index] + lstsq_intercepts[index], label = i)

        # plot boundary lines
        for i in boundaries:
            plt.axvline(x=i, color = 'grey', linestyle='-.')
        plt.scatter([],[], color='grey', linestyle='-.', label = 'Density Segment Boundaries')

        plt.xlabel('log density (g/cm^3)')
        plt.ylabel('log pressure (erg/cm^3)')
        plt.title('Fitting the SLy9 Data')
        plt.legend()
        plt.show()

        return lstsq_slopes

    else:
        # if no boundaries are input
        lstsq_densities = np.array([densities])
        lstsq_pressures = np.array([pressures])
        m, m_err, b, b_err = best_fit(lstsq_densities, lstsq_pressures)
        plt.scatter(densities, pressures, alpha = 0.1)
        plt.plot(densities, m * densities + b, label = m)

        plt.xlabel('log density (g/cm^3)')
        plt.ylabel('log pressure (erg/cm^3)')
        plt.title('Fitting the SLy9 Data')
        plt.legend()
        plt.show()

        return m





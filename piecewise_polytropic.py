import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm

c = 2.99792458 * 1e10
G = 6.6742 * 1e-8
Msun = 1.9891 * 1e33
    
# Making general length constants to shorten equations
    
L = G * Msun / c**2

# establishing normalizing values
def normalizing_values(gamma):
    # c = Msun = G = 1

    radius_ = L
    mass_ = Msun
    mass_density_ = Msun / L**3
    internal_energy_ = (Msun * c**2) / L**3
    pressure_ = (Msun * c**2) / L**3
    k_ = ((Msun * c**2) / L**3) * (Msun / L**3)**(-gamma)
    return radius_, mass_, mass_density_, internal_energy_, pressure_, k_

def mass_density_function(pressure, k, gamma):
    # polytopic EOS
    mass_density = (pressure / k)**(1/gamma)
    return mass_density

def pressure_function(mass_density, k, gamma):
    # polytropic EOS
    pressure = k * mass_density**gamma
    return pressure

def internal_energy_function(pressure, mass_density, gamma):
    # specific internal energy
    internal_energy = pressure / ((gamma - 1) * mass_density)
    return internal_energy

def energy_density_function(pressure, mass_density, gamma,k):
    # energy density function
    
    internal_energy = internal_energy_function(pressure, mass_density, gamma)
    energy_density = mass_density * (1 + internal_energy)
    return energy_density

def energy_density_a_function(pressure, mass_density, gamma, k, past_vals):
    # continuity-enforcing energy density function
    # equation from Read 2008
    gamma_past, mass_density_past, k_past = past_vals
    pressure_past = pressure_function(mass_density_past, k_past, gamma_past)
    
    energy_density_past = energy_density_function(pressure_past, mass_density_past, gamma_past, k_past)
    a = (energy_density_past / mass_density_past) - 1 - (k / (gamma - 1) * (mass_density_past ** gamma_past))
    energy_density = (1+a)*mass_density + k /(gamma - 1) * mass_density**gamma
    return energy_density
    
def speed_of_sound(gamma, k, mass_density):
    # speed of sound is given by (dp/drho) / (deps/drho)
    dp_drho = gamma * k * mass_density**(gamma - 1)
    deps_drho = 1 + (gamma * k / (gamma - 1)) * mass_density**(gamma-1)
    return dp_drho / deps_drho

def derivatives(radius, pressure, mass, k, gamma, tidal_y, a_trigger = False, past_vals = None):
    # star structure
    mass_density = mass_density_function(pressure, k, gamma)

    # checking to see if continuity needs to be established for our parametrized energy density
    if a_trigger:
        energy_density = energy_density_a_function(pressure, mass_density, gamma, k, past_vals)
    else:
        energy_density = energy_density_function(pressure, mass_density, gamma, k)
    
    dphi = (mass + 4 * np.pi * radius**3 * pressure) / (radius * (radius - 2 * mass))
    dp = -(energy_density + pressure) * dphi
    dm = 4 * np.pi * radius**2 * energy_density

    
    # tidal deformability
    # y' = -y^2 / r - F(r) / r * y - r*Q(r)
    # evolving tidal deformability along with star structure

    # constants
    elambda = (1 - (2 * mass / radius)) ** -1
    sound_speed = max(speed_of_sound(gamma, k, mass_density), 1e-3)

    F = (1 - 4 * np.pi * radius**2 * (energy_density - pressure)) * elambda
    Q = (-6 / radius**2) * elambda + 4 * np.pi * (5*energy_density + 9*pressure + (energy_density + pressure) / sound_speed) * elambda - 4*dphi**2
    dy = -tidal_y**2 / radius - (F * tidal_y) / radius - radius * Q

    return dphi, dp, dm, dy


def RK4_step(radius, pressure, mass, k, gamma, dr, grav_potential_factor, tidal_y, a_trigger = False, past_vals = None):

    # run through rk4
    # integrating gravitational potential factor, mass, pressure, and tidal deformability
    # mass density is simply calculated from EOS
    phi1, p1, m1, y1 = derivatives(radius, pressure, mass, k, gamma, tidal_y, a_trigger, past_vals)
    phi2, p2, m2, y2 = derivatives(radius + ((1/2) * dr), pressure + ((1/2) * p1 * dr), mass + ((1/2) * m1 * dr), k, gamma, tidal_y + ((1/2) * y1 * dr), a_trigger, past_vals)
    phi3, p3, m3, y3 = derivatives(radius + ((1/2) * dr), pressure + ((1/2) * p2 * dr), mass + ((1/2) * m2 * dr), k, gamma, tidal_y + ((1/2) * y2 * dr), a_trigger, past_vals)
    phi4, p4, m4, y4 = derivatives(radius + dr, pressure + (p3 * dr), mass + (m3 * dr), k, gamma, tidal_y + (y3 * dr), a_trigger, past_vals)

    new_grav_potential_factor = grav_potential_factor + ((1/6) * (phi1 + 2*phi2 + 2*phi3 + phi4) * dr)
    new_mass = mass + ((1/6) * (m1 + 2*m2 + 2*m3 + m4) * dr)
    new_pressure = pressure + ((1/6) * (p1 + 2*p2 + 2*p3 + p4) * dr)
    new_mass_density = mass_density_function(new_pressure, k, gamma)
    new_tidal_y = tidal_y + ((1/6) * (y1 + 2*y2 + 2*y3 + y4) * dr)

    return new_grav_potential_factor, new_mass, new_pressure, new_mass_density, new_tidal_y

def RK4_solver(central_mass_density, starting_radius, dr, gamma_values, k_values, mass_density_cutoffs, full_mass_density_values):

    
    # determine which parameter of gamma to start with
    mass_density_temp = 10**np.array(mass_density_cutoffs)
    initial_gamma_id = np.searchsorted(mass_density_temp, central_mass_density)
    initial_gamma = gamma_values[initial_gamma_id]

    # normalizing values have an _ after    
    radius_, mass_, mass_density_, internal_energy_, pressure_, k_ = normalizing_values(initial_gamma)

    gamma_values = np.array(gamma_values)

    # now im making every value dimensionless (c=G=Msun=1)
    full_mass_density_values = full_mass_density_values / mass_density_
    
    norm_dr = dr / radius_

    mass_density_cutoffs = np.array(mass_density_cutoffs)
    norm_mass_density_cutoffs = (10 ** mass_density_cutoffs) / mass_density_
    
    norm_starting_radius = starting_radius / radius_
    norm_central_mass_density = central_mass_density / mass_density_

    central_gamma_id = np.searchsorted(norm_mass_density_cutoffs, norm_central_mass_density)
    central_gamma = gamma_values[central_gamma_id]
    central_k = k_values[central_gamma_id]

    norm_central_pressure = central_k * norm_central_mass_density ** central_gamma
    norm_central_energy_density = energy_density_function(norm_central_pressure, norm_central_mass_density, central_gamma, central_k)
    norm_central_mass = (4/3) * np.pi * norm_central_energy_density * norm_starting_radius**3

    # initial condition for tidal deformability:
    # if H is the metric pertubation, y = R*H'/H, and y(r=0) ~ 2 - H(r->0).
    initial_tidal_y = 2 - 2/7 * np.pi * (5*norm_central_energy_density + 9*norm_central_pressure) * norm_starting_radius**2

    
    # creating lists for each of the values, to be appended to
    pressures = [norm_central_pressure]
    radii = [norm_starting_radius]
    masses = [norm_central_mass]
    mass_densities = [norm_central_mass_density]
    metric_potentials = [0]
    tidal_ys = [initial_tidal_y]

    while pressures[-1] > 1e-10:

        # update variables
        radius = radii[-1]
        mass = masses[-1]
        pressure = pressures[-1]
        grav_potential_factor = metric_potentials[-1]
        mass_density = mass_densities[-1]
        tidal_y = tidal_ys[-1]

        # determine which parameters of gamma and k to use based on mass density
        gamma_id = np.searchsorted(norm_mass_density_cutoffs, mass_density)
        gamma = gamma_values[gamma_id]
        k = k_values[gamma_id]

        # if we aren't in the lowest mass density bin for our parametric EOS, we need to
        # establish continuity. This tells us the parameters of the lower mass density bin.
        if gamma != gamma_values[0]:
            a_trigger = True
            gamma_past = gamma_values[gamma_id - 1]
            mass_density_past = full_mass_density_values[gamma_id - 1]
            k_past = k_values[gamma_id - 1]
            past_vals = gamma_past, mass_density_past, k_past
        else:
            a_trigger = False
            past_vals = None


        # integrating mass, pressure, mass density, gravitational potential factor, and tidal deformability y
        new_grav_potential_factor, new_mass, new_pressure, new_mass_density, new_tidal_y = RK4_step(radius, pressure, mass, k, gamma, norm_dr, grav_potential_factor, tidal_y, a_trigger, past_vals)

        # if it goes negative in this update, kill it
        if np.isnan(new_pressure):
            break
        
        # add values to arrays
        metric_potentials.append(new_grav_potential_factor)
        masses.append(new_mass)
        pressures.append(new_pressure)
        mass_densities.append(new_mass_density)
        radii.append(radius + norm_dr)
        tidal_ys.append(new_tidal_y)

    # turn into numpy arrays
    metric_potentials = np.array(metric_potentials)
    masses = np.array(masses)
    pressures = np.array(pressures)
    mass_densities = np.array(mass_densities)
    radii = np.array(radii)
    tidal_ys = np.array(tidal_ys)




    
    # add dimensions back in
    masses = masses * mass_
    pressures = pressures * pressure_
    mass_densities = mass_densities * mass_density_
    radii = radii * radius_

    # now we calculate the actual tidal deformability
    # k2 and tidal are both the deformability
    # tidal has a significant dependence on M/R.
    C = (G / c**2) * (masses[-1] / radii[-1])
    y = tidal_ys[-1]
    k2 = 8/5*C**5 * (1-2*C)**2 * (2+2*C*(y-1)-y) / (2*C*(6-3*y+3*C*(5*y-8)) + 4*C**3*(13-11*y+C*(3*y-2) + 2*C**2*(1+y)) + 3*(1-2*C)**2*(2-y+2*C*(y-1))*np.log(1-2*C))
    tidal = 2/3 * k2 * C**-5

    
    # shift metric potentials to match exterior schwarzchild solution
    exterior_schw_sol = 0.5 * math.log(1 - (2*G*masses[-1] / (radii[-1] * c**2)))
    metric_potentials = metric_potentials + (exterior_schw_sol - metric_potentials[-1])

    return radii, masses, pressures, mass_densities, metric_potentials, tidal_ys, tidal, k2

def neutron_star(central_mass_densities, gamma_values, density_cutoffs = np.array([14.7, 14.9, 15]), starting_radius = 1, dr = 50, all_data = False):

    c = 2.99792458 * 1e10
    G = 6.6742 * 1e-8
    Msun = 1.9891 * 1e33
        
    # Making general length constants to shorten equations
        
    L = G * Msun / c**2
    
    # need four density values for the four piecewise components
    density_values = np.array([density_cutoffs[0] - 0.1])
    density_values = np.append(density_values, density_cutoffs)
    density_values = 10**density_values
    
    # from tabular
    initial_mass_density, initial_pressure = 10**14.68512146, 10**34.43729934

    # calculate values of k
    k0 = initial_pressure / (initial_mass_density ** gamma_values[0])
    k = [k0]
    for index, i in enumerate(density_values):
        if (len(k)) == (len(gamma_values)):
            break
        pressure_i = pressure_function(i, k[index], gamma_values[index])
        k.append(pressure_i / (density_values[index]**gamma_values[index+1]))
    k_ = ((Msun * c**2) / L**3) * (Msun / L**3)**(-np.array(gamma_values))
    k = np.array(k)
    k_values = k / k_
    

    # initializing lists
    masses_list = []
    radii_list = []
    tidal_deform_list = []
    k2_list = []
    data_list = []
    
    # initialize figure and colorbar
    fig, ax = plt.subplots(2, figsize = (12,14))
    norm = plt.Normalize(vmin=min(central_mass_densities), vmax=max(central_mass_densities))
    cmap = cm.viridis

    # loop over each given central mass (this will be the # of stars)
    for index, i in enumerate(central_mass_densities):
        if all_data:
            data = RK4_solver(i, starting_radius, dr, gamma_values, k_values, density_cutoffs, density_values)
            radii, masses, pressures, mass_densities, metric_potentials, tidal_ys, tidal_deform, k2 = data
            data_list.append(data)
        else:
            radii, masses, pressures, mass_densities, metric_potentials, tidal_ys, tidal_deform, k2 = RK4_solver(i, starting_radius, dr, gamma_values, k_values, density_cutoffs, density_values)
        # one last redundant indice check
        j = -1
        if np.isnan(masses[j]):
            j = -2
        # plot MR curve
        ax[0].plot(radii / 1e5, masses / Msun, color = cmap(norm(i)))
        ax[0].scatter(radii[j] / 1e5, masses[j] / Msun, color = cmap(norm(i)))
        masses_list.append(masses[j] / Msun)
        radii_list.append(radii[j] / 1e5)
        tidal_deform_list.append(tidal_deform)
        k2_list.append(k2)
        
        # speed of sound check -- must be < 1
        speed_vs_radius(ax[1], mass_densities, radii, gamma_values, density_cutoffs, k_values, cmap, norm)

    # add in color bars to plots
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax[0], orientation='vertical', label='Central Mass Density')
    fig.colorbar(sm, ax=ax[1], orientation='vertical', label='Central Mass Density')    

    # make plots pretty
    max_radius = np.max(radii_list)
    max_mass = np.max(masses_list)
    ax[0].plot(np.linspace(0, max_radius + 1.5, 2), max_mass * np.ones(2), color = 'black', linestyle = '--', alpha = 0.5, label = 'maximum computed radius')
    ax[0].plot(np.linspace(0, max_radius + 1.5, 2), 2 * np.ones(2), color = 'orange', linestyle = '-', alpha = 0.5, label = 'observed maximum')
    ax[0].set_xlabel('Radius (km)')
    ax[0].set_ylabel('Mass (M☉)')
    ax[0].set_title('Radius vs Mass - Neutron Star')
    ax[0].set_xlim(0, max_radius + 1.5)
    ax[0].set_ylim(0,None)
    ax[0].legend()

    domain_lower_limit = 0
    domain_upper_limit = max_radius + 1.5
    ax[1].plot([domain_lower_limit, domain_upper_limit], [1, 1], linestyle = '--', color = 'black', alpha = 0.5, label = 'Maximum Speed')
    ax[1].set_xlim(0, max_radius + 1.5)
    ax[1].set_ylim(0,None)
    ax[1].legend()
    
    plt.show()

    if all_data:
        return masses_list, radii_list, tidal_deform_list, data_list
    else: 
        return masses_list, radii_list, tidal_deform_list

def speed_of_sound_dimensional(gamma, k, mass_density):
    # similar to speed of sound above, but this has units of c = c
    dp_drho = gamma * k * mass_density**(gamma - 1)
    deps_drho = c**2 + (gamma * k / (gamma - 1)) * mass_density**(gamma-1)
    return dp_drho / deps_drho

def speed_vs_radius(ax, mass_densities, radii, gamma_values, density_cutoffs, k_values, cmap = None, norm = None):

    # find the speed of sound in each section of star
    # parametric EOS requires specific density values for each parameter gamma, k
    density_cutoffs = 10**density_cutoffs
    
    mask1 = mass_densities < density_cutoffs[0]
    mass_densities1 = mass_densities[mask1]
    
    mask2 = (mass_densities < density_cutoffs[1]) & (mass_densities >= density_cutoffs[0])
    mass_densities2 = mass_densities[mask2]
    
    mask3 = (mass_densities < density_cutoffs[2]) & (mass_densities >= density_cutoffs[1])
    mass_densities3 = mass_densities[mask3]
    
    mask4 = mass_densities >= density_cutoffs[2]
    mass_densities4 = mass_densities[mask4]

    
    norm_k = []
    for i in gamma_values:
        radius_, mass_, mass_density_, internal_energy_, pressure_, k_ = normalizing_values(i)
        norm_k.append(k_)
    norm_k = np.array(norm_k)

    
    c1 = speed_of_sound_dimensional(gamma_values[0], k_values[0] * norm_k[0], mass_densities1)
    c2 = speed_of_sound_dimensional(gamma_values[1], k_values[1] * norm_k[1], mass_densities2)
    c3 = speed_of_sound_dimensional(gamma_values[2], k_values[2] * norm_k[2], mass_densities3)
    c4 = speed_of_sound_dimensional(gamma_values[3], k_values[3] * norm_k[3], mass_densities4)
    
    ax.plot(radii[mask1] / 1e5, c1, c = cmap(norm(mass_densities[0])))
    ax.plot(radii[mask2] / 1e5, c2, c = cmap(norm(mass_densities[0])))
    ax.plot(radii[mask3] / 1e5, c3, c = cmap(norm(mass_densities[0])))
    ax.plot(radii[mask4] / 1e5, c4, c = cmap(norm(mass_densities[0])))
    ax.set_xlabel('Radius (km)')
    ax.set_ylabel('Sound velocity (c^2)')
    ax.set_title('Sanity Check - Sound velocity (fraction of c^2) in terms of radius -- Must be less than 1')

    return

def plot_tidal_deformability(masses, tidal_deform, central_mass_densities, cutoff_value = 2):
    # create plot, colorbar
    fig, ax = plt.subplots(1, figsize = (10,6))
    norm = plt.Normalize(vmin=min(central_mass_densities), vmax=max(central_mass_densities))
    cmap = cm.viridis

    # we trim certain stars off because of unrealistically small mass
    
    masses = np.array(masses[cutoff_value:])
    tidal_deform = np.array(tidal_deform[cutoff_value:])
    central_mass_densities = np.array(central_mass_densities[cutoff_value:])

    for index, i in enumerate(central_mass_densities):
        ax.scatter(masses[index], np.log10(tidal_deform[index]), color = cmap(norm(i)))

    plt.plot(masses, np.log10(tidal_deform), linestyle = '--', alpha = 0.2, color = 'black')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, orientation='vertical', label='Central Mass Density')

    ax.set_title('Mass vs Dimensionless Tidal Deformability Λ')
    ax.set_xlabel('Mass M☉')
    ax.set_ylabel('Tidal Deformability log(Λ)')
    ax.set_ylim(0, None)

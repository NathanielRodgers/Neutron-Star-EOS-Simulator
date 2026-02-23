import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm


# various constants necessary for making values dimensionless

c = 2.99792458 * 1e10
G = 6.6742 * 1e-8
Msun = 1.9891 * 1e33

# Making general length constants to shorten equations

L = G * Msun / c**2

# establishing normalizing values

def normalizing_values(gamma):
    radius_ = L
    mass_ = Msun
    mass_density_ = Msun / L**3
    internal_energy_ = (Msun * c**2) / L**3
    pressure_ = (Msun * c**2) / L**3
    k_ = ((Msun * c**2) / L**3) * (Msun / L**3)**(-gamma)
    return radius_, mass_, mass_density_, internal_energy_, pressure_, k_

def mass_density_function(pressure, k, gamma):
    mass_density = (pressure / k)**(1/gamma)
    return mass_density

def internal_energy_function(pressure, mass_density, gamma):
    internal_energy = pressure / ((gamma - 1) * mass_density)
    return internal_energy

def energy_density_function(pressure, mass_density, gamma):
    internal_energy = internal_energy_function(pressure, mass_density, gamma)
    energy_density = mass_density * (1 + internal_energy)
    return energy_density

def derivatives(radius, pressure, mass, k, gamma):
    mass_density = mass_density_function(pressure, k, gamma)
    energy_density = energy_density_function(pressure, mass_density, gamma)
    dphi = (mass + 4 * np.pi * radius**3 * pressure) / (radius * (radius - 2 * mass))
    dp = -(energy_density + pressure) * dphi
    dm = 4 * np.pi * radius**2 * energy_density
    return dphi, dp, dm


def RK4_step(radius, pressure, mass, k, gamma, dr, grav_potential_factor):

    phi1, p1, m1 = derivatives(radius, pressure, mass, k, gamma)
    phi2, p2, m2 = derivatives(radius + ((1/2) * dr), pressure + ((1/2) * p1 * dr), mass + ((1/2) * m1 * dr), k, gamma)
    phi3, p3, m3 = derivatives(radius + ((1/2) * dr), pressure + ((1/2) * p2 * dr), mass + ((1/2) * m2 * dr), k, gamma)
    phi4, p4, m4 = derivatives(radius + dr, pressure + (p3 * dr), mass + (m3 * dr), k, gamma)

    new_grav_potential_factor = grav_potential_factor + ((1/6) * (phi1 + 2*phi2 + 2*phi3 + phi4) * dr)
    new_mass = mass + ((1/6) * (m1 + 2*m2 + 2*m3 + m4) * dr)
    new_pressure = pressure + ((1/6) * (p1 + 2*p2 + 2*p3 + p4) * dr)
    new_mass_density = mass_density_function(new_pressure, k, gamma)

    return new_grav_potential_factor, new_mass, new_pressure, new_mass_density

def RK4_solver(central_mass_density, starting_radius, dr, gamma, k):

    radius_, mass_, mass_density_, internal_energy_, pressure_, k_ = normalizing_values(gamma)

    norm_k = k
    norm_dr = dr / radius_
    # central values (radius ~ 0, mass density, pressure, mass ~ 0)
    
    norm_starting_radius = starting_radius / radius_
    norm_central_mass_density = central_mass_density / mass_density_
    norm_central_pressure = norm_k * norm_central_mass_density ** gamma
    norm_central_energy_density = energy_density_function(norm_central_pressure, norm_central_mass_density, gamma)
    norm_central_mass = (4/3) * np.pi * norm_central_energy_density * norm_starting_radius**3

    # creating lists for each of the values, to be appended to
    pressures = [norm_central_pressure]
    radii = [norm_starting_radius]
    masses = [norm_central_mass]
    mass_densities = [norm_central_mass_density]
    metric_potentials = [0]

    
    while pressures[-1] > 1e-10:
        radius = radii[-1]
        mass = masses[-1]
        pressure = pressures[-1]
        grav_potential_factor = metric_potentials[-1]

        new_grav_potential_factor, new_mass, new_pressure, new_mass_density = RK4_step(radius, pressure, mass, norm_k, gamma, norm_dr, grav_potential_factor)

        
        if np.isnan(new_pressure):
            break
        
        
        metric_potentials.append(new_grav_potential_factor)
        masses.append(new_mass)
        pressures.append(new_pressure)
        mass_densities.append(new_mass_density)
        radii.append(radius + norm_dr)


    # turn into numpy arrays
    metric_potentials = np.array(metric_potentials)
    masses = np.array(masses)
    pressures = np.array(pressures)
    mass_densities = np.array(mass_densities)
    radii = np.array(radii)
    
    # add dimensions back in
    masses = masses * mass_
    pressures = pressures * pressure_
    mass_densities = mass_densities * mass_density_
    radii = radii * radius_

    # shift metric potentials to match exterior schwarzchild solution
    exterior_schw_sol = 0.5 * math.log(1 - (2*G*masses[-1] / (radii[-1] * c**2)))
    metric_potentials = metric_potentials + (exterior_schw_sol - metric_potentials[-1])

    return radii, masses, pressures, mass_densities, metric_potentials


def neutron_star(central_mass_densities, gamma, k, starting_radius = 1, dr = 50):
    
    radius_, mass_, mass_density_, internal_energy_, pressure_, k_ = normalizing_values(gamma)

    fig, ax = plt.subplots(2, figsize = (12,14))
    norm = plt.Normalize(vmin=min(central_mass_densities), vmax=max(central_mass_densities))
    cmap = cm.viridis

    masses_list = []
    radii_list = []
    for index, i in enumerate(central_mass_densities):
        radii, masses, pressures, mass_densities, metric_potentials = RK4_solver(i, starting_radius, dr, gamma, k)
        ax[0].plot(radii / 1e5, masses / Msun, color = cmap(norm(i)))
        ax[0].scatter(radii[-1] / 1e5, masses[-1] / Msun, color = cmap(norm(i)))
        masses_list.append(masses[-1] / Msun)
        radii_list.append(radii[-1] / 1e5)

        plot_sound_speed(ax[1], mass_densities, radii, gamma, k, cmap, norm)
    
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

    return radii, masses, pressures, mass_densities, metric_potentials


def speed_of_sound(gamma, k, mass_density):
    dp_drho = gamma * k * mass_density**(gamma - 1)
    deps_drho = c**2 + (1 / (gamma - 1)) * (gamma * k * mass_density**(gamma-1))
    return dp_drho / deps_drho

def plot_sound_speed(ax, mass_densities, radii, gamma, k, cmap, norm):
    radius_, mass_, mass_density_, internal_energy_, pressure_, k_ = normalizing_values(gamma)
    k = k * k_
    c = speed_of_sound(gamma, k, mass_densities)
    ax.plot(radii / 1e5, c, color = cmap(norm(mass_densities[0])))
    ax.set_xlabel('Radius (km)')
    ax.set_ylabel('Sound velocity (c^2)')
    ax.set_title('Sanity Check - Sound velocity (fraction of c^2) in terms of radius -- Must be less than 1')

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/7wTeFyQu)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=21989075)


README:

There is one jupyter notebook, three scripts, and two data tables included in this project.

Jupyter notebook: contains the linear-storytelling of this project, and uses each of the scripts and data tables to do so. Goes over the development of my EOS, the source of my parameters' values, and more.

Scripts:
polytropic.py: Polytropic EOS solver for TOV equations. Contains neutron_star(central_mass_densities, gamma, k, starting_radius = 1, dr = 50), which returns the maximum masses and radii of each star, and a graph of M/R and sound of speed v radius.

piecewise_polytropic.py: Piecewise polytropic EOS solver for TOV equations. Contains neutron_star(central_mass_densities, gamma_values, density_cutoffs = np.array([14.7, 14.9, 15]), starting_radius = 1, dr = 50, all_data = False). Returns the maximum masses and radii of each star and the tidal deformabilities, as well as the entire structure and thermodynamic quantities of each star if all_data=True. Also contains plot_tidal_deformability(masses, tidal_deformability, central_mass_densities), which plots the tidal deformabilities of each star over the masses.

tabular_analysis.py: Reads the .nb and .thermos eos files from SLy9 (CompOSE database), and plots best fit lines for logP/logRho to constrain EOS. Can be used for both polytropic and piecewise polytropic EOS, based on what density cutoffs you give (none, one+).

Data Tables (eos. files)
eos.nb - contains baryon number densities -- had to be converted from baryon number to cgs
eos.thermo - contains various thermodynamical quantities, each row is correlated to a row in eos.nb. Had to be converted from various metrics to cgs
These come from https://compose.obspm.fr/eos/86, and information on column headers is found on CompOSE's general information doc.

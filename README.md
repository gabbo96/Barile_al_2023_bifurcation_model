# 1-D numerical model of a river bifurcation
This repository contains the code of the numerical model used for the simulations discussed in Barile et al. (2023), submitted to Earth Surface Dynamics and currently under review (preprint available at https://egusphere.copernicus.org/preprints/2023/egusphere-2023-1551/).

Simulations can be performed by running the main script "bifoModel_Barile_al_2023.py", which imports and uses all functions contained in the auxiliary file "functions_bifoModel.py".
The input parameters can be set at the beginning of the main script, where comments explain the meaning of each parameter.
Every time a simulation is run, the code creates a subfolder for the simulation outputs inside the "output_folder", and then saves all plots in it. The code also saves the time series of the numerical values of many variables in individual .csv files.

The "output_folder" contains two examples of subfolders generated by the code for two simulations: one shows a bifurcation that reaches an unbalanced and fully active long-term state, while the other reaches a ``partial avulsion'' condition.

Please refer to the article for a detailed explanation of the notation and possible outcomes of numerical simulations.

Contact: Gabriele Barile, gabriele.barile@unitn.it
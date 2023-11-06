import os
import numpy as np
import matplotlib as mpl
from scipy import optimize as opt
from matplotlib import pyplot as plt
from functions_bifoModel import *

plt.style.use("science.mplstyle")

# ---------------------------------------------------------------------------- #
#                           INPUTS AND MODEL SETTINGS                          #
# ---------------------------------------------------------------------------- #

# Model main settings
# =(inlet step at t=0)/(inlet step at equilibrium computed via BRT)
inStepIC = 0.05
# if deltaQ exceeds this value during the simulation, the simulation
avulsion_threshold = 0.9
# stops and considers an avulsion to have happened

# I/O settings
output_folder_path = 'output_folder'
saveResults = True  # switch to create a folder for the simulation and save inputs, outputs, plots, and variables in there

# Plot parameters
numPlots = 40

# Numerical parameters
CFL = 0.9  # Courant-Friedrichs-Lewy parameter for timestep computation
dx = 25  # distance between grid points [m]
Mnode = 15  # number of nodes used to discretize the node cells in the g.v.f. equation
dt_set = 0  # if == 0, CFL condition is used
tend = 3000  # end of simulation expressed in n° of Exner timescales
# if ∆Q remains constant for nIterEq iterations, the simulation ends
nIterEq = int(5e2)
maxIter = int(1e6)  # max number of iterations during time evolution
tol = 1e-10  # Iterations tolerance

# Hydraulic and geometry parameters
# sediment transport formula. Available options: 'P78' (Parker78), 'MPM', 'P90' (Parker90), 'EH' (Engelund&Hansen)
TF = 'MPM'
L = 1000
beta_0 = 20
theta_0 = 0.07
d_s0 = 0.02  # relative roughness ds_a = d50/D_0
C_0 = 12
alpha_def = 9    # if ==0, alpha at equilibrium is computed according the procedure defined by Redolfi et al., 2019
"""
In all simulations shown in Barile et al., 2023 alpha has been set equal to 9, which corresponds to alpha=4.5
in the classic BRT model, where a full step (rather than the triangular step described in the paper) is employed.
"""

d50 = 0.02  # median grain size [m]
r = 0.5  # Ikeda parameter

# Physical constants
p = 0.6  # bed porosity
delta = 1.65  # relative density of the sediment
g = 9.81

# Initial guess for BRT solver.
BRT_init_guess = [1.2, 0.8, 0.9]  # Db, Dc, S/Sa

# ---------------------------------------------------------------------------- #
#              IC & BC DEFINITION AND BRT EQUILIBRIUM COMPUTATION              #
# ---------------------------------------------------------------------------- #
# Channel a IC (noted with subscript _a0)
D_0 = d50/d_s0
if C_0 == 0:
    C_0 = 6+2.5*np.log(D_0/(2.5*d50))
W_0 = beta_0*2*D_0
S_0 = theta_0*delta*d_s0
phi_a0, phi_Da, phi_Ta = phis_scalar(theta_0, TF, C_0)
Q_0 = uniFlowQ(W_0, S_0, D_0, g, C_0)
Qs_0 = W_0*np.sqrt(g*delta*d50**3)*phi_a0
Fr_a0 = Q_0/(W_0*D_0*np.sqrt(g*D_0))
M = int(L*D_0/dx)+1  # number of nodes

# Widths of the branches
W_b = W_0/2
W_c = W_0/2

# Compute βR and then equilibrium value for alpha assuming betaR=betaC (Redolfi et al., 2019)
betaR = betaR_MR(theta_0,  r, phi_Da, phi_Ta, C_0)
betaC = betaC_MR(theta_0, 1, r, phi_Da, phi_Ta, C_0)
alpha_MR = betaR/betaC
if alpha_def == 0:
    alpha = alpha_MR*2  # ! alpha is doubled to account for the triangular step effect
else:
    alpha = alpha_def

BRT_out = opt.fsolve(fSys_BRT, BRT_init_guess, (TF, theta_0, Q_0,
                     Qs_0, D_0, W_0, S_0, W_b, W_c, alpha/2, r, g, delta, d50, C_0))
Db_BRT, Dc_BRT = BRT_out[:2]*D_0
Sb_BRT = BRT_out[2]*S_0
Sc_BRT = Sb_BRT
Qb_BRT = uniFlowQ(W_b, Sb_BRT, Db_BRT, g, C_0)
Qc_BRT = uniFlowQ(W_c, Sb_BRT, Dc_BRT, g, C_0)
deltaQ_BRT = (Qb_BRT-Qc_BRT) / Q_0
inStep_BRT = -(Db_BRT-Dc_BRT) / D_0
theta_b_BRT = Sb_BRT*Db_BRT/(delta*d_s0)
theta_c_BRT = Sc_BRT*Dc_BRT/(delta*d_s0)

# Exner time computation
Tf = (1-p)*W_0*D_0/(Qs_0/W_0)

# Arrays initialization to IC
t = np.zeros(maxIter+1)
eta_avg = np.zeros(maxIter+1)
inStep = np.zeros(maxIter+1)
deltaQ = np.zeros(maxIter+1)
eta_a = np.zeros((maxIter+1, M))
eta_b = np.zeros((maxIter+1, M))
eta_c = np.zeros((maxIter+1, M))
D_a = np.ones((maxIter+1, M))*D_0
D_node = np.ones((maxIter+1, Mnode))*D_0
D_b = np.ones((maxIter+1, M))*D_0
D_c = np.ones((maxIter+1, M))*D_0
S_a = np.ones((maxIter+1, M-1))*S_0
S_b = np.ones((maxIter+1, M-1))*S_0
S_c = np.ones((maxIter+1, M-1))*S_0
Theta_a = np.ones((maxIter+1, M))*theta_0
Theta_b = np.ones((maxIter+1, M))*theta_0
Theta_c = np.ones((maxIter+1, M))*theta_0
Q_b = np.ones(maxIter+1)*Q_0/2
Q_c = np.ones(maxIter+1)*Q_0/2
Qs_a = np.ones((maxIter+1, M))*Qs_0
Qs_b = np.ones((maxIter+1, M))*Qs_0/2
Qs_c = np.ones((maxIter+1, M))*Qs_0/2
Qs_y = np.zeros(maxIter+1)
Hd_b = np.zeros(maxIter+1)
Hd_c = np.zeros(maxIter+1)
# longitudinal coordinates of nodes along channel a
xi = np.linspace(0, (M-1)*dx, M)

# Bed elevation and width IC. The bed elevation of the first cell of each branch is equal to the corresponding node cell
inStep[0] = inStepIC*inStep_BRT
if abs(inStep[0]) < 1e-3:
    inStep[0] = -d50
eta_a[0, :] = np.linspace(S_0*(alpha*W_0+(M-1)*dx), S_0*alpha*W_0, num=M)

# branches have the same initial slope and are shifted with respect to each other
eta_b[0, :] = np.linspace(inStep[0]*D_0/2, inStep[0]*D_0/2-S_0*(M-1)*dx, num=M)
eta_c[0, :] = np.linspace(-inStep[0]*D_0/2, -inStep[0]
                          * D_0/2-S_0*(M-1)*dx, num=M)

# Branches downstream BC: H(t)=H_0
# H_0 is set according to the normal depth in the downstream branches
H_0 = 0.5*(eta_b[0, -1]+eta_c[0, -1])+D_0
Hd_b[0] = H_0
Hd_c[0] = H_0
# ---------------------------------------------------------------------------- #
#               CREATE SIMULATION FOLDER AND SAVE INPUTS TO FILE               #
# ---------------------------------------------------------------------------- #
# Setup folder for current simulation.
simulation_name = f"{TF}_L{L:.0f}_beta{beta_0:.1f}_theta{theta_0:.3f}_ds{d_s0:.3f}_alphaeq{alpha:.2f}_dx{dx:.0f}"
if saveResults:
    simulation_folder_path = f'{output_folder_path}/{simulation_name}'
else:
    simulation_folder_path = f'{output_folder_path}/_temp'

# Create folder for simulation. WARNING: if the folder already exists, all files will be overwritten
os.makedirs(simulation_folder_path, exist_ok=True)

# Compose strings containing the input parameters, the initial conditions and the equilibrium solutions.
# Then print them and save them in the simulation folder.
str_inputs = f"Initial inlet step = {inStep[0]:.4f}\
        \n\nINPUT PARAMETERS:\
        \nSolid transport formula = {TF}\n\
        \nM = {M}\ndx = {dx} m\nCFL = {CFL}\nt_end = {tend} Tf\nL = {L}\nbeta_0 = {beta_0}\ntheta_0 = {theta_0}\
        \nd_s0 = {d_s0}\nd50 = {d50} m\nalpha_eq = {alpha:.2f}\nr = {r}\n\
        \nRESONANT ASPECT RATIO AND EQUILIBRIUM alpha ACCORDING TO REDOLFI ET AL., 2019\
        \nbeta_R = {betaR:.2f}\nalpha_MR = {alpha_MR:.2f}\n(beta_a-beta_R)/beta_R = {(beta_0-betaR)/betaR:.2f}\n\
        \nMAIN CHANNEL IC:\
        \nW_a = {W_0} m\nS_a = {S_0:.1e}\nD_0 = {D_0} m\nFr_a = {Fr_a0:.2f}\nL = {L*D_0} m\nL/W_a = {L*D_0/W_0:.1f}\
        \nQ_a = {Q_0:.2f} m^3 s^-1\nQs_a = {Qs_0:.2e} m^3 s^-1\nExner time: Tf = {Tf/3600:.2f} h\
        \n\nBRT EQUILIBRIUM SOLUTION:\
        \nDelta_Q = {deltaQ_BRT:.3f}\nDelta_eta = {inStep_BRT:.3f}\nS/S_a = {Sb_BRT/S_0:.3f}\n\n"

print(str_inputs)
with open(os.path.join(simulation_folder_path, '_inputs.txt'), 'w') as f:
    f.writelines(str_inputs)

# ---------------------------------------------------------------------------- #
#                             TIME EVOLUTION                                #
# ---------------------------------------------------------------------------- #
eqIndex = 0  # time index at which system reaches equilibrium
avulsionFlag = False
nanFlag = False
for n in range(0, maxIter):
    # ------------- STABILITY CONDITION AND END-OF-SIMULATION-CHECKS ------------- #

    # Compute dt according to CFL condition and update time. Check if system has reached equilibrium
    if dt_set == 0:
        # Compute propagation celerity along the three branches
        Ceta_a = C_eta(Q_0, W_0, D_a[n, :], g, delta, d50, p, C_0, TF)
        Ceta_b = C_eta(Q_b[n], W_b, D_b[n, :], g, delta, d50, p, C_0, TF)
        Ceta_c = C_eta(Q_c[n], W_c, D_c[n, :], g, delta, d50, p, C_0, TF)
        Cmax = max(max(Ceta_a), max(Ceta_b), max(Ceta_c))
        dt = CFL*dx/Cmax
    else:
        dt = dt_set
    t[n+1] = t[n]+dt

    # Check for avulsion
    if abs(deltaQ[n]) > avulsion_threshold:
        print("\nAn avulsion occured\n")
        avulsionFlag = True
        eqIndex = n
        break

    # Check if equilibrium or end of simulation time have been reached
    if eqIndex == 0 and n > nIterEq:  # check for equilibrium
        if np.all(abs((deltaQ[n-1-nIterEq:n+1]+1e-10)/(deltaQ[n]+1e-10)-1) < tol):
            eqIndex = n
            print('\nEquilibrium reached\n')
            break

    if t[n+1] >= (tend*Tf):
        eqIndex = n
        print('\nEnd time reached\n')
        break

    # Check for nans: if there is any, end the simulation
    if np.any(np.isnan(D_a[n, :])) or np.any(np.isnan(D_b[n, :])) or np.any(np.isnan(D_c[n, :])):
        print("\nNan values found in water depth arrays\n")
        nanFlag = True
        eqIndex = n
        break

    # ----------- WATER DISCHARGE PARTITIONING AND PROFILES COMPUTATION ---------- #

    # Impose the downstream BC
    Hd_b[n+1] = H_0
    Hd_c[n+1] = H_0
    D_b[n+1, -1] = Hd_b[n+1]-eta_b[n, -1]
    D_c[n+1, -1] = Hd_c[n+1]-eta_c[n, -1]

    # Compute the discharge partitioning at the node by coupling the computation of the water-surface profiles
    # with the nodal boundary condition of constant water level in transverse direction
    D_b[n+1, :] = buildProfile_rk4(D_b[n+1, -1],
                                   Q_b[n], W_b, S_b[n, :], dx, g, C_0)
    D_c[n+1, :] = buildProfile_rk4(D_c[n+1, -1],
                                   Q_c[n], W_c, S_c[n, :], dx, g, C_0)
    if abs((D_b[n+1, 0]-D_c[n+1, 0]+inStep[n]*D_0)/(0.5*(D_b[n+1, 0]+D_c[n+1, 0]))) > np.sqrt(tol):
        Q_b[n+1] = opt.fsolve(fSys_rk4, Q_b[n], (D_b[n+1, -1], D_c[n+1, -1],
                              D_0, inStep[n], Q_0, W_b, W_c, S_b[n, :], S_c[n, :], dx, g, C_0))[0]
        Q_c[n+1] = Q_0-Q_b[n+1]

        D_b[n+1, :] = buildProfile_rk4(D_b[n+1, -1],
                                       Q_b[n+1], W_b, S_b[n, :], dx, g, C_0)
        D_c[n+1, :] = buildProfile_rk4(D_c[n+1, -1],
                                       Q_c[n+1], W_c, S_c[n, :], dx, g, C_0)
    else:
        Q_b[n+1] = Q_b[n]
        Q_c[n+1] = Q_c[n]

    # Compute water-surface profile along the node cells
    D_node[n+1, -1] = 0.5*(D_b[n+1, 0]+D_c[n+1, 0])
    D_node[n+1, :] = buildTriStepProfile(D_node[n+1, -1], eta_a[n, -1],
                                         eta_b[n, 0], eta_c[n, 0], alpha, W_0, Q_0, C_0, g, M=Mnode)

    # Compute water-surface profile along channel a
    D_a[n+1, -1] = D_node[n+1, 0]
    D_a[n+1, :] = buildProfile_rk4(D_a[n+1, -1],
                                   Q_0, W_0, S_a[n, :], dx, g, C_0)

    # ----------------------- SOLID DISCHARGES COMPUTATION ----------------------- #

    # Update shields and Qs update for channels b and c
    Theta_a[n+1, :] = shieldsUpdate(Q_0, W_0, D_a[n+1, :], d50, g, delta, C_0)
    Theta_b[n+1, :] = shieldsUpdate(Q_b[n+1],
                                    W_b, D_b[n+1, :], d50, g, delta, C_0)
    Theta_c[n+1, :] = shieldsUpdate(Q_c[n+1],
                                    W_c, D_c[n+1, :], d50, g, delta, C_0)
    Qs_a[n+1, :] = W_0*np.sqrt(g*delta*d50**3) * \
        phis(Theta_a[n+1, :], TF, C_0)[0]
    Qs_b[n+1, :] = W_b*np.sqrt(g*delta*d50**3) * \
        phis(Theta_b[n+1, :], TF, C_0)[0]
    Qs_c[n+1, :] = W_c*np.sqrt(g*delta*d50**3) * \
        phis(Theta_c[n+1, :], TF, C_0)[0]

    # Compute liquid and solid discharge in transverse direction at the node
    Q_y = 0.5*(Q_b[n+1]-Q_c[n+1])
    Qs_y[n+1] = Qs_a[n+1, -1]*(Q_y/Q_0-alpha*r/np.sqrt(Theta_a[n+1, -1])
                               * (eta_b[n, 0]-eta_c[n, 0])/(0.5*(W_0+W_b+W_c)))

    # ------------------- EXNER EQUATION INTEGRATIONS OVER TIME ------------------ #

    # Solve Exner equation along channel a
    eta_a[n+1, 0] = eta_a[n, 0]-dt/dx*(Qs_a[n+1, 0]-Qs_0)/(W_0*(1-p))
    eta_a[n+1, 1:] = eta_a[n, 1:]-dt/dx * \
        (Qs_a[n+1, 1:]-Qs_a[n+1, :-1])/(W_0*(1-p))

    # Solve Exner equation along the node cells (using the triangular step formulation) to update their bed elevations
    eta_b[n+1, 0] = eta_b[n, 0]+eta_a[n, -1]-eta_a[n+1, -1]-4*dt / \
        ((1-p)*alpha*W_0*0.5*(W_0+W_b+W_c)) * \
        (Qs_b[n+1, 0]-Qs_y[n+1]-0.5*Qs_a[n+1, -1])
    eta_c[n+1, 0] = eta_c[n, 0]+eta_a[n, -1]-eta_a[n+1, -1]-4*dt / \
        ((1-p)*alpha*W_0*0.5*(W_0+W_b+W_c)) * \
        (Qs_c[n+1, 0]+Qs_y[n+1]-0.5*Qs_a[n+1, -1])

    # Solve Exner equation along channels b and c to update bed elevation according to an upwind scheme (Fr<1, so bed level perturbations travel downstream)
    eta_b[n+1, 1:] = eta_b[n, 1:]-dt/dx * \
        (Qs_b[n+1, 1:]-Qs_b[n+1, :-1])/(W_b*(1-p))
    eta_c[n+1, 1:] = eta_c[n, 1:]-dt/dx * \
        (Qs_c[n+1, 1:]-Qs_c[n+1, :-1])/(W_c*(1-p))

    # Update bed slopes
    S_a[n+1, :] = (eta_a[n+1, :-1]-eta_a[n+1, 1:])/dx
    S_b[n+1, :] = (eta_b[n+1, :-1]-eta_b[n+1, 1:])/dx
    S_c[n+1, :] = (eta_c[n+1, :-1]-eta_c[n+1, 1:])/dx

    # Update average+asymmetry indicators
    deltaQ[n+1] = (Q_b[n+1]-Q_c[n+1])/Q_0
    inStep[n+1] = (eta_b[n+1, 0]-eta_c[n+1, 0])/D_0
    eta_avg[n+1] = 0.5*(eta_b[n+1, 0]+eta_c[n+1, 0])/D_0

    # Time print
    if n % 500 == 0:
        print(f"Elapsed time = {t[n+1]/Tf:.1f} Tf, ∆Q = {deltaQ[n+1]:.3f}")

# ---------------------------------------------------------------------------- #
#                                 PRINT OUTPUTS                                #
# ---------------------------------------------------------------------------- #
# Print final ∆Q and compare it with that computed through BRT model
if abs(deltaQ_BRT) > 1e-3 and abs(inStep_BRT) > 1e-3:
    deltaQ_BRT_err = (abs(deltaQ[n])-deltaQ_BRT)/deltaQ_BRT
    inStep_BRT_err = (inStep[n]-inStep_BRT)/inStep_BRT
else:
    deltaQ_BRT_err = 0
    inStep_BRT_err = 0

str_outputs = f"-----------------------------------------\
    \nEQUILIBRIUM STATE AND COMPARISON WITH BRT\
    \n-----------------------------------------\
    \nFinal Delta_Q = {deltaQ[n]:.4f}\
    \nDelta_Q at equilibrium according to BRT = {deltaQ_BRT:.4f}\
    \nDelta_Q relative difference = {deltaQ_BRT_err:.1%}\
    \n\nFinal Delta_eta = {inStep[n]:.4f}\
    \nDelta_eta at equilibrium according to BRT = {inStep_BRT:.4f}\
    \nDelta_eta relative difference = {inStep_BRT_err:.1%}\
    \nAverage node cell elevation = {eta_avg[n]:.4f}"

# bed slope and water depth at equilibrium in the downstream branches
S_b_avg = np.mean(S_b[n, 1:])
S_c_avg = np.mean(S_c[n, 1:])
D_b_avg = np.mean(D_b[n, 1:])
D_c_avg = np.mean(D_c[n, 1:])
D_b_uf = uniFlowD(Q_b[n], W_b, S_b[n, 1], g, C_0)
D_c_uf = uniFlowD(Q_c[n], W_c, S_c[n, 1], g, C_0)
str_outputs += f"\
    \n\nAverage bed slopes: S_b/S_0 = {S_b_avg/S_0:.3f}, S_c/S_0 = {S_c_avg/S_0:.3f}\
    \nAverage slope according to BRT: S/S_0 = {Sb_BRT/S_0:.3f}\
    \n\nAverage water depths: D_b/D_0 = {D_b_avg/D_0:.3f}, D_c/D_0 = {D_c_avg/D_0:.3f}\
    \nWater depths at inlets/normal depths: D_b_inlet/D_b_uf = {D_b[n,1]/D_b_uf:.3f}, D_c_inlet/D_c_uf = {D_c[n,1]/D_c_uf:.3f}\n"

# avulsion occurrence
str_outputs += f"\n-------------\
    \nOTHER OUTPUTS\
    \n-------------\
    \nFull avulsion occurred (0:no, 1:yes): {avulsionFlag:d}"

# longitudinal step
if eta_c[n, 0] < eta_c[n, 1] or eta_b[n, 0] < eta_b[n, 1]:
    longStep = (D_b[n, 1]-D_c[n, 1])/D_0
    str_outputs += f"\nLongitudinal step: (D_b-D_c)/D_0 = {longStep:.3f}"
else:
    longStep = 0
    str_outputs += "\nNo longitudinal step"

# Compute average water depth of the branches and the bedslope
S_node = (eta_a[n, -1]-0.5*(eta_b[n, 0]+eta_c[n, 0]))/(alpha*W_0)
str_outputs += f"\nAverage water depth at the inlets of the branches:D_avg/D_0 = {D_node[n,-1]/D_0:.3f}\
    \nBed slope of the node cells: S_node/S_0 = {S_node/S_0:.3f}\n"
print(str_outputs)

# Save outputs in the simulation folder
with open(os.path.join(simulation_folder_path, '_outputs.txt'), 'w') as f:
    f.writelines(str_outputs)

# ---------------------------------------------------------------------------- #
#                       SAVE INPUTS AND OUTPUTS TO FILES                       #
# ---------------------------------------------------------------------------- #

# Save outputs and relevant variables in a subfolder called "variables" inside
# the simulation folder
os.makedirs(os.path.join(simulation_folder_path, "variables"), exist_ok=True)
np.savetxt(os.path.join(simulation_folder_path,
           "variables", "timesteps.csv"), t[:n+1]/Tf)
np.savetxt(os.path.join(simulation_folder_path,
           "variables", "deltaQ_time.csv"), deltaQ[:n+1])
np.savetxt(os.path.join(simulation_folder_path,
           "variables", "eta_avg_time.csv"), eta_avg[:n+1])
np.savetxt(os.path.join(simulation_folder_path,
           "variables", "inletStep_time.csv"), inStep[:n+1])
np.savetxt(os.path.join(simulation_folder_path,
           "variables", "D_a_final.csv"), D_a[n, :])
np.savetxt(os.path.join(simulation_folder_path,
           "variables", "D_b_final.csv"), D_b[n, :])
np.savetxt(os.path.join(simulation_folder_path,
           "variables", "D_c_final.csv"), D_c[n, :])
np.savetxt(os.path.join(simulation_folder_path,
           "variables", "S_a_final.csv"), S_a[n, :])
np.savetxt(os.path.join(simulation_folder_path,
           "variables", "S_b_final.csv"), S_b[n, :])
np.savetxt(os.path.join(simulation_folder_path,
           "variables", "S_c_final.csv"), S_c[n, :])
np.savetxt(os.path.join(simulation_folder_path, "variables",
           "S_b_avg_time.csv"), np.mean(S_b[:n+1, 1:]/S_0, axis=1))
np.savetxt(os.path.join(simulation_folder_path, "variables",
           "S_c_avg_time.csv"), np.mean(S_c[:n+1, 1:]/S_0, axis=1))
np.savetxt(os.path.join(simulation_folder_path,
           "variables", "Theta_a_final.csv"), Theta_a[n, :])
np.savetxt(os.path.join(simulation_folder_path,
           "variables", "Theta_b_final.csv"), Theta_b[n, :])
np.savetxt(os.path.join(simulation_folder_path,
           "variables", "Theta_c_final.csv"), Theta_c[n, :])

# ---------------------------------------------------------------------------- #
#                                     PLOTS                                    #
# ---------------------------------------------------------------------------- #

# -------------- ∆Q, ∆η and sediment discharges trends over time ------------- #
# Plot ∆Q evolution over time along with exponential fittings
plt.figure()
plt.plot(t[1:n+1]/Tf, deltaQ[1:n+1]/deltaQ_BRT)
plt.title("Discharge asymmetry vs time")
plt.xlabel(r"$t/T_F$ [-]")
plt.ylabel(r"$\Delta Q/\Delta Q_{BRT}$ [-]")
plt.grid()
plt.savefig(os.path.join(simulation_folder_path, "deltaQtime.pdf"))

# Plot ∆η evolution over time along with exponential fittings
plt.figure()
plt.plot(t[1:n+1]/Tf, inStep[1:n+1]/inStep_BRT)
plt.title('Inlet step vs time')
plt.xlabel(r"$t/T_F$ [-]")
plt.ylabel(r'$\Delta \eta/\Delta \eta_{BRT}$ [-]')
plt.grid()
plt.savefig(os.path.join(simulation_folder_path, "inSteptime.pdf"))

# Plot evolution of node cells over time
plt.figure()
plt.plot(t[:n+1]/Tf, (eta_b[:n+1, 0]-eta_b[0, 0])/D_0, label='Node cell B')
plt.plot(t[:n+1]/Tf, (eta_c[:n+1, 0]-eta_c[0, 0])/D_0, label='Node cell C')
plt.title('Evolution of node cells elevation over time')
plt.xlabel(r"$t/T_F$ [-]")
plt.ylabel(r'$(\eta-\eta_0)/D_0$ [-]')
plt.grid()
plt.savefig(os.path.join(simulation_folder_path, "nodeCells.pdf"))

# Plot bed evolution at relevant cross-sections (upstream, middle, downstream)
fig, axs = plt.subplots(1, 3)
lw = 2.5  # line width of subplots
axs[0].plot(t[:n+1]/Tf, (eta_b[:n+1,       1]-eta_b[0,      1]) /
            D_0, label='Branch B', linewidth=lw)
axs[0].plot(t[:n+1]/Tf, (eta_c[:n+1,       1]-eta_c[0,      1]) /
            D_0, label='Branch C', linewidth=lw)
axs[1].plot(t[:n+1]/Tf, (eta_b[:n+1, int(M/2)]-eta_b[0, int(M/2)]) /
            D_0, label='Branch B', linewidth=lw)
axs[1].plot(t[:n+1]/Tf, (eta_c[:n+1, int(M/2)]-eta_c[0, int(M/2)]) /
            D_0, label='Branch C', linewidth=lw)
axs[2].plot(t[:n+1]/Tf, (eta_b[:n+1, -      1]-eta_b[0,     -1]) /
            D_0, label='Branch B', linewidth=lw)
axs[2].plot(t[:n+1]/Tf, (eta_c[:n+1, -      1]-eta_c[0,     -1]) /
            D_0, label='Branch C', linewidth=lw)

for (ax, title) in zip(axs.flat, ['upstream', 'middle', 'downstream']):
    ax.set_title(title)
    ax.set_xlabel(r'$t/T_F$ [-]')
    ax.legend()
    ax.grid()
axs[0].set_ylabel(r'$(\eta-\eta_0)/D_0$ [-]')
fig.suptitle("Scaled bed elevation over time")
plt.tight_layout()
plt.savefig(os.path.join(simulation_folder_path,
            "branchesBedEvolution_sections.pdf"))

# Plot evolution of average bed slopes over time along branches B and C
plt.figure()
plt.plot(t[:n+1]/Tf, np.mean(S_b[:n+1, 1:]/S_0, axis=1),
         label='Branch B average slope')
plt.plot(t[:n+1]/Tf, np.mean(S_c[:n+1, 1:]/S_0, axis=1),
         label='Branch C average slope')
plt.title('Branches slopes vs time')
plt.xlabel(r"$t/T_F$ [-]")
plt.ylabel(r'$S/S_{a0}$ [-]')
plt.legend()
plt.grid()
plt.savefig(os.path.join(simulation_folder_path, "meanSlopes.pdf"))

# Plot also the ratio between the average slopes
plt.figure()
plt.plot(t[:n+1]/Tf, np.mean(S_b[:n+1, 1:], axis=1) /
         np.mean(S_c[:n+1, 1:], axis=1))
plt.title('Average slope ratio vs time')
plt.xlabel(r"$t/T_F$ [-]")
plt.ylabel(r'$S_b/S_c$ [-]')
plt.grid()
plt.savefig(os.path.join(simulation_folder_path, "slopeRatio.pdf"))

# Plot trends of relevant solid discharges over time
plt.figure()
plt.plot(t[:n+1]/Tf, Qs_a[:n+1, -1] / Qs_0, label=r'$Q_{sa}^{OUT}$')
plt.plot(t[:n+1]/Tf, (Qs_b[:n+1, 0]+Qs_c[:n+1, 0]) /
         Qs_0, label=r'$Q_{sb}^{IN}+Q_{sc}^{IN}$')
plt.title('Solid discharge vs time')
plt.xlabel(r"$t/T_F$ [-]")
plt.ylabel(r'$Q_s/Q_{s0}$ [-]')
plt.legend()
plt.grid()
plt.savefig(os.path.join(simulation_folder_path, "Qsa_QsbQsc.pdf"))

plt.figure()
plt.plot(t[:n+1]/Tf, Qs_y[:n+1]/Qs_0, label=r'$Q_{sy}$')
plt.plot(t[:n+1]/Tf, Qs_b[:n+1, 0]/Qs_0, label=r'$Q_{sb}^{IN} $')
plt.plot(t[:n+1]/Tf, Qs_b[:n+1, -1]/Qs_0, label=r'$Q_{sb}^{OUT}$')
plt.title('Solid discharge vs time')
plt.xlabel(r"$t/T_F$ [-]")
plt.ylabel(r'$Q_s/Q_{s0}$ [-]')
plt.legend()
plt.grid()
plt.savefig(os.path.join(simulation_folder_path, "Qsb_Qsy.pdf"))

plt.figure()
plt.plot(t[:n+1]/Tf, Qs_y[:n+1]/Qs_0, label=r'$Q_{sy}$')
plt.plot(t[:n+1]/Tf, Qs_c[:n+1, 0]/Qs_0, label=r'$Q_{sc}^{IN} $')
plt.plot(t[:n+1]/Tf, Qs_c[:n+1, -1]/Qs_0, label=r'$Q_{sc}^{OUT}$')
plt.title('Solid discharge vs time')
plt.xlabel(r"$t/T_F$ [-]")
plt.ylabel(r'$Q_s/Q_{s0}$ [-]')
plt.legend()
plt.grid()
plt.savefig(os.path.join(simulation_folder_path, "Qsc_Qsy.pdf"))

# ------------- Evolution of longitudinal bed profiles over time ------------- #
bed_folder_path = os.path.join(
    simulation_folder_path, "longitudinal_bed_profiles")
os.makedirs(bed_folder_path, exist_ok=True)

# Plot bed elevation profiles scaled with reference plane
plt.figure()
current_fig_number = plt.gcf().number

plotTimeIndexes = np.linspace(0, n-1, numPlots).astype(int)
crange = np.linspace(0, 1, numPlots)
bed_colors = plt.cm.viridis(crange)
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=t[n]/Tf)
for plotTimeIndex, color in zip(plotTimeIndexes, bed_colors):
    plt.figure(current_fig_number)
    plt.plot(xi/W_0, (eta_a[plotTimeIndex, :]-eta_a[0, :])/D_0, color=color)
    plt.figure(current_fig_number+1)
    plt.plot(xi/W_0, (eta_b[plotTimeIndex, :]-eta_b[0, :])/D_0, color=color)
    plt.figure(current_fig_number+2)
    plt.plot(xi/W_0, (eta_c[plotTimeIndex, :]-eta_c[0, :])/D_0, color=color)

for i, branch in enumerate(['A', 'B', 'C']):
    plt.figure(current_fig_number+i)
    plt.title(f'Branch {branch} bed evolution over time')
    plt.xlabel(r'$x/W_a$ [-]')
    plt.ylabel(r'$(\eta-\eta_0)/D_0$ [-]')
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), orientation='vertical')\
        .set_label(label=r"$t/T_F$ [-]")
    plt.grid()
    plt.savefig(os.path.join(bed_folder_path,
                f'branch{branch}_bed_scaled.pdf'))

# Plot non-scaled (dimensional) bed elevation profiles
plt.figure()
current_fig_number = plt.gcf().number
bed_profiles_out = np.zeros([3, M, numPlots])
for i, (plotTimeIndex, color) in enumerate(zip(plotTimeIndexes, bed_colors)):
    bed_profiles_out[:, :, i] = np.vstack(
        [eta_a[plotTimeIndex, :], eta_b[plotTimeIndex, :], eta_c[plotTimeIndex, :]])
    plt.figure(current_fig_number)
    plt.plot(xi, eta_a[plotTimeIndex, :], color=color)
    plt.figure(current_fig_number+1)
    plt.plot(xi, eta_b[plotTimeIndex, :], color=color)
    plt.figure(current_fig_number+2)
    plt.plot(xi, eta_c[plotTimeIndex, :], color=color)

for i, branch in enumerate(['A', 'B', 'C']):
    plt.figure(current_fig_number+i)
    plt.title(f'Branch {branch} bed evolution over time')
    plt.xlabel(r'$x$ [m]')
    plt.ylabel(r'$\eta$ [m]')
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), orientation='vertical')\
        .set_label(label=r"$t/T_F$ [-]")
    plt.grid()
    plt.savefig(os.path.join(bed_folder_path, f'branch{branch}_bed.pdf'))

    # Save to text file the bed elevation profiles arrays displayed in the plot
    np.savetxt(os.path.join(simulation_folder_path, "variables",
                            f"bed_profiles_branch{branch}.csv"), bed_profiles_out[i, :, :])

# Plot bed elevation profiles of the two downstream branches in the same figure
plt.figure()
plotTimeIndexes = np.linspace(0, n-1, numPlots)
crange = np.linspace(0.2, 0.85, numPlots)
bed_colors = [plt.cm.Blues(crange), plt.cm.Oranges(crange)]
norm = mpl.colors.Normalize(vmin=0, vmax=t[n]/Tf)
for plotTimeIndex, color_b, color_c in zip(plotTimeIndexes.astype(int), *bed_colors):
    if plotTimeIndex == plotTimeIndexes[-1]:  # plot with labels
        plt.plot(xi, eta_b[plotTimeIndex, :],
                 label='Dominant branch', color=color_b)
        plt.plot(xi, eta_c[plotTimeIndex, :],
                 label='Non-dominant branch', color=color_c)
    else:  # no labels
        plt.plot(xi, eta_b[plotTimeIndex, :], color=color_b)
        plt.plot(xi, eta_c[plotTimeIndex, :], color=color_c)

plt.plot(xi, eta_b[0, :], linestyle='dashed',
         color='black', label='Initial conditions', lw=2.5)
plt.title('Time evolution of longitudinal bed elevation profiles')
plt.xlabel(r"$x$ [m]")
plt.ylabel(r'$\eta$ [m]')
plt.legend()
plt.grid()
plt.savefig(os.path.join(bed_folder_path, "bedProfiles_B_and_C.pdf"))

# ------------ Plot final bed elevation and water-surface profiles ----------- #
# Set colors for bed and water-surface elevation
bed_color = 'goldenrod'
water_color = 'royalblue'

# Generate arrays: consider n-1 timestep to avoid printing nan values in water depth
x_a = xi
x_n = np.linspace(x_a[-1], x_a[-1]+alpha*W_0, Mnode)
x_bc = np.linspace(x_n[-1], x_n[-1]+L*D_0, M)
eta_ab = np.linspace(eta_a[n-1, -1], eta_b[n-1, 0], Mnode)
eta_ac = np.linspace(eta_a[n-1, -1], eta_c[n-1, 0], Mnode)
H_a = eta_a[n-1, :]+D_a[n-1, :]
H_b = eta_b[n-1, :]+D_b[n-1, :]
H_c = eta_c[n-1, :]+D_c[n-1, :]
H_n = D_node[n-1, :]+0.5*(eta_ab+eta_ac)

# Plot bed elevation profiles
plt.plot(x_a, eta_a[n-1, :], color=bed_color, label=r'Bed elevation $\eta$')
plt.plot(x_n, eta_ab,   color=bed_color, linestyle='dashed')
plt.plot(x_n, eta_ac,   color=bed_color, linestyle='dotted')
plt.plot(x_bc, eta_b[n-1, :], color=bed_color, linestyle='dashed')
plt.plot(x_bc, eta_c[n-1, :], color=bed_color, linestyle='dotted')

# Plot water-surface profiles
plt.plot(x_a, H_a, color=water_color, label=r'Water surface $H$')
plt.plot(x_n, H_n, color=water_color)
plt.plot(x_bc, H_b, color=water_color, linestyle='dashed')
plt.plot(x_bc, H_c, color=water_color, linestyle='dotted')
plt.title('Final configuration')
plt.xlabel(r"$x$ [m]")
plt.ylabel(r'$H,\eta$ [m]')
plt.legend()
plt.grid()
plt.savefig(f'{bed_folder_path}/finalProfiles.pdf')

# Redo plots, but scaled with S0
plt.figure()
plt.plot(x_a, eta_a[n-1, :]-(eta_a[n-1, 0]-S_0*x_a),
         color=bed_color, label=r'Bed elevation $\eta$')
plt.plot(x_n, eta_ab - (eta_a[n-1, 0]-S_0*x_n),
         color=bed_color, linestyle='dashed')
plt.plot(x_n, eta_ac - (eta_a[n-1, 0]-S_0*x_n),
         color=bed_color, linestyle='dotted')
plt.plot(x_bc, eta_b[n-1, :]-(eta_a[n-1, 0]-S_0*x_bc),
         color=bed_color, linestyle='dashed')
plt.plot(x_bc, eta_c[n-1, :]-(eta_a[n-1, 0]-S_0*x_bc),
         color=bed_color, linestyle='dotted')

# Plot water-surface profiles
plt.plot(x_a, H_a-(eta_a[n-1, 0]-S_0*x_a),
         color=water_color, label=r'Water surface $H$')
plt.plot(x_n, H_n-(eta_a[n-1, 0]-S_0*x_n), color=water_color)
plt.plot(x_bc, H_b-(eta_a[n-1, 0]-S_0*x_bc),
         color=water_color, linestyle='dashed')
plt.plot(x_bc, H_c-(eta_a[n-1, 0]-S_0*x_bc),
         color=water_color, linestyle='dotted')
plt.title('Final configuration relative to reference plane')
plt.xlabel(r"$x$ [m]")
plt.ylabel(r'$H-H_0,\eta-\eta_0$ [m]')
plt.legend()
plt.grid()
plt.savefig(f'{bed_folder_path}/finalProfiles_scaled.pdf')

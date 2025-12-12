import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from src.model import params_vaccination, Athresh_val, z_init_vaccination, Athresh_event, model_vaccination
import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams.update({'font.size':15})
matplotlib.rcParams.update({'font.size':15})
matplotlib.rcParams['xtick.major.size'] = 7
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['xtick.minor.size'] = 5
matplotlib.rcParams['xtick.minor.width'] = 3
matplotlib.rcParams['ytick.major.size'] = 7
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['ytick.minor.size'] = 5
matplotlib.rcParams['ytick.minor.width'] = 3
matplotlib.rcParams['axes.linewidth'] = 3

z0 = z_init_vaccination()
L0 = z0[0]
second_vaccination_time = 21
A_titer = Athresh_val(params_vaccination())

def solve_ode(params):
    sol1 = integrate.solve_ivp(model_vaccination, [0, second_vaccination_time], z0, args = ([params]), method = "BDF", dense_output = True)
    second_vacc_z = sol1.y[:,-1].copy()
    second_vacc_z[0] += L0
    sol2 = integrate.solve_ivp(model_vaccination, [second_vaccination_time,250], second_vacc_z, args = ([params]), method = "BDF", dense_output = True, events = [Athresh_event])
    t_simvals = np.concatenate((sol1.t, sol2.t))
    A_simvals = params["Abcrmin"]*np.concatenate((sol1.y[12], sol2.y[12])) + params["Abcrmax"]*np.concatenate((sol1.y[13], sol2.y[13]))
    
    return t_simvals, A_simvals

params_a = params_vaccination()

params_b = params_vaccination()
params_b["kAPC"] = params_b["kAPC"]*0.5

params_c = params_vaccination()
params_c["kAPC"] = params_c["kAPC"]*0.1

params_d = params_vaccination()
params_d["kAPC"] = params_d["kAPC"]*0

t_simvals_a, A_simvals_a = solve_ode(params_a)
t_simvals_b, A_simvals_b = solve_ode(params_b)
t_simvals_c, A_simvals_c = solve_ode(params_c)
t_simvals_d, A_simvals_d = solve_ode(params_d)

fig1, ax1 = plt.subplots(figsize = (8.5,4))
ax1.plot(t_simvals_a, A_simvals_a, label = "$k_{APC}/k_P = 1$", lw = 3, color = "black")
ax1.plot(t_simvals_b, A_simvals_b, label = "$k_{APC}/k_P = 0.5$", lw = 3, color = "blue")
ax1.plot(t_simvals_c, A_simvals_c, label = "$k_{APC}/k_P = 0.1$", lw = 3, color = "green")
ax1.plot(t_simvals_d, A_simvals_d, label = "$k_{APC}/k_P = 0$", lw = 3, color = "red")
ax1.set_xlabel("Days after 1st vaccination")
ax1.set_ylabel("$T$ (ng/mL)")
ax1.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
ax1.set_yticks(np.arange(0, 15e5, 3.5e5))
ax1.axhline(A_titer, color = "black", ls = "dotted", lw = 3)
ax1.axhline(2*A_titer, color = "black", ls = "dashed", lw = 3)
ax1.legend()
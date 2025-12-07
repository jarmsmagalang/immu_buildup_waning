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
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches

z0 = z_init_vaccination()
L0 = z0[0]
second_vaccination_time = 21

params_a = params_vaccination()
A_titer = Athresh_val(params_a)

sol1_a = integrate.solve_ivp(model_vaccination, [0, second_vaccination_time], z0, args = ([params_a]), method = "BDF", dense_output = True)
second_vacc_z_a = sol1_a.y[:,-1].copy()
second_vacc_z_a[0] += L0
sol2_a = integrate.solve_ivp(model_vaccination, [second_vaccination_time,250], second_vacc_z_a, args = ([params_a]), method = "BDF", dense_output = True, events = [Athresh_event])
t_simvals_a = np.concatenate((sol1_a.t, sol2_a.t))
A_simvals_a = params_a["Abcrmin"]*np.concatenate((sol1_a.y[12], sol2_a.y[12])) + params_a["Abcrmax"]*np.concatenate((sol1_a.y[13], sol2_a.y[13]))

params_b = params_vaccination()
params_b["kAPC"] = 0

sol1_b = integrate.solve_ivp(model_vaccination, [0, second_vaccination_time], z0, args = ([params_b]), method = "BDF", dense_output = True)
second_vacc_z_b = sol1_b.y[:,-1].copy()
second_vacc_z_b[0] += L0
sol2_b = integrate.solve_ivp(model_vaccination, [second_vaccination_time,250], second_vacc_z_b, args = ([params_b]), method = "BDF", dense_output = True, events = [Athresh_event])
t_simvals_b = np.concatenate((sol1_b.t, sol2_b.t))
A_simvals_b = params_b["Abcrmin"]*np.concatenate((sol1_b.y[12], sol2_b.y[12])) + params_b["Abcrmax"]*np.concatenate((sol1_b.y[13], sol2_b.y[13]))

def sci_fmt(y, pos):
    if y == 0:
        return "0"
    exponent = int(np.log10(y))
    return f"1e{exponent}"


fig1, ax1 = plt.subplots(figsize = (8.5,4))
ax1.plot(t_simvals_a, A_simvals_a, label = "$k_{APC}/k_P = 1$", lw = 3, color = "black")
ax1.plot(t_simvals_b, A_simvals_b, label = "$k_{APC}/k_P = 0$", lw = 3, color = "red")
ax1.set_xlabel("Days after 1st vaccination")
ax1.set_ylabel("$T$ (ng/mL)")
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax1.axhline(A_titer, color = "black", ls = "dotted", lw = 3)
ax1.legend()
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from immuwane.model import params_vaccination, Athresh_val, z_init_vaccination, Athresh_event, model_vaccination
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

### Figure 7A
params = params_vaccination()
z0 = z_init_vaccination()
L0 = z0[0]

second_vaccination_time = 21

sol1 = integrate.solve_ivp(model_vaccination, [0, second_vaccination_time], z0, args = ([params]), method = "BDF", dense_output = True)
second_vacc_z = sol1.y[:,-1].copy()
second_vacc_z[0] += L0
sol2 = integrate.solve_ivp(model_vaccination, [second_vaccination_time,250], second_vacc_z, args = ([params]), method = "BDF", dense_output = True, events = [Athresh_event])
t_simvals = np.concatenate((sol1.t, sol2.t))
A_simvals = params["Abcrmin"]*np.concatenate((sol1.y[12], sol2.y[12])) + params["Abcrmax"]*np.concatenate((sol1.y[13], sol2.y[13]))

params_BN = params_vaccination()
params_BN["BN"] = 1e5
sol1_BN = integrate.solve_ivp(model_vaccination, [0, second_vaccination_time], z0, args = ([params_BN]), method = "BDF", dense_output = True)
second_vacc_z_BN = sol1_BN.y[:,-1].copy()
second_vacc_z_BN[0] += L0
sol2_BN = integrate.solve_ivp(model_vaccination, [second_vaccination_time,250], second_vacc_z_BN, args = ([params_BN]), method = "BDF", dense_output = True, events = [Athresh_event])
t_BN_simvals = np.concatenate((sol1_BN.t, sol2_BN.t))
A_BN_simvals = params["Abcrmin"]*np.concatenate((sol1_BN.y[12], sol2_BN.y[12])) + params["Abcrmax"]*np.concatenate((sol1_BN.y[13], sol2_BN.y[13]))

params_tBM = params_vaccination()
params_tBM["tauBM"] = 25
sol1_tBM = integrate.solve_ivp(model_vaccination, [0, second_vaccination_time], z0, args = ([params_tBM]), method = "BDF", dense_output = True)
second_vacc_z_tBM = sol1_tBM.y[:,-1].copy()
second_vacc_z_tBM[0] += L0
sol2_tBM = integrate.solve_ivp(model_vaccination, [second_vaccination_time,250], second_vacc_z_tBM, args = ([params_tBM]), method = "BDF", dense_output = True, events = [Athresh_event])
t_tBM_simvals = np.concatenate((sol1_tBM.t, sol2_tBM.t))
A_tBM_simvals = params_tBM["Abcrmin"]*np.concatenate((sol1_tBM.y[12], sol2_tBM.y[12])) + params_tBM["Abcrmax"]*np.concatenate((sol1_tBM.y[13], sol2_tBM.y[13]))

sol1_single_vacc = integrate.solve_ivp(model_vaccination, [0, 250], z0, args = ([params]), method = "BDF", dense_output = True)
A_single_vacc = params["Abcrmin"]*sol1_single_vacc.y[12] + params["Abcrmax"]*sol1_single_vacc.y[13]
sol1_BN_single_vacc = integrate.solve_ivp(model_vaccination, [0, 250], z0, args = ([params_BN]), method = "BDF", dense_output = True)
A_BN_single_vacc = params["Abcrmin"]*sol1_BN_single_vacc.y[12] + params["Abcrmax"]*sol1_BN_single_vacc.y[13]
sol1_tBM_single_vacc = integrate.solve_ivp(model_vaccination, [0, 250], z0, args = ([params_tBM]), method = "BDF", dense_output = True)
A_tBM_single_vacc = params["Abcrmin"]*sol1_tBM_single_vacc.y[12] + params["Abcrmax"]*sol1_tBM_single_vacc.y[13]

t_titer = [sol2.t_events[0][0], sol2.t_events[0][1]]
A_titer = Athresh_val(params)

### Figure 7B
pv_vals = np.logspace(1,5, 100)

params1 = params_vaccination()
titer_vals1 = []
for pv1 in pv_vals:
    params1["pV"] = pv1
    titer_vals1.append(Athresh_val(params1))

params05 = params_vaccination()
params05["betaIA"] = params["betaIA"]*0.5
titer_vals05 = []
for pv05 in pv_vals:
    params05["pV"] = pv05
    titer_vals05.append(Athresh_val(params05))
    
params02 = params_vaccination()
params02["betaIA"] = params["betaIA"]*0.2
titer_vals02 = []
for pv02 in pv_vals:
    params02["pV"] = pv02
    titer_vals02.append(Athresh_val(params02))

### Figure 7C
def Athresh_event_double(t,z,params):
    L, S, I, Pf, Pfdc, Pssm, BG, BML, BMH, B1L, B1H, B2, A1, A2, AM = z
    A = (A1*params["Abcrmin"]) + (A2*params["Abcrmax"])
    thresh = 2*Athresh_val(params)
    
    return A - thresh

cases = np.arange(0,100,7)[1:]

Athresh_event.direction = 0
Athresh_event_double.direction = 0

t_Avals_a = []
t_Avals_b = []

for c in np.arange(0,len(cases)):
       
    second_vacc_time = cases[c]
    sol1 = integrate.solve_ivp(model_vaccination, [0,second_vacc_time], z0,
                                args = ([params]), method = "BDF", dense_output = True)
    
    second_vacc_z = sol1.y[:,-1].copy()
    second_vacc_z[0] += L0
    
    sol2a = integrate.solve_ivp(model_vaccination, [second_vacc_time,500], second_vacc_z,
                                args = ([params]), method = "BDF", dense_output = True, events = [Athresh_event])
    
    sol2b = integrate.solve_ivp(model_vaccination, [second_vacc_time,500], second_vacc_z,
                                args = ([params]), method = "BDF", dense_output = True, events = [Athresh_event_double])
    
    if len(sol2a.t_events[0])>1:
        t_Avals_a.append(sol2a.t_events[0][1]-sol2a.t_events[0][0])
    else:
        t_Avals_a.append(0)
        
    if len(sol2b.t_events[0])>1:
        t_Avals_b.append(sol2b.t_events[0][1]-sol2b.t_events[0][0])
    else:
        t_Avals_b.append(0)

cases_str = [str(case) for case in cases]

def sci_fmt(y, pos):
    if y == 0:
        return "0"
    exponent = int(np.log10(y))
    return f"1e{exponent}"

p1 = patches.FancyArrowPatch((t_titer[0], A_titer), (t_titer[1], A_titer), arrowstyle = "|-|, widthA= 10, widthB= 10", color = "blue", lw = 3, zorder = 10)

fig1, ax1 = plt.subplots(figsize = (12.5,4))
ax1.plot(t_simvals, A_simvals, label = "Old", lw = 3, color = "black")
ax1.plot(t_BN_simvals, A_BN_simvals, label = "Young $B_N$", lw = 3, color = "red")
ax1.plot(t_tBM_simvals, A_tBM_simvals, label = "Young $\\tau_{BM}$", lw = 3, color = "green")
ax1.plot(sol1_single_vacc.t, A_single_vacc, lw = 3, ls = "dashed", color = "black")
ax1.plot(sol1_BN_single_vacc.t, A_BN_single_vacc, lw = 3, ls = "dashed", color = "red")
ax1.plot(sol1_tBM_single_vacc.t, A_tBM_single_vacc, lw = 3, ls = "dashed", color = "green")
ax1.add_patch(p1)
ax1.set_xlabel("Days after 1st vaccination")
ax1.set_ylabel("$T$ (ng/mL)")
ax1.set_yticks(np.arange(0, 7e5, 1.5e5))
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax1.axhline(A_titer, color = "black", ls = "dotted", lw = 3)
ax1.legend()

fig2, ax2 = plt.subplots(figsize = (4.5,4))
ax2.plot(pv_vals, titer_vals02, label = r"$q=0.2$", color = "green", lw = 3)
ax2.plot(pv_vals, titer_vals05, label = r"$q=0.5$", color = "red", lw = 3)
ax2.plot(pv_vals, titer_vals1, label = r"$q=1.0$", color = "black", lw = 3)
ax2.axvline(params["pV"]*2, color = "black", lw = 3, ls = "dashed")
ax2.axvline(params["pV"], color = "black", lw = 3, ls = "dotted")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_ylim(bottom = 1e4)
ax2.xaxis.set_major_formatter(FuncFormatter(sci_fmt))
ax2.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
ax2.set_ylabel("$T_c$ (ng/mL)")
ax2.set_xlabel("$p_V$ (copies/(cell*day))")
ax2.legend()

fig3, ax3 = plt.subplots(figsize = (8,4))
ax3.bar(cases_str, t_Avals_a, width = 0.4, color = "black", label = "$T_c$")
ax3.bar(cases_str, t_Avals_b, width = 0.4, color = "red", label = "$2\\times T_c$")
ax3.set_ylabel("Protection time (days)")
ax3.set_xlabel("Time of second dose $t^*$ (days)")
ax3.legend()
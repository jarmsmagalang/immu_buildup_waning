import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
from src.model import params_vaccination, z_init_vaccination, Athresh_event, model_vaccination
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

def param_range(params, r, exp_var = ""):
    param_range = {}
    for k in list(params.keys()):
        val = params[k]
        param_range.update({k: [val-(val*r), val+(val*r)] })
    
    if exp_var == "bN":
        param_range["BN"] = [0.5e5, 1.5e5]
    elif exp_var == "tauBM":
        param_range["tauBM"] = [19,25]
    elif exp_var == "tauA":
        param_range["tauA"] = [30,90]
    return param_range

def run_model(p):
    """
    Run ODE model computing for the protection amplification with a varying parameter set, used for the sensitivity analysis

    Parameters
    ----------
    p : dict
        Set of parameters.

    Returns
    -------
    vals_out : array
        Numpy array containing the input parameters, and the protection amplification.

    """
    
    params = params_vaccination()
    param_names = list(params.keys())
    
    par = {k:v for k,v in zip(param_names, p)}

    second_vacc_time = 21
    
    tfinal = 1000
    
    Athresh_event.direction = 0
    
    z_init = z_init_vaccination()
    
    L0 = 2.5e5
    
    sol0 = integrate.solve_ivp(model_vaccination, [0,tfinal], z_init,
                                args = ([par]), method = "BDF", dense_output = True, events = [Athresh_event])
    z0 = sol0.y
    A1vals0 = z0[11]
    A2vals0 = z0[12]
    Amax0 = max((A1vals0*par["Amin"]) + (A2vals0*par["Amax"]))
    
    sol1 = integrate.solve_ivp(model_vaccination, [0,second_vacc_time], z_init,
                                args = ([par]), method = "BDF", dense_output = True)
    
    second_vacc_z = sol1.y[:,-1].copy()
    second_vacc_z[0] += L0
    sol2 = integrate.solve_ivp(model_vaccination, [second_vacc_time,tfinal], second_vacc_z,
                                args = ([par]), method = "BDF", dense_output = True, events = [Athresh_event])
    
    z = np.concatenate((sol1.y, sol2.y), axis = 1)
    
    A1vals = z[11]
    A2vals = z[12]
    Amax = max((A1vals*par["Amin"]) + (A2vals*par["Amax"]))
    
    A_amp = Amax/Amax0
        
    vals_out = np.append(p, A_amp)
    return vals_out

params = params_vaccination()
param_names = list(params.keys())

param_ranges = list(param_range(params, 0.1, exp_var = "").values())
problem = {"num_vars": len(param_names),
           "names": param_names,
           "bounds": param_ranges}
param_values = saltelli.sample(problem, N=2048)
Ylist = []
for p in param_values:
    Ylist.append(run_model(p))
df = pd.DataFrame(data = Ylist, columns = param_names+["Aamp"])
Y = df["Aamp"].to_numpy()
Si_Aamp = sobol.analyze(problem, Y, print_to_console=True)

df_sens = pd.DataFrame(data = {"param_names": param_names,
                               "total_sobol": Si_Aamp["ST"]})

df_sens["param_names_tex"] = [r'$k_L$',
                                r'$\tau_L$',
                                r'$k_S$',
                                r'$k_{APC}$',
                                r'$k_1$',
                                r'$S_0$',
                                r'$\tau_I$',
                                r'$p_p$',
                                r'$\tau_{Pf}$',
                                r'$\tau_{FDC}$',
                                r'$\tau_{SSM}$',
                                r'$k_P$',
                                r'$c_N$',
                                r'$B_N$',
                                r'$\epsilon$',
                                r'$a_{pL}$',
                                r'$N_{GC}$',
                                r'$\tau_{BM}$',
                                r'$c_1$',
                                r'$\tau_{B1}$',
                                r'$h$',
                                r'$\tau_{B2}$',
                                r'$v$',
                                r'$p_A$',
                                r'$\tau_A$',
                                r'$A_{BCR}^{min}$',
                                r'$A_{BCR}^{max}$',
                                r'$C$',
                                r'$\beta_{IA}$',
                                r'$\beta_{PfA}$',
                                r'$\gamma_{IA}$',
                                r'$\gamma_{PfA}$',
                                r'$\beta_{VA}$',
                                r'$\tau_V$',
                                r'$k_V$',
                                r'$p_V$',
                                r'$\tau_{IV}$']

df_sens["param_description"] = [r'$L$ uptake rate',
                                r'$L$ lifetime',
                                r'$S$ infection rate',
                                r'$S$ uptake of $P_F$',
                                r'$S$ recovery rate',
                                r'Initial $S$ density',
                                r'$I$ lifetime',
                                r'$P_F$ production rate',
                                r'$P_F$ lifetime',
                                r'$P_{FDC}$ lifetime',
                                r'$P_{SSM}$ lifetime',
                                r'$P_F$ uptake rate',
                                r'$B_G$ amplification',
                                r'Initial $B_G$ density',
                                r'$B_G$ differentiation',
                                r'Light zone apoptosis',
                                r'Number of GCs',
                                r'$B_M$ lifetime',
                                r'$B_M$ amplification',
                                r'$B_1$ lifetime',
                                r'$B_1$ specification',
                                r'$B_2$ lifetime',
                                r'$B_G$ specification',
                                r'$A$ production rate',
                                r'$A$ lifetime',
                                r'Maximum BCR affinity',
                                r'Minimum BCR affinity',
                                r'Affinity amplification',
                                r'$I$ neutralization by $A$',
                                r'$P_F$ neutralization by $A$',
                                r'$A$ consumption by $I$',
                                r'$A$ consumption by $P_F$',
                                r'$V$ neutralization by $A$',
                                r'$V$ lifetime',
                                r'$V$ uptake rate',
                                r'$V$ production rate',
                                r'$I$ lifetime due to V']

df_sens = df_sens.sort_values("total_sobol", ascending = False)

z0 = z_init_vaccination()
L0 = z0[0]

v1 = 0.8
v2 = 1.0
second_vaccination_time = 21

sol1 = integrate.solve_ivp(model_vaccination, [0, second_vaccination_time], z0, args = ([params]), method = "BDF", dense_output = True)
second_vacc_z = sol1.y[:,-1].copy()
second_vacc_z[0] += L0
sol2 = integrate.solve_ivp(model_vaccination, [second_vaccination_time,250], second_vacc_z, args = ([params]), method = "BDF", dense_output = True, events = [Athresh_event])
t_simvals = np.concatenate((sol1.t, sol2.t))
A_simvals = params["Abcrmin"]*np.concatenate((sol1.y[12], sol2.y[12])) + params["Abcrmax"]*np.concatenate((sol1.y[13], sol2.y[13]))

params_v1 = params_vaccination()
params_v1["v"] = v1
sol1_v1 = integrate.solve_ivp(model_vaccination, [0, second_vaccination_time], z0, args = ([params_v1]), method = "BDF", dense_output = True)
second_vacc_z_v1 = sol1_v1.y[:,-1].copy()
second_vacc_z_v1[0] += L0
sol2_v1 = integrate.solve_ivp(model_vaccination, [second_vaccination_time,250], second_vacc_z_v1, args = ([params_v1]), method = "BDF", dense_output = True)
t_v1_simvals = np.concatenate((sol1_v1.t, sol2_v1.t))
A_v1_simvals = params_v1["Abcrmin"]*np.concatenate((sol1_v1.y[12], sol2_v1.y[12])) + params_v1["Abcrmax"]*np.concatenate((sol1_v1.y[13], sol2_v1.y[13]))

params_v2 = params_vaccination()
params_v2["v"] = v2
sol1_v2 = integrate.solve_ivp(model_vaccination, [0, second_vaccination_time], z0, args = ([params_v2]), method = "BDF", dense_output = True)
second_vacc_z_v2 = sol1_v2.y[:,-1].copy()
second_vacc_z_v2[0] += L0
sol2_v2 = integrate.solve_ivp(model_vaccination, [second_vaccination_time,250], second_vacc_z_v2, args = ([params_v2]), method = "BDF", dense_output = True)
t_v2_simvals = np.concatenate((sol1_v2.t, sol2_v2.t))
A_v2_simvals = params_v2["Abcrmin"]*np.concatenate((sol1_v2.y[12], sol2_v2.y[12])) + params_v2["Abcrmax"]*np.concatenate((sol1_v2.y[13], sol2_v2.y[13]))

sol1_single_vacc = integrate.solve_ivp(model_vaccination, [0, 250], z0, args = ([params]), method = "BDF", dense_output = True)
A_single_vacc = params["Abcrmin"]*sol1_single_vacc.y[12] + params["Abcrmax"]*sol1_single_vacc.y[13]
sol1_v1_single_vacc = integrate.solve_ivp(model_vaccination, [0, 250], z0, args = ([params_v1]), method = "BDF", dense_output = True)
A_v1_single_vacc = params_v1["Abcrmin"]*sol1_v1_single_vacc.y[12] + params_v1["Abcrmax"]*sol1_v1_single_vacc.y[13]
sol1_v2_single_vacc = integrate.solve_ivp(model_vaccination, [0, 250], z0, args = ([params_v2]), method = "BDF", dense_output = True)
A_v2_single_vacc = params_v2["Abcrmin"]*sol1_v2_single_vacc.y[12] + params_v2["Abcrmax"]*sol1_v2_single_vacc.y[13]

A_titer = [max(A_single_vacc), max(A_simvals)]
A_v1_titer = [max(A_v1_single_vacc), max(A_v1_simvals)]
A_v2_titer = [max(A_v2_single_vacc), max(A_v2_simvals)]

t_Aamp = []

params_v = params_vaccination()
vcases = np.linspace(0,1,100)

for v in np.arange(0,len(vcases)):
      
    params_v["v"] = vcases[v]
    sol1_v = integrate.solve_ivp(model_vaccination, [0, second_vaccination_time], z0, args = ([params_v]), method = "BDF", dense_output = True)
    second_vacc_z_v = sol1_v.y[:,-1].copy()
    second_vacc_z_v[0] += L0
    sol2_v = integrate.solve_ivp(model_vaccination, [second_vaccination_time,1000], second_vacc_z_v, args = ([params_v]), method = "BDF", dense_output = True, events = [Athresh_event])
    A_simvals_v = params_v["Abcrmin"]*np.concatenate((sol1_v.y[12], sol2_v.y[12])) + params_v["Abcrmax"]*np.concatenate((sol1_v.y[13], sol2_v.y[13]))

    sol1_single_vacc_v = integrate.solve_ivp(model_vaccination, [0, 1000], z0, args = ([params_v]), method = "BDF", dense_output = True)
    A_single_vacc_v = params_v["Abcrmin"]*sol1_single_vacc_v.y[12] + params_v["Abcrmax"]*sol1_single_vacc_v.y[13]

    t_Aamp.append(max(A_simvals_v)/max(A_single_vacc_v))

bartext = df_sens["param_description"].tolist()

df_sens["barcolor"] = "black"
df_sens.loc[df_sens.param_names.isin(['betaVA', 'tauV', 'kV', 'pV', 'tauIV']), "barcolor"] = "red"

fig1, ax1 = plt.subplots(figsize = (15,8))
bars = ax1.bar(df_sens["param_names_tex"], df_sens["total_sobol"], color = df_sens["barcolor"].tolist(), width = 0.5)
for r in range(len(bars))[:10]:
    height = bars[r].get_height()
    ax1.text(bars[r].get_x() + (1.25*bars[r].get_width()), height+(0.01), bartext[r], ha='right', va='bottom', rotation=90)
ax1.set_ylabel("Total Sobol' Index")
ax1.tick_params(axis = "x", labelrotation=90)
ax1.set_ylim(0.2,1)
fig1.tight_layout()

ax2 = fig1.add_axes([0.5, 0.45, 0.45, 0.45])
ax2.plot(vcases, t_Aamp, color = "black", lw = 3)
ax2.set_ylabel("Titer amplification ($T_{double}/T_{single}$)")
ax2.set_xlabel("$v$")
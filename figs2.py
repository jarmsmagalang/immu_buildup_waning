import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
from immuwane.model import params_vaccination, z_init_vaccination, Athresh_event, model_vaccination
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
    Run ODE model computing for the protection time with a varying parameter set, used for the sensitivity analysis

    Parameters
    ----------
    p : dict
        Set of parameters.

    Returns
    -------
    vals_out : array
        Numpy array containing the input parameters, and the protection time.

    """
    
    params = params_vaccination()
    param_names = list(params.keys())
    
    par = {k:v for k,v in zip(param_names, p)}

    second_vacc_time = 21
    
    tfinal = 1000
    
    Athresh_event.direction = 0
    
    z_init = z_init_vaccination()
    
    L0 = 2.5e5
    
    sol1 = integrate.solve_ivp(model_vaccination, [0,second_vacc_time], z_init,
                                args = ([par]), method = "BDF", dense_output = True)
    
    second_vacc_z = sol1.y[:,-1].copy()
    second_vacc_z[0] += L0
    sol2 = integrate.solve_ivp(model_vaccination, [second_vacc_time,tfinal], second_vacc_z,
                                args = ([par]), method = "BDF", dense_output = True, events = [Athresh_event])
    
    if len(sol2.t_events[0])==2:
        A_time = sol2.t_events[0][1]-sol2.t_events[0][0]
    else:
        A_time = 0
        
    vals_out = np.append(p, A_time)
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
df = pd.DataFrame(data = Ylist, columns = param_names+["opt_svt"])
Y = df["opt_svt"].to_numpy()
Si_Athresh = sobol.analyze(problem, Y, print_to_console=True)

df_sens = pd.DataFrame(data = {"param_names": param_names,
                               "total_sobol": Si_Athresh["ST"]})

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
                                r'$S$ absorption of $P_F$',
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
params_pv = params_vaccination()
second_vacc_time = 21
pcases = np.linspace(30,3000,101)

Athresh_event.direction = 0

t_Avals = []

for pv in np.arange(0,len(pcases)):
      
    params_pv["pV"] = pcases[pv]
    sol1 = integrate.solve_ivp(model_vaccination, [0,second_vacc_time], z0,
                                args = ([params_pv]), method = "BDF", dense_output = True)
    
    second_vacc_z = sol1.y[:,-1].copy()
    second_vacc_z[0] += L0
    
    sol2 = integrate.solve_ivp(model_vaccination, [second_vacc_time,50000], second_vacc_z,
                                args = ([params_pv]), method = "BDF", dense_output = True, events = [Athresh_event])
   
    if len(sol2.t_events[0])>1:
        t_Avals.append(sol2.t_events[0][-1]-sol2.t_events[0][-2])
    else:
        
        t_Avals.append(0)

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
ax1.set_ylim(0.25,1)
fig1.tight_layout()

ax2 = fig1.add_axes([0.5, 0.45, 0.45, 0.45])
ax2.plot(pcases, t_Avals, color = "black", lw = 3)
ax2.set_ylabel("Protection time (days)")
ax2.set_xlabel("$p_V$ (copies/(cell*day))")

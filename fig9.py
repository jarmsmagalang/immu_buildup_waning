import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
from immuwane.model import params_vaccination, model_vaccination, z_init_vaccination, Athresh_event
import scipy.integrate as integrate
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

param_ranges_tauA = list(param_range(params, 0.1, exp_var = "tauA").values())
problem_tauA = {"num_vars": len(param_names),
                "names": param_names,
                "bounds": param_ranges_tauA}
param_values_tauA = saltelli.sample(problem_tauA, N=2048)
Ylist_tauA = []
for p_tauA in param_values_tauA:
    Ylist_tauA.append(run_model(p_tauA))
df_tauA = pd.DataFrame(data = Ylist_tauA, columns = param_names+["opt_svt"])
Y_tauA = df_tauA["opt_svt"].to_numpy()

param_ranges_BN = list(param_range(params, 0.1, exp_var = "BN").values())
problem_BN = {"num_vars": len(param_names),
              "names": param_names,
              "bounds": param_ranges_BN}
param_values_BN = saltelli.sample(problem_BN, N=2048)
Ylist_BN = []
for p_BN in param_values_BN:
    Ylist_BN.append(run_model(p_BN))
df_BN = pd.DataFrame(data = Ylist_BN, columns = param_names+["opt_svt"])
Y_BN = df_BN["opt_svt"].to_numpy()

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

df_sens = df_sens[~df_sens.param_names.isin(['betaVA', 'tauV', 'kV', 'pV', 'tauIV'])].sort_values("total_sobol", ascending = False).iloc[:10]

Y2 = Y[Y>0]
fit_shape, fit_loc, fit_scale =spst.lognorm.fit(Y2)
xvals = np.linspace(min(Y2), max(Y2), 1000)
yvals = spst.lognorm.pdf(xvals, fit_shape, loc=fit_loc, scale=fit_scale)
dist = spst.lognorm(fit_shape, loc=fit_loc, scale=fit_scale)
fit_mean = dist.mean()
fit_var = dist.var()

counts_base, bin_edges_base = np.histogram(Y2, bins="auto", density=True)
cdf_base = np.cumsum(counts_base)
waning_base = 1-(cdf_base / cdf_base[-1])
bin_centers_base = (bin_edges_base[1:] + bin_edges_base[:-1]) / 2

counts_tauA, bin_edges_tauA = np.histogram(Y_tauA[Y_tauA>0], bins="auto", density=True)
cdf_tauA = np.cumsum(counts_tauA)
waning_tauA = 1-(cdf_tauA / cdf_tauA[-1])
bin_centers_tauA = (bin_edges_tauA[1:] + bin_edges_tauA[:-1]) / 2

counts_BN, bin_edges_BN = np.histogram(Y_BN[Y_BN>0], bins="auto", density=True)
cdf_BN = np.cumsum(counts_BN)
waning_BN = 1-(cdf_BN / cdf_BN[-1])
bin_centers_BN = (bin_edges_BN[1:] + bin_edges_BN[:-1]) / 2

bartext = ["$B_2$ lifetime",
           "$B_G$ specification",
           "$B_1$ specification",
           "$A$ production rate",
           "$B_1$ lifetime",
           "$L$ uptake rate",
           "$L$ lifetime",
           "$S$ infection rate",
           "$B_M$ lifetime",
           "$P_F$ production rate"]

fig1, ax1 = plt.subplots(figsize = (6,9))
bars = ax1.bar(df_sens["param_names_tex"], df_sens["total_sobol"], color = "black", width = 0.5)
for r in range(len(bars)):
    height = bars[r].get_height()
    ax1.text(bars[r].get_x() + (1*bars[r].get_width()), height+(height*0.01), bartext[r], ha='right', va='bottom', rotation=90)#, rotation_mode='anchor')
ax1.set_ylabel("Total Sobol' Index")
ax1.tick_params(axis = "x", labelrotation=90)
ax1.set_ylim(0.25,0.45)
fig1.tight_layout()

param_text = (
    r"\begin{eqnarray*}"+
    r"\mathrm{shape} &=&"+str(round(fit_shape, 3))+" \\"+
    r"\\mathrm{loc} &=&"+'{:.3f}'.format(round(fit_loc, 3))+" \\"+
    r"\\mathrm{scale} &=&"+str(round(fit_scale, 3))+
    r"\end{eqnarray*}"
)

fig2, ax2 = plt.subplots(figsize = (6,4.5))
ax2.hist(Y2, bins = "auto", density = True, color = "black")
ax2.plot(xvals, yvals, color = "red", lw = 3)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax2.set_yticks(np.arange(0, 1.2e-2, 2.5e-3))
matplotlib.rcParams["text.usetex"] = True
ax2.text(
    0.95, 0.95, param_text,
    transform=ax2.transAxes,
    fontsize=15, va="top", ha="right",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)
matplotlib.rcParams["text.usetex"] = False
ax2.set_xlabel("Protection time (days)")
ax2.set_ylabel("Density")

fig3, ax3 = plt.subplots(figsize = (6,4.5))
ax3.plot(bin_centers_base, waning_base, lw = 3, color = "black", label = "10% var.")
ax3.plot(bin_centers_tauA, waning_tauA, lw = 5, ls = "dotted", color = "blue", label = "$\\tau_A$ var.")
ax3.plot(bin_centers_BN, waning_BN, lw = 5, ls = "dotted", color = "green", label = "$B_N$ var.")
ax3.legend()
ax3.set_xlabel("Protection time (days)")
ax3.set_ylabel("Waning function")
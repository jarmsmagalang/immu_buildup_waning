import numpy as np
import matplotlib.pyplot as plt
from immuwane.model import params_vaccination, z_init_vaccination
import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams.update({'font.size':15})
matplotlib.rcParams.update({'font.size':15})
matplotlib.rcParams['axes.linewidth'] = 3
matplotlib.rcParams['xtick.major.size'] = 7
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['xtick.minor.size'] = 5
matplotlib.rcParams['xtick.minor.width'] = 3
matplotlib.rcParams['ytick.major.size'] = 7
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['ytick.minor.size'] = 5
matplotlib.rcParams['ytick.minor.width'] = 3

z0 = z_init_vaccination()
L0 = z0[0]
def I_max(tauI, tauL, kL, S0):
    params = params_vaccination()
    params["tauI"] = tauI
    params["tauL"] = tauL
    params["kL"] = kL
    params["S0"] = S0
    
    tauLC = 1/(params["kL"]*params["S0"]+(1/params["tauL"]))
    x = params["tauI"]/tauLC
    return params["kL"]*L0*params["S0"]*params["tauI"]*x**(-x/(x-1))

def t0(tauI, tauL, kL, S0):
    params = params_vaccination()
    params["tauI"] = tauI
    params["tauL"] = tauL
    params["kL"] = kL
    params["S0"] = S0
    
    tauLC = 1/(params["kL"]*params["S0"]+(1/params["tauL"]))
    x = params["tauI"]/tauLC
    return params["tauI"]*(np.log(x)/(x-1))

params = params_vaccination()
tauIvals = np.linspace(0,20,100)

Imaxvals_1a = [I_max(tauI, params["tauL"], params["kL"], params["S0"]) for tauI in tauIvals]
Imaxvals_1b = [I_max(tauI, params["tauL"]*2, params["kL"], params["S0"]) for tauI in tauIvals]
Imaxvals_1c = [I_max(tauI, params["tauL"]/2, params["kL"], params["S0"]) for tauI in tauIvals]

Imaxvals_2a = [I_max(tauI, params["tauL"], params["kL"], params["S0"]) for tauI in tauIvals]
Imaxvals_2b = [I_max(tauI, params["tauL"], params["kL"]*2, params["S0"]) for tauI in tauIvals]
Imaxvals_2c = [I_max(tauI, params["tauL"], params["kL"]/2, params["S0"]) for tauI in tauIvals]

Imaxvals_3a = [I_max(tauI, params["tauL"], params["kL"], params["S0"]) for tauI in tauIvals]
Imaxvals_3b = [I_max(tauI, params["tauL"], params["kL"], params["S0"]*10) for tauI in tauIvals]
Imaxvals_3c = [I_max(tauI, params["tauL"], params["kL"], params["S0"]/10) for tauI in tauIvals]

t0vals_1a = [t0(tauI, params["tauL"], params["kL"], params["S0"]) for tauI in tauIvals]
t0vals_1b = [t0(tauI, params["tauL"]*2, params["kL"], params["S0"]) for tauI in tauIvals]
t0vals_1c = [t0(tauI, params["tauL"]/2, params["kL"], params["S0"]) for tauI in tauIvals]

t0vals_2a = [t0(tauI, params["tauL"], params["kL"], params["S0"]) for tauI in tauIvals]
t0vals_2b = [t0(tauI, params["tauL"], params["kL"]*2, params["S0"]) for tauI in tauIvals]
t0vals_2c = [t0(tauI, params["tauL"], params["kL"]/2, params["S0"]) for tauI in tauIvals]

t0vals_3a = [t0(tauI, params["tauL"], params["kL"], params["S0"]) for tauI in tauIvals]
t0vals_3b = [t0(tauI, params["tauL"], params["kL"], params["S0"]*10) for tauI in tauIvals]
t0vals_3c = [t0(tauI, params["tauL"], params["kL"], params["S0"]/10) for tauI in tauIvals]

fig1, (ax1a, ax1b) = plt.subplots(nrows = 2, sharex = True, figsize = (5,9))
ax1a.plot(tauIvals, Imaxvals_1b, lw = 3, c = "red", label = "14")
ax1b.plot(tauIvals, t0vals_1b, lw = 3, c = "red", label = "14")
ax1a.plot(tauIvals, Imaxvals_1a, lw = 3, c = "black", label = "7 (reference)")
ax1b.plot(tauIvals, t0vals_1a, lw = 3, c = "black", label = "7 (reference)")
ax1a.plot(tauIvals, Imaxvals_1c, lw = 3, c = "blue", label = "3.5")
ax1b.plot(tauIvals, t0vals_1c, lw = 3, c = "blue", label = "3.5")
ax1a.set_yticks(np.arange(0, 1.6e5, 0.5e5))
ax1a.set_ylabel("$I_{max}$")
ax1a.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax1b.set_ylabel("$t_0$")
ax1b.set_xlabel("$\\tau_I$ (days)")
ax1a.legend(title = "$\\tau_L$ (days)")

fig2, (ax2a, ax2b) = plt.subplots(nrows = 2, sharex = True, figsize = (5,9))
ax2a.plot(tauIvals, Imaxvals_2b, lw = 3, c = "red", label = "4")
ax2b.plot(tauIvals, t0vals_2b, lw = 3, c = "red", label = "4")
ax2a.plot(tauIvals, Imaxvals_2a, lw = 3, c = "black", label = "2 (reference)")
ax2b.plot(tauIvals, t0vals_2a, lw = 3, c = "black", label = "2 (reference)")
ax2a.plot(tauIvals, Imaxvals_2c, lw = 3, c = "blue", label = "1")
ax2b.plot(tauIvals, t0vals_2c, lw = 3, c = "blue", label = "1")
ax2a.set_yticks(np.arange(0, 1.6e5, 0.5e5))
ax2a.set_ylabel("$I_{max}$")
ax2a.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax2b.set_ylabel("$t_0$")
ax2b.set_xlabel("$\\tau_I$ (days)")
ax2a.legend(title = "$k_L$ ($10^{-7}$ml/(day$\\cdot$cell))")

fig3, (ax3a, ax3b) = plt.subplots(nrows = 2, sharex = True, figsize = (5,9))
ax3a.plot(tauIvals, Imaxvals_3b, lw = 3, c = "red", label = "10")
ax3b.plot(tauIvals, t0vals_3b, lw = 3, c = "red", label = "10")
ax3a.plot(tauIvals, Imaxvals_3a, lw = 3, c = "black", label = "1 (reference)")
ax3b.plot(tauIvals, t0vals_3a, lw = 3, c = "black", label = "1 (reference)")
ax3a.plot(tauIvals, Imaxvals_3c, lw = 3, c = "blue", label = "0.1")
ax3b.plot(tauIvals, t0vals_3c, lw = 3, c = "blue", label = "0.1")
ax3a.set_ylabel("$I_{max}$")
ax3a.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax3b.set_ylabel("$t_0$")
ax3b.set_xlabel("$\\tau_I$ (days)")
ax3a.legend(title = "$S_0$ ($10^6$ cells/mL)", loc = "lower right")
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from immuwane.model import params_vaccination, z_init_vaccination, model_vaccination
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
from matplotlib.ticker import FuncFormatter

params = params_vaccination()
z0 = z_init_vaccination()

L0 = z0[0]
tauLC = 1/(params["kL"]*params["S0"]+(1/params["tauL"])) #See discussion after Eq. A1
I0 = (params["kL"]*L0*params["S0"]*params["tauI"])/((params["tauI"]/tauLC)-1) #Eq. A2
x = params["tauI"]/tauLC #Eq. A3
I_max = params["kL"]*L0*params["S0"]*params["tauI"]*x**(-x/(x-1)) #Eq. A3
t0 = params["tauI"]*(np.log(x)/(x-1)) #See discussion after Eq. A3
tauPC = 1/(params["kP"]*params["S0"]+(1/params["tauPf"])) #See discussion after Eq. A3
P0 = params["pp"]*I0*tauPC #Eq. A4

def L_ana(t):
    """
    LNP density with S=S0, Eq. A1

    Parameters
    ----------
    t : float
        Time.

    Returns
    -------
    float
        LNP density at S=S0 at time t.

    """
    return L0*np.exp(-(params["kL"]*params["S0"]+(1/params["tauL"]))*t)

def I_ana(t):
    """
    I density with S=S0, Eq. A2

    Parameters
    ----------
    t : float
        Time.

    Returns
    -------
    float
        I density at S=S0 at time t.

    """
    return I0*(np.exp(-t/params["tauI"])-np.exp(-t/tauLC))

def Pf_ana(t):
    """
    Pf density with S=S0, Eq. A4

    Parameters
    ----------
    t : float
        Time.

    Returns
    -------
    float
        Pf density at S=S0 at time t.

    """
    term1 = (params["tauI"]/(params["tauI"]-tauPC))*(np.exp(-t/params["tauI"]) - np.exp(-t/tauPC))
    term2 = (tauLC/(tauLC-tauPC))*(np.exp(-t/tauLC) - np.exp(-t/tauPC))
    return P0*(term1-term2)

def Pfdc_ana(tauPi, t):
    """
    Pfdc density with S=S0, Eq. A5

    Parameters
    ----------
    t : float
        Time.

    Returns
    -------
    float
        Pfdc density at S=S0 at time t.

    """
    term1a = (params["tauI"]**2)/((params["tauI"]-tauPC)*(params["tauI"]-tauPi))
    term1b = np.exp(-t/params["tauI"])-np.exp(-t/tauPi)
    
    term2a = (tauLC**2)/((tauLC-tauPC)*(tauLC-tauPi))
    term2b = np.exp(-t/tauLC)-np.exp(-t/tauPi)
    
    term3a = ((tauPC**2)*(params["tauI"]-tauLC))/((tauPC-params["tauI"])*(tauPC-tauLC)*(tauPC-tauPi))
    term3b = np.exp(-t/tauPC)-np.exp(-t/tauPi)
    
    return (params["kP"]/2)*params["S0"]*P0*tauPi*((term1a*term1b)-(term2a*term2b)+(term3a*term3b))

def model_vaccination_noA(t, z, params):
    """
    Modified ODE for the extended model without an antibody response

    Parameters
    ----------
    t : float
        Time point at which to solve the ODE.
    z : tuple
        Density of the cells, antigens, and antibodies to be solved at time t. Follows this order:
            L: LNPs, Eq. 1
            S: Susceptible cells, Eq. 2
            I: Infected cells, Eq. 3#
            Pf: Free floating protein, Eq. 4#
            Pfdc: Protein immune complex presented to the GC, Eq. 5A
            Pssm: Protein immune complex presented to the memory B cell, Eq. 5B
            BG: Germinal center B cells, Eq. 6*
            BML: Low affinity memory B cells, Eq. 9#
            BMH: High affinity memory B cells, Eq. 10#
            B1L: Low affinity short living plasma cells, Eq. 12
            B1H: High affinity short living plasma cells, Eq. 13
            B2: Long living plasma cells, Eq. 11
            A1: Low affinity antibodies, Eq. 14#
            A2: High affinity antibodies, Eq. 15#
            Abcr: Affinity maturation, Eq. 8#
    params : dict
        Dictionary of parameters.

    Returns
    -------
    z : tuple
        Densities of the cells, antigens, and antibodies solved at time t+dt, where dt is set by the ODE solver.
        Follows the order of the input z.

    """
    
    L, S, I, Pf, Pfdc, Pssm, BG, B0L, B0H, B1L, B1H, B2, A1, A2, Abcr = z
    
    A = (params["Abcrmin"]*A1) + (params["Abcrmax"]*A2)
    
    gGC=(1 - np.exp(-params["C"] *Pfdc*Abcr/(A+BG+1))) 
    gB0L=(1 - np.exp(-params["C"] *Pssm/(A+B0L+1)))
    gB0H=(1 - np.exp(-params["C"] *Pssm/( (A1/params["Abcrmax"]) + A2 +B0H+1)))
    
    f = (params["Abcrmax"]-Abcr)/(params["Abcrmax"]-params["Abcrmin"])
    
    dLdt = - (params["kL"]*S*L) - (L/params["tauL"])
    dSdt = - (params["kS"]*S*L) - (params["kAPC"]*S*Pf) + (params["k1"]*(params["S0"]-S))
    dIdt = (params["kS"]*S*L) - (I/params["tauI"])
    dPfdt = (params["pp"]*I) - (params["kP"]*S*Pf) - (Pf/params["tauPf"])
    dPfdcdt = ((params["kP"]/2)*S*Pf) - (Pfdc/params["taufdc"])
    dPssmdt = ((params["kP"]/2)*S*Pf) - (Pssm/params["taussm"])
    dBGdt = (params["cN"]*gGC*params["BN"]) - (params["epsilon"]*BG) - (params["apL"]*(1-gGC)*BG)
    dB0Ldt = (params["NGC"]*f*params["epsilon"]*BG) + ((1-params["h"])*params["c1"]*gB0L*B0L) - (B0L/params["tauB0"])
    dB0Hdt = (params["NGC"]*params["v"]*(1-f)*params["epsilon"]*BG) + ((1-params["h"])*params["c1"]*gB0H*B0H) - (B0H/params["tauB0"])
    dB1Ldt = (params["h"]*params["c1"]*gB0L*B0L) - (B1L/params["tauB1"])
    dB1Hdt = (params["h"]*params["c1"]*gB0H*B0H) - (B1H/params["tauB1"])
    dB2dt = (params["NGC"]*(1-params["v"])*(1-f)*params["epsilon"]*BG) - (B2/params["tauB2"])
    dA1dt = 0
    dA2dt = 0
    dAbcrdt = params["epsilon"]*(params["Abcrmin"] - Abcr + ((params["Abcrmax"]-params["Abcrmin"])*gGC) )
    
    z = dLdt, dSdt, dIdt, dPfdt, dPfdcdt, dPssmdt, dBGdt, dB0Ldt, dB0Hdt, dB1Ldt, dB1Hdt, dB2dt, dA1dt, dA2dt, dAbcrdt
    
    return z

def model_vaccination_S0noA(t, z, params):
    """
    Modified ODE for the extended model without an antibody response and at S=S0

    Parameters
    ----------
    t : float
        Time point at which to solve the ODE.
    z : tuple
        Density of the cells, antigens, and antibodies to be solved at time t. Follows this order:
            L: LNPs, Eq. 1
            S: Susceptible cells, Eq. 2
            I: Infected cells, Eq. 3#
            Pf: Free floating protein, Eq. 4#
            Pfdc: Protein immune complex presented to the GC, Eq. 5A
            Pssm: Protein immune complex presented to the memory B cell, Eq. 5B
            BG: Germinal center B cells, Eq. 6*
            BML: Low affinity memory B cells, Eq. 9#
            BMH: High affinity memory B cells, Eq. 10#
            B1L: Low affinity short living plasma cells, Eq. 12
            B1H: High affinity short living plasma cells, Eq. 13
            B2: Long living plasma cells, Eq. 11
            A1: Low affinity antibodies, Eq. 14#
            A2: High affinity antibodies, Eq. 15#
            Abcr: Affinity maturation, Eq. 8#
    params : dict
        Dictionary of parameters.

    Returns
    -------
    z : tuple
        Densities of the cells, antigens, and antibodies solved at time t+dt, where dt is set by the ODE solver.
        Follows the order of the input z.

    """
    
    L, S, I, Pf, Pfdc, Pssm, BG, B0L, B0H, B1L, B1H, B2, A1, A2, Abcr = z
    
    A = (params["Abcrmin"]*A1) + (params["Abcrmax"]*A2)
    
    betaPfA1 = params["betaPfA"]*params["Abcrmin"]
    betaPfA2 = params["betaPfA"]*params["Abcrmax"]
    
    gGC=(1 - np.exp(-params["C"] *Pfdc*Abcr/(A+BG+1))) 
    gB0L=(1 - np.exp(-params["C"] *Pssm/(A+B0L+1)))
    gB0H=(1 - np.exp(-params["C"] *Pssm/( (A1/params["Abcrmax"]) + A2 +B0H+1)))
    
    f = (params["Abcrmax"]-Abcr)/(params["Abcrmax"]-params["Abcrmin"])
    
    dLdt = - (params["kL"]*params["S0"]*L) - (L/params["tauL"])
    dSdt = params["S0"]
    dIdt = (params["kS"]*params["S0"]*L) - (I/params["tauI"])
    dPfdt = (params["pp"]*I) - (params["kP"]*params["S0"]*Pf) - (Pf/params["tauPf"]) - (betaPfA1*A1*Pf) - (betaPfA2*A2*Pf)
    dPfdcdt = ((params["kP"]/2)*params["S0"]*Pf) - (Pfdc/params["taufdc"])
    dPssmdt = ((params["kP"]/2)*params["S0"]*Pf) - (Pssm/params["taussm"])
    dBGdt = (params["cN"]*gGC*params["BN"]) - (params["epsilon"]*BG) - (params["apL"]*(1-gGC)*BG)
    dB0Ldt = (params["NGC"]*f*params["epsilon"]*BG) + ((1-params["h"])*params["c1"]*gB0L*B0L) - (B0L/params["tauB0"])
    dB0Hdt = (params["NGC"]*params["v"]*(1-f)*params["epsilon"]*BG) + ((1-params["h"])*params["c1"]*gB0H*B0H) - (B0H/params["tauB0"])
    dB1Ldt = (params["h"]*params["c1"]*gB0L*B0L) - (B1L/params["tauB1"])
    dB1Hdt = (params["h"]*params["c1"]*gB0H*B0H) - (B1H/params["tauB1"])
    dB2dt = (params["NGC"]*(1-params["v"])*(1-f)*params["epsilon"]*BG) - (B2/params["tauB2"])
    dA1dt = 0
    dA2dt = 0
    dAbcrdt = params["epsilon"]*(params["Abcrmin"] - Abcr + ((params["Abcrmax"]-params["Abcrmin"])*gGC) )
    
    z = dLdt, dSdt, dIdt, dPfdt, dPfdcdt, dPssmdt, dBGdt, dB0Ldt, dB0Hdt, dB1Ldt, dB1Hdt, dB2dt, dA1dt, dA2dt, dAbcrdt
    
    return z

tvals = np.linspace(0,20, 101)
L_anavals = [L_ana(t) for t in tvals]
I_anavals = [I_ana(t) for t in tvals]
Pf_anavals = [Pf_ana(t) for t in tvals]
Pfdc_anavals = [Pfdc_ana(params["taufdc"], t) for t in tvals]
Pssm_anavals = [Pfdc_ana(params["taussm"], t) for t in tvals]

sol = integrate.solve_ivp(model_vaccination, [0,20], z0, args = ([params]), method = "BDF", dense_output = True)
t_simvals = sol.t
L_simvals = sol.y[0]
I_simvals = sol.y[2]
Pf_simvals = sol.y[3]
Pfdc_simvals = sol.y[4]
Pssm_simvals = sol.y[5]

solnoA = integrate.solve_ivp(model_vaccination_noA, [0,20], z0, args = ([params]), method = "BDF", dense_output = True)
t_noAsimvals = solnoA.t
L_noAsimvals = solnoA.y[0]
I_noAsimvals = solnoA.y[2]
Pf_noAsimvals = solnoA.y[3]
Pfdc_noAsimvals = solnoA.y[4]
Pssm_noAsimvals = solnoA.y[5]

solS0noA = integrate.solve_ivp(model_vaccination_S0noA, [0,20], z0, args = ([params]), method = "BDF", dense_output = True)
t_S0noAsimvals = solS0noA.t
L_S0noAsimvals = solS0noA.y[0]
I_S0noAsimvals = solS0noA.y[2]
Pf_S0noAsimvals = solS0noA.y[3]
Pfdc_S0noAsimvals = solS0noA.y[4]
Pssm_S0noAsimvals = solS0noA.y[5]

def sci_fmt(y, pos):
    if y == 0:
        return "0"
    exponent = int(np.log10(y))
    return f"1e{exponent}"

fig1, ax1 = plt.subplots()
ax1.plot(tvals, L_anavals, label = "Theo.", lw = 3, color = "red")
ax1.plot(t_simvals, L_simvals, label = "Sim.", lw = 3, color = "black")
ax1.plot(t_noAsimvals, L_noAsimvals, label = "No AB", lw = 6, ls = "dotted", color = "green")
ax1.plot(t_S0noAsimvals, L_S0noAsimvals, label = "$S_0$, no AB", lw = 6, ls = "dotted", color = "blue")
ax1.set_yscale("log")
ax1.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
ax1.set_xlabel("Days after 1st vaccination")
ax1.set_ylabel("$L$ (copies/mL)")
ax1.legend()

fig2, ax2 = plt.subplots()
ax2.plot(tvals, I_anavals, lw = 3, color = "red")
ax2.plot(t_simvals, I_simvals, lw = 3, color = "black")
ax2.plot(t_noAsimvals, I_noAsimvals, lw = 6, ls = "dotted", color = "green")
ax2.plot(t_S0noAsimvals, I_S0noAsimvals, lw = 6, ls = "dotted", color = "blue")
ax2.axhline(I_max, lw = 2, color = "black", ls = "dashed", label = "$I_{max}$")
ax2.axvline(t0, lw = 2, color = "black", ls = "dotted", label = "$t_0$" )
ax2.set_yticks(np.arange(0, 7e4, 15000))
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax2.set_ylabel("$I$ (cells/mL)")
ax2.set_xlabel("Days after 1st vaccination")
ax2.legend(bbox_to_anchor = (1, 0.95))

fig3, ax3 = plt.subplots()
ax3.plot(tvals, Pf_anavals, label = "Theo.", lw = 3, color = "red")
ax3.plot(t_simvals, Pf_simvals, label = "Sim.", lw = 3, color = "black")
ax3.plot(t_noAsimvals, Pf_noAsimvals, label = "No AB", lw = 6, ls = "dotted", color = "green")
ax3.plot(t_S0noAsimvals, Pf_S0noAsimvals, label = "S0, no AB", lw = 6, ls= "dotted", color = "blue")
ax3.set_ylabel("$P_f$ (copies/mL)")
ax3.set_xlabel("Days after 1st vaccination")
ax3.legend()

fig4, (ax4a, ax4b) = plt.subplots(nrows = 2, sharex = True)
ax4a.plot(tvals, Pfdc_anavals, label = "Theo.", lw = 3, color = "red")
ax4a.plot(t_simvals, Pfdc_simvals, label = "Sim.", lw = 3, color = "black")
ax4a.plot(t_noAsimvals, Pfdc_noAsimvals, label = "No AB", lw = 6, ls = "dotted", color = "green")
ax4a.plot(t_S0noAsimvals, Pfdc_S0noAsimvals, label = "S0, no AB", lw = 6, ls = "dotted", color = "blue")
ax4a.set_yticks(np.arange(0, 1e6, 250000))
ax4a.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax4a.set_ylabel("$P_{FDC}$ (copies/mL)")

ax4b.plot(tvals, Pssm_anavals, label = "Theo.", lw = 3, color = "red")
ax4b.plot(t_simvals, Pssm_simvals, label = "Sim.", lw = 3, color = "black")
ax4b.plot(t_noAsimvals, Pssm_noAsimvals, label = "No AB", lw = 6, ls = "dotted", color = "green")
ax4b.plot(t_S0noAsimvals, Pssm_S0noAsimvals, label = "S0, no AB", lw = 6, ls = "dotted", color = "blue")
ax4b.set_yticks(np.arange(0, 1e6, 250000))
ax4b.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax4b.set_ylabel("$P_{SSM}$ (copies/mL)")
ax4b.set_xlabel("Days after 1st vaccination")
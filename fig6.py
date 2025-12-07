import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from src.model import params_vaccination
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

def z_init_vaccination():
    """
    Modified initial densities used in the extended model, including the activation functions for the GC and BM

    Returns
    -------
    L0 : float
        Initial LNP density.
    S0 : float
        Initial susceptible cell density.
    I0 : float
        Initial infected cell density.
    Pf0 : float
        Initial free-floating protein density.
    Pfdc0 : float
        Initial GC protein immune complex density.
    Pssm0 : float
        Initial BM protein immune complex density.
    BG0 : float
        Initial germinal center B cell density.
    BML0 : float
        Initial low affinity memory B cell density.
    BMH0 : float
        Initial high affinity memory B cell density.
    B1L0 : float
        Initial low affinity short living plasma cell density.
    B1H0 : float
        Initial high affinity short living plasma cell density.
    B20 : float
        Initial long living plasma cell density.
    A10 : float
        Initial low affinity antibody density.
    A20 : float
        Initial high affinity antibody density.
    AM0 : float
        Initial affinity maturation value.
    gGC0 : float
        Initial GC activation value.
    gBML0 : float
        Initial BML activation value.
    gBMH0 : float
        Initial BMH activation value.

    """
    L0 = 2.5e5
    S0 = 1e6
    I0 = 0
    Pf0 = 0
    Pfdc0 = 0
    Pssm0 = 0
    BG0 = 0
    BML0 = 0
    BMH0 = 0
    B1L0 = 0
    B1H0 = 0
    B20 = 0
    A10 = 0
    A20 = 0
    AM0 = 1
    gGC0 = 0
    gBML0 = 0
    gBMH0 = 0
    return (L0, S0, I0, Pf0, Pfdc0, Pssm0, BG0, BML0, BMH0, B1L0, B1H0, B20, A10, A20, AM0, gGC0, gBML0, gBMH0)

def model_vaccination(t, z, params):
    """
    Modified ODE for the extended model that includes the activation functions for the GC and BM

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
            gGC: GC activation function, Eq. A10
            gBML: BML activation function, Eq. A11
            gBMH: BMH activation function, Eq. A11
    params : dict
        Dictionary of parameters.

    Returns
    -------
    z : tuple
        Densities of the cells, antigens, and antibodies solved at time t+dt, where dt is set by the ODE solver.
        Follows the order of the input z.

    """
    
    L, S, I, Pf, Pfdc, Pssm, BG, BML, BMH, B1L, B1H, B2, A1, A2, Abcr, var_gGC, var_gBML, var_gBMH = z
    
    A = (params["Abcrmin"]*A1) + (params["Abcrmax"]*A2)
    
    betaIA1 = params["betaIA"]*params["Abcrmin"]
    betaIA2 = params["betaIA"]*params["Abcrmax"]
    betaPfA1 = params["betaPfA"]*params["Abcrmin"]
    betaPfA2 = params["betaPfA"]*params["Abcrmax"]
    gammaIA1 = params["gammaIA"]*params["Abcrmin"] 
    gammaIA2 = params["gammaIA"]*params["Abcrmax"] 
    gammaPfA1 = params["gammaPfA"]*params["Abcrmin"]
    gammaPfA2 = params["gammaPfA"]*params["Abcrmax"]
    
    gGC=(1 - np.exp(-params["C"] *Pfdc*Abcr/(A+BG+1))) 
    gBML=(1 - np.exp(-params["C"] *Pssm/(A+BML+1)))
    gBMH=(1 - np.exp(-params["C"] *Pssm/( (A1/params["Abcrmax"]) + A2 +BMH+1)))
    
    f = (params["Abcrmax"]-Abcr)/(params["Abcrmax"]-params["Abcrmin"])
    
    dLdt = - (params["kL"]*S*L) - (L/params["tauL"])
    dSdt = - (params["kS"]*S*L) - (params["kAPC"]*S*Pf) + (params["k1"]*(params["S0"]-S))
    dIdt = (params["kS"]*S*L) - (I/params["tauI"]) - (betaIA1*A1*I) - (betaIA2*A2*I)
    dPfdt = (params["pp"]*I) - (params["kP"]*S*Pf) - (Pf/params["tauPf"]) - (betaPfA1*A1*Pf) - (betaPfA2*A2*Pf)
    dPfdcdt = ((params["kP"]/2)*S*Pf) - (Pfdc/params["taufdc"])
    dPssmdt = ((params["kP"]/2)*S*Pf) - (Pssm/params["taussm"])
    dBGdt = (params["cN"]*gGC*params["BN"]) - (params["epsilon"]*BG) - (params["apL"]*(1-gGC)*BG)
    dBMLdt = (params["NGC"]*f*params["epsilon"]*BG) + ((1-params["h"])*params["c1"]*gBML*BML) - (BML/params["tauBM"])
    dBMHdt = (params["NGC"]*params["v"]*(1-f)*params["epsilon"]*BG) + ((1-params["h"])*params["c1"]*gBMH*BMH) - (BMH/params["tauBM"])
    dB1Ldt = (params["h"]*params["c1"]*gBML*BML) - (B1L/params["tauB1"])
    dB1Hdt = (params["h"]*params["c1"]*gBMH*BMH) - (B1H/params["tauB1"])
    dB2dt = (params["NGC"]*(1-params["v"])*(1-f)*params["epsilon"]*BG) - (B2/params["tauB2"])
    dA1dt = (params["pA"]*B1L) - (A1/params["tauA"]) - (gammaIA1*I*A1) - (gammaPfA1*Pf*A1)
    dA2dt = (params["pA"]*B1H) + (params["pA"]*B2) - (A2/params["tauA"]) - (gammaIA2*I*A2) - (gammaPfA2*Pf*A2)
    dAbcrdt = params["epsilon"]*(params["Abcrmin"] - Abcr + ((params["Abcrmax"]-params["Abcrmin"])*gGC) )
    
    dgGCdt = gGC - var_gGC
    dgBMLdt = gBML- var_gBML
    dgBMHdt = gBMH - var_gBMH
    
    z = dLdt, dSdt, dIdt, dPfdt, dPfdcdt, dPssmdt, dBGdt, dBMLdt, dBMHdt, dB1Ldt, dB1Hdt, dB2dt, dA1dt, dA2dt, dAbcrdt, dgGCdt, dgBMLdt, dgBMHdt
    
    return z

def model_vaccination_gA1(t, z, params):
    """
    Modified ODE for the extended model that includes the activation functions for the GC and BM, with only A1 feedback

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
            gGC: GC activation function, Eq. A10
            gBML: BML activation function, Eq. A11
            gBMH: BMH activation function, Eq. A11
    params : dict
        Dictionary of parameters.

    Returns
    -------
    z : tuple
        Densities of the cells, antigens, and antibodies solved at time t+dt, where dt is set by the ODE solver.
        Follows the order of the input z.

    """
    
    L, S, I, Pf, Pfdc, Pssm, BG, BML, BMH, B1L, B1H, B2, A1, A2, Abcr, var_gGC, var_gBML, var_gBMH = z
    
    A = (params["Abcrmin"]*A1) + (params["Abcrmax"]*A2)
    
    betaIA1 = params["betaIA"]*params["Abcrmin"]
    betaIA2 = params["betaIA"]*params["Abcrmax"]
    betaPfA1 = params["betaPfA"]*params["Abcrmin"]
    betaPfA2 = params["betaPfA"]*params["Abcrmax"]
    gammaIA1 = params["gammaIA"]*params["Abcrmin"] 
    gammaPfA1 = params["gammaPfA"]*params["Abcrmin"]
    
    gGC=(1 - np.exp(-params["C"] *Pfdc*Abcr/(A+BG+1))) 
    gBML=(1 - np.exp(-params["C"] *Pssm/(A+BML+1)))
    gBMH=(1 - np.exp(-params["C"] *Pssm/( (A1/params["Abcrmax"]) + A2 +BMH+1)))
    
    f = (params["Abcrmax"]-Abcr)/(params["Abcrmax"]-params["Abcrmin"])
    
    dLdt = - (params["kL"]*S*L) - (L/params["tauL"])
    dSdt = - (params["kS"]*S*L) - (params["kAPC"]*S*Pf) + (params["k1"]*(params["S0"]-S))
    dIdt = (params["kS"]*S*L) - (I/params["tauI"]) - (betaIA1*A1*I) - (betaIA2*A2*I)
    dPfdt = (params["pp"]*I) - (params["kP"]*S*Pf) - (Pf/params["tauPf"]) - (betaPfA1*A1*Pf) - (betaPfA2*A2*Pf)
    dPfdcdt = ((params["kP"]/2)*S*Pf) - (Pfdc/params["taufdc"])
    dPssmdt = ((params["kP"]/2)*S*Pf) - (Pssm/params["taussm"])
    dBGdt = (params["cN"]*gGC*params["BN"]) - (params["epsilon"]*BG) - (params["apL"]*(1-gGC)*BG)
    dBMLdt = (params["NGC"]*f*params["epsilon"]*BG) + ((1-params["h"])*params["c1"]*gBML*BML) - (BML/params["tauBM"])
    dBMHdt = (params["NGC"]*params["v"]*(1-f)*params["epsilon"]*BG) + ((1-params["h"])*params["c1"]*gBMH*BMH) - (BMH/params["tauBM"])
    dB1Ldt = (params["h"]*params["c1"]*gBML*BML) - (B1L/params["tauB1"])
    dB1Hdt = (params["h"]*params["c1"]*gBMH*BMH) - (B1H/params["tauB1"])
    dB2dt = (params["NGC"]*(1-params["v"])*(1-f)*params["epsilon"]*BG) - (B2/params["tauB2"])
    dA1dt = (params["pA"]*B1L) - (A1/params["tauA"]) - (gammaIA1*I*A1) - (gammaPfA1*Pf*A1)
    dA2dt = 0
    dAbcrdt = params["epsilon"]*(params["Abcrmin"] - Abcr + ((params["Abcrmax"]-params["Abcrmin"])*gGC) )
    
    dgGCdt = gGC - var_gGC
    dgBMLdt = gBML- var_gBML
    dgBMHdt = gBMH - var_gBMH
    
    z = dLdt, dSdt, dIdt, dPfdt, dPfdcdt, dPssmdt, dBGdt, dBMLdt, dBMHdt, dB1Ldt, dB1Hdt, dB2dt, dA1dt, dA2dt, dAbcrdt, dgGCdt, dgBMLdt, dgBMHdt
    
    return z

def model_vaccination_gA2(t, z, params):
    """
    Modified ODE for the extended model that includes the activation functions for the GC and BM, with only A2 feedback

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
            gGC: GC activation function, Eq. A10
            gBML: BML activation function, Eq. A11
            gBMH: BMH activation function, Eq. A11
    params : dict
        Dictionary of parameters.

    Returns
    -------
    z : tuple
        Densities of the cells, antigens, and antibodies solved at time t+dt, where dt is set by the ODE solver.
        Follows the order of the input z.

    """
    
    L, S, I, Pf, Pfdc, Pssm, BG, BML, BMH, B1L, B1H, B2, A1, A2, Abcr, var_gGC, var_gBML, var_gBMH = z
    
    A = (params["Abcrmin"]*A1) + (params["Abcrmax"]*A2)
    
    betaIA1 = params["betaIA"]*params["Abcrmin"]
    betaIA2 = params["betaIA"]*params["Abcrmax"]
    betaPfA1 = params["betaPfA"]*params["Abcrmin"]
    betaPfA2 = params["betaPfA"]*params["Abcrmax"]
    gammaIA2 = params["gammaIA"]*params["Abcrmax"]
    gammaPfA2 = params["gammaPfA"]*params["Abcrmax"]
    
    gGC=(1 - np.exp(-params["C"] *Pfdc*Abcr/(A+BG+1))) 
    gBML=(1 - np.exp(-params["C"] *Pssm/(A+BML+1)))
    gBMH=(1 - np.exp(-params["C"] *Pssm/( (A1/params["Abcrmax"]) + A2 +BMH+1)))
    
    f = (params["Abcrmax"]-Abcr)/(params["Abcrmax"]-params["Abcrmin"])
    
    dLdt = - (params["kL"]*S*L) - (L/params["tauL"])
    dSdt = - (params["kS"]*S*L) - (params["kAPC"]*S*Pf) + (params["k1"]*(params["S0"]-S))
    dIdt = (params["kS"]*S*L) - (I/params["tauI"]) - (betaIA1*A1*I) - (betaIA2*A2*I)
    dPfdt = (params["pp"]*I) - (params["kP"]*S*Pf) - (Pf/params["tauPf"]) - (betaPfA1*A1*Pf) - (betaPfA2*A2*Pf)
    dPfdcdt = ((params["kP"]/2)*S*Pf) - (Pfdc/params["taufdc"])
    dPssmdt = ((params["kP"]/2)*S*Pf) - (Pssm/params["taussm"])
    dBGdt = (params["cN"]*gGC*params["BN"]) - (params["epsilon"]*BG) - (params["apL"]*(1-gGC)*BG)
    dBMLdt = (params["NGC"]*f*params["epsilon"]*BG) + ((1-params["h"])*params["c1"]*gBML*BML) - (BML/params["tauBM"])
    dBMHdt = (params["NGC"]*params["v"]*(1-f)*params["epsilon"]*BG) + ((1-params["h"])*params["c1"]*gBMH*BMH) - (BMH/params["tauBM"])
    dB1Ldt = (params["h"]*params["c1"]*gBML*BML) - (B1L/params["tauB1"])
    dB1Hdt = (params["h"]*params["c1"]*gBMH*BMH) - (B1H/params["tauB1"])
    dB2dt = (params["NGC"]*(1-params["v"])*(1-f)*params["epsilon"]*BG) - (B2/params["tauB2"])
    dA1dt = 0
    dA2dt = (params["pA"]*B1H) + (params["pA"]*B2) - (A2/params["tauA"]) - (gammaIA2*I*A2) - (gammaPfA2*Pf*A2)
    dAbcrdt = params["epsilon"]*(params["Abcrmin"] - Abcr + ((params["Abcrmax"]-params["Abcrmin"])*gGC) )
    
    dgGCdt = gGC - var_gGC
    dgBMLdt = gBML- var_gBML
    dgBMHdt = gBMH - var_gBMH
    
    z = dLdt, dSdt, dIdt, dPfdt, dPfdcdt, dPssmdt, dBGdt, dBMLdt, dBMHdt, dB1Ldt, dB1Hdt, dB2dt, dA1dt, dA2dt, dAbcrdt, dgGCdt, dgBMLdt, dgBMHdt
    
    return z

def model_vaccination_gPf(t, z, params):
    """
    Modified ODE for the extended model that includes the activation functions for the GC and BM, with no feedback

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
            gGC: GC activation function, Eq. A10
            gBML: BML activation function, Eq. A11
            gBMH: BMH activation function, Eq. A11
    params : dict
        Dictionary of parameters.

    Returns
    -------
    z : tuple
        Densities of the cells, antigens, and antibodies solved at time t+dt, where dt is set by the ODE solver.
        Follows the order of the input z.

    """
    
    L, S, I, Pf, Pfdc, Pssm, BG, BML, BMH, B1L, B1H, B2, A1, A2, Abcr, var_gGC, var_gBML, var_gBMH = z
    
    A = (params["Abcrmin"]*A1) + (params["Abcrmax"]*A2)
    
    betaIA1 = params["betaIA"]*params["Abcrmin"]
    betaIA2 = params["betaIA"]*params["Abcrmax"]
    betaPfA1 = params["betaPfA"]*params["Abcrmin"]
    betaPfA2 = params["betaPfA"]*params["Abcrmax"]
    
    gGC=(1 - np.exp(-params["C"] *Pfdc*Abcr/(A+BG+1))) 
    gBML=(1 - np.exp(-params["C"] *Pssm/(A+BML+1)))
    gBMH=(1 - np.exp(-params["C"] *Pssm/( (A1/params["Abcrmax"]) + A2 +BMH+1)))
    
    f = (params["Abcrmax"]-Abcr)/(params["Abcrmax"]-params["Abcrmin"])
    
    dLdt = - (params["kL"]*S*L) - (L/params["tauL"])
    dSdt = - (params["kS"]*S*L) - (params["kAPC"]*S*Pf) + (params["k1"]*(params["S0"]-S))
    dIdt = (params["kS"]*S*L) - (I/params["tauI"]) - (betaIA1*A1*I) - (betaIA2*A2*I)
    dPfdt = (params["pp"]*I) - (params["kP"]*S*Pf) - (Pf/params["tauPf"]) - (betaPfA1*A1*Pf) - (betaPfA2*A2*Pf)
    dPfdcdt = ((params["kP"]/2)*S*Pf) - (Pfdc/params["taufdc"])
    dPssmdt = ((params["kP"]/2)*S*Pf) - (Pssm/params["taussm"])
    dBGdt = (params["cN"]*gGC*params["BN"]) - (params["epsilon"]*BG) - (params["apL"]*(1-gGC)*BG)
    dBMLdt = (params["NGC"]*f*params["epsilon"]*BG) + ((1-params["h"])*params["c1"]*gBML*BML) - (BML/params["tauBM"])
    dBMHdt = (params["NGC"]*params["v"]*(1-f)*params["epsilon"]*BG) + ((1-params["h"])*params["c1"]*gBMH*BMH) - (BMH/params["tauBM"])
    dB1Ldt = (params["h"]*params["c1"]*gBML*BML) - (B1L/params["tauB1"])
    dB1Hdt = (params["h"]*params["c1"]*gBMH*BMH) - (B1H/params["tauB1"])
    dB2dt = (params["NGC"]*(1-params["v"])*(1-f)*params["epsilon"]*BG) - (B2/params["tauB2"])
    dA1dt = 0
    dA2dt = 0
    dAbcrdt = params["epsilon"]*(params["Abcrmin"] - Abcr + ((params["Abcrmax"]-params["Abcrmin"])*gGC) )
    
    dgGCdt = gGC - var_gGC
    dgBMLdt = gBML- var_gBML
    dgBMHdt = gBMH - var_gBMH
    
    z = dLdt, dSdt, dIdt, dPfdt, dPfdcdt, dPssmdt, dBGdt, dBMLdt, dBMHdt, dB1Ldt, dB1Hdt, dB2dt, dA1dt, dA2dt, dAbcrdt, dgGCdt, dgBMLdt, dgBMHdt
    
    return z

params = params_vaccination()
z0 = z_init_vaccination()

sol = integrate.solve_ivp(model_vaccination, [0, 300], z0, args = ([params]), method = "BDF", dense_output = True)
gGC_simvals = sol.y[15]
gBMH_simvals = sol.y[17]
t_simvals = sol.t
B1H_simvals = sol.y[10]
BG_simvals = sol.y[6]

sol_gA1 = integrate.solve_ivp(model_vaccination_gA1, [0,300], z0, args = ([params]), method = "BDF", dense_output = True)
gGC_A1_simvals = sol_gA1.y[15]
gBMH_A1_simvals = sol_gA1.y[17]
t_gA1_simvals = sol_gA1.t
B1H_gA1_simvals = sol_gA1.y[10]
BG_gA1_simvals = sol_gA1.y[6]

sol_gA2 = integrate.solve_ivp(model_vaccination_gA2, [0,300], z0, args = ([params]), method = "BDF", dense_output = True)
gGC_A2_simvals = sol_gA2.y[15]
gBMH_A2_simvals = sol_gA2.y[17]
t_gA2_simvals = sol_gA2.t
B1H_gA2_simvals = sol_gA2.y[10]
BG_gA2_simvals = sol_gA2.y[6]

sol_gPf = integrate.solve_ivp(model_vaccination_gPf, [0,300], z0, args = ([params]), method = "BDF", dense_output = True)
gGC_Pf_simvals = sol_gPf.y[15]
gBMH_Pf_simvals = sol_gPf.y[17]
t_gPf_simvals = sol_gPf.t
B1H_gPf_simvals = sol_gPf.y[10]
BG_gPf_simvals = sol_gPf.y[6]

fig1, (ax1a, ax1b) = plt.subplots(nrows = 2, sharex = True)
ax1a.plot(t_simvals, B1H_simvals, label = "$g_{BMH}$", lw = 3, color = "black")
ax1a.plot(t_gA1_simvals, B1H_gA1_simvals, label = "$g_{BMH}(A_1)$", lw = 3, color = "green")
ax1a.plot(t_gA2_simvals, B1H_gA2_simvals, label = "$g_{BMH}(A_2)$", lw = 3, color = "blue")
ax1a.plot(t_gPf_simvals, B1H_gPf_simvals, label = "$g_{BMH}(P_f)$", lw = 3, color = "red")
ax1b.plot(t_simvals, gBMH_simvals, label = "$g_{BMH}$", lw = 3, color = "black")
ax1b.plot(t_gA1_simvals, gBMH_A1_simvals, label = "$g_{BMH}(A_1)$", ls = "dashed", lw = 3, color = "green")
ax1b.plot(t_gA2_simvals, gBMH_A2_simvals, label = "$g_{BMH}(A_2)$", ls = "dashed", lw = 3, color = "blue")
ax1b.plot(t_gPf_simvals, gBMH_Pf_simvals, label = "$g_{BMH}(P_f)$", ls = "dashed", lw = 3, color = "red")

ax1a.set_xlim(right = 110)
ax1b.set_xlim(right = 110)
ax1a.set_ylabel("$B_{1H}$ (cells/mL)")
ax1b.set_ylabel("$g_{BMH}$")
ax1b.set_xlabel("Days after 1st vaccination")
ax1a.legend()

fig2, (ax2a, ax2b) = plt.subplots(nrows = 2, sharex = True)
ax2a.plot(t_simvals, BG_simvals, label = "$g_{BGC}$", lw = 3, color = "black")
ax2a.plot(t_gA1_simvals, BG_gA1_simvals, label = "$g_{BGC}(A_1)$", lw = 3, color = "green")
ax2a.plot(t_gA2_simvals, BG_gA2_simvals, label = "$g_{BGC}(A_2)$", lw = 3, color = "blue")
ax2a.plot(t_gPf_simvals, BG_gPf_simvals, label = "$g_{BGC}(P_f)$", lw = 3, color = "red")
ax2b.plot(t_simvals, gGC_simvals, label = "$g_{BGC}$", lw = 3, color = "black")
ax2b.plot(t_gA1_simvals, gGC_A1_simvals, label = "$g_{BGC}(A_1)$", ls = "dashed", lw = 3, color = "green")
ax2b.plot(t_gA2_simvals, gGC_A2_simvals, label = "$g_{BGC}(A_2)$", ls = "dashed", lw = 3, color = "blue")
ax2b.plot(t_gPf_simvals, gGC_Pf_simvals, label = "$g_{BGC}(P_f)$", ls = "dashed", lw = 3, color = "red")
ax2a.set_ylabel("$B_{GC}$ (cells/mL)")
ax2b.set_ylabel("$g_{GC}$")
ax2b.set_xlabel("Days after 1st vaccination")
ax2a.legend()
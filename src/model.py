import numpy as np

def model_vaccination(t, z, params):
    """
    Base ODE for the extended model that considers the vaccination, promary response, and secondary response

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
    
    L, S, I, Pf, Pfdc, Pssm, BG, BML, BMH, B1L, B1H, B2, A1, A2, Abcr = z
    
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
    dBMLdt = (f*params["epsilon"]*BG) + ((1-params["h"])*params["c1"]*gBML*BML) - (BML/params["tauBM"])
    dBMHdt = (params["v"]*(1-f)*params["epsilon"]*BG) + ((1-params["h"])*params["c1"]*gBMH*BMH) - (BMH/params["tauBM"])
    dB1Ldt = (params["h"]*params["c1"]*gBML*BML) - (B1L/params["tauB1"])
    dB1Hdt = (params["h"]*params["c1"]*gBMH*BMH) - (B1H/params["tauB1"])
    dB2dt = ((1-params["v"])*(1-f)*params["epsilon"]*BG) - (B2/params["tauB2"])
    dA1dt = (params["pA"]*B1L) - (A1/params["tauA"]) - (gammaIA1*I*A1) - (gammaPfA1*Pf*A1)
    dA2dt = (params["pA"]*B1H) + (params["pA"]*B2) - (A2/params["tauA"]) - (gammaIA2*I*A2) - (gammaPfA2*Pf*A2)
    dAbcrdt = params["epsilon"]*(params["Abcrmin"] - Abcr + ((params["Abcrmax"]-params["Abcrmin"])*gGC) )
    
    z = dLdt, dSdt, dIdt, dPfdt, dPfdcdt, dPssmdt, dBGdt, dBMLdt, dBMHdt, dB1Ldt, dB1Hdt, dB2dt, dA1dt, dA2dt, dAbcrdt
    
    return z

def z_init_vaccination():
    """
    Initial densities used in the extended model

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
    return (L0, S0, I0, Pf0, Pfdc0, Pssm0, BG0, BML0, BMH0, B1L0, B1H0, B20, A10, A20, AM0)

def params_vaccination():
    """
    Returns a dictionary of the input parameters
    
    Returns
    -------
    params : dict
        Dictionary of model parameters
    """
    
    params = {"kL": 2e-7, # Absorption rate of LNP by S (ml/(day*cell)) 
              "tauL": 7, # LNP lifetime (days)
              "kS": 2e-7, # Infection rate of S by LNP (ml/(day*copy))
              "kAPC": 1e-7, # Activation of S per Pi
              "k1": 1/14, # S recruitment (1/day)
              "S0": 1e6, # Constant S recruitment
              "tauI": 4, #I lifetime (days)
              "pp": 25, # Pseudo-antigen production rate (copies/(day*cell))
              "tauPf": 2, # Pf lifetime (days)
              "taufdc": 20, # Pfdc lifetime (days)
              "taussm": 2, # Pssm lifetime (days)
              "kP": 1e-7, # Absorption rate of Pf by S (ml/(day*cell))
              "cN": 1e-3, # Maximum BN recruitment rate into GC (1/day)
              "BN": 5e4, # BN pool size
              "epsilon": 0.05, # Rate of BG leaving GC (1/day)
              "apL": 0.5, # Light zone apoptosis (1/day)
              "NGC": 1, # Number of GC
              "tauBM": 18, # BM lifetime (days)
              "c1": 0.6, # BM amplification (1/days)
              "tauB1": 5, # B1 lifetime (days)
              "h": 0.6, # Fraction of BM that specify into B1
              "tauB2": 180, # B2 lifetime (days)
              "v": 0.9, # Fraction of high affinity BG to specify to BMH
              "pA": 2, # Antibody production rate (ng/(day*cell))
              "tauA": 60, # Antibody lifetime
              "Abcrmin": 1, # Low BCR affinity
              "Abcrmax": 25, # High BCR affinity
              "C": 0.1, # Signal activation by BCR (cells/copy)
              "betaIA": 5e-5, # Rate of infected cell neutralization (ml/day/cell)
              "betaPfA": 5e-5, # Rate of pseudo antigen neutralization (ml/day/cell)
              "gammaIA": 5e-5*2.5, # Rate of antibody consumption by I
              "gammaPfA": 5e-5*5e-4, # Rate of antibody consumption by Pf
              'betaVA': 5e-5, # Rate of virus neutralization
              'tauV': 2, # Virus lifetime
              'kV': 1e-7, # Viral infection rate
              'pV': 2500, # Viral proliferation rate
              'tauIV': 0.5 # Lifetime of cell infected by virus
              }
    return params

def Athresh_val(params):
    """
    Returns the critical antibody titer value Tc, Eq. A32

    Parameters
    ----------
    params : dict
        Dictionary of input parameters.

    Returns
    -------
    thresh : float
        Critical antibody titer value Tc.

    """
    
    x = ((1/params["betaIA"])*(1/params["tauIV"])) + ((1/params["betaVA"])*((1/params["tauV"])+(params["kV"]*params["S0"])))
    y = (((1/params["tauV"])+(params["kV"]*params["S0"]))*(1/params["tauIV"])-(params["pV"]*params["kV"]*params["S0"])) * (1/params["betaVA"]) * (1/params["betaIA"])

    thresh = -(x/2) + np.sqrt((x**2)/4 - y)
    return thresh

def Athresh_event(t,z,params):
    """
    Returns the event of crossing the antibody titer value Tc, used in the ODE solver.

    Parameters
    ----------
    t : float
        Time point at which to solve the ODE.
    z : tuple
        Density of the cells, antigens, and antibodies to be solved at time t. Follows this order:
            L: LNPs,
            S: Susceptible cells,
            I: Infected cells,
            Pf: Free floating protein,
            Pfdc: Protein immune complex presented to the GC
            Pssm: Protein immune complex presented to the memory B cell
            BG: Germinal center B cells
            BML: Low affinity memory B cells
            BMH: High affinity memory B cells
            B1L: Low affinity short living plasma cells
            B1H: High affinity short living plasma cells
            B2: Long living plasma cells
            A1: Low affinity antibodies
            A2: High affinity antibodies
            Abcr: Affinity maturation
    params : dict
        Dictionary of parameters.

    Returns
    -------
    float
        Returns the event of crossing the critical antibody titer.

    """
    
    L, S, I, Pf, Pfdc, Pssm, BG, BML, BMH, B1L, B1H, B2, A1, A2, AM = z
    A = (A1*params["Abcrmin"]) + (A2*params["Abcrmax"])
    thresh = Athresh_val(params)
    
    return A - thresh
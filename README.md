# Modelling variability of the immunity build-up and waning following RNA-based vaccination
**Juan Magalang , Tyll Krueger , Joerg Galle**

This repository contains the source code for generating the figures in the article[^1]. Furthermore, this repository contains the main ODE model found in the text (found in `src/model.py`), along with the parameters found in Tables A1 to A3.

The documentation of each function in `model.py` is included as docstrings and can be accessed by using `help(name_of_function)`.  This README provides a sample usage of the functions within `model.py`.

The functions within `model.py` are as follows:

 1. `model_vaccination`: ODE model as defined in Eqs 1, 2 ,3#, 4#, 5A, 5B, 6*, 8#, 9#, 10#, 11, 12, 13,14#, and 15#
 2. `z_init_vaccination`: Initial populations of the compartments within the ODE
 3. `params_vaccination`: Parameter dictionary as defined in Tables A1, A2, A3
 4. `Athresh_val`: Critical antibody titer $T_c$
 5. `Athresh_event`: `scipy.integrate` event that tracks the moment at which antibody titer crosses $T_c$

The compartments and equations within `model.py` are contained in tuples, and will follow the order:
 1. `L`: LNPs, Eq. 1
 2. `S`: Susceptible cells, Eq. 2
 3. `I`: Infected cells, Eq. 3#
 4. `Pf`: Free floating protein, Eq. 4#
 5. `Pfdc`: Protein immune complex presented to the GC, Eq. 5A
 6.  `Pssm`: Protein immune complex presented to the memory B cell, Eq. 5B
 7. `BG`: Germinal center B cells, Eq. 6*
 8. `BML`: Low affinity memory B cells, Eq. 9#
 9. `BMH`: High affinity memory B cells, Eq. 10#
 10. `B1L`: Low affinity short living plasma cells, Eq. 12
 11. `B1H`: High affinity short living plasma cells, Eq. 13
 12. `B2`: Long living plasma cells, Eq. 11
 13. `A1`: Low affinity antibodies, Eq. 14#
 14. `A2`: High affinity antibodies, Eq. 15#
 15. `Abcr`: Affinity maturation, Eq. 8#

## Integrating the ODE
We first import the necessary python packages for integrating the ODE.
```
import scipy.integrate as integrate
```
Suppose we would like to run the model from time `t=0` to `t=15` days:
```
t0 = 0
tf = 15
```
The parameters of the model and initial populations are contained in special functions in `model.py`.
```
params = params_vaccination() # yields a dictionary of parameters found in Tables A1, A2, A3, and A5
z0 = z_init_vaccination() # yields a tuple of initial values found in Table A4
```
The function `model_vaccination` contains the ODEs that describe the extended model. We can see how the ODE evolves from `t0` to `tf` using `scipy.integrate`.

```
sol = integrate.solve_ivp(model_vaccination, [t0, tf], z0, args = ([params]), method = "BDF", dense_output = True)
```
Finally, we obtain the trajectory arrays of each compartment within the ODE by doing:
```
t = sol.t # Time

L = sol.y[0] # LNPs
S = sol.y[1] # Susceptible cells
I = sol.y[2] # Infected cells
Pf = sol.y[3] # Free floating protein,
Pfdc = sol.y[4] # Protein immune complex presented to the GC
Pssm = sol.y[5] # Protein immune complex presented to the memory B cell
BG = sol.y[6] # Germinal center B cells
BML = sol.y[7] # Low affinity memory B cells
BMH = sol.y[8] # High affinity memory B cells
B1L = sol.y[9] # Low affinity short living plasma cells
B1H = sol.y[10] # High affinity short living plasma cells
B2 = sol.y[11] # Long living plasma cells
A1 = sol.y[12] # Low affinity antibodies
A2 = sol.y[13] # High affinity antibodies
Abcr = sol.y[14] # Affinity maturation
A = params["Abcrmin"]*A1 + params["Abcrmax"]*A2 # Antibody titer
```

## Performing a second vaccination

The paper extensively describes the dynamics of the immune system following a second vaccination. This section will describe how it is performed within the script.

We first use the same imports and initializations as the previous section.

```
import scipy.integrate as integrate
import numpy as np

t0 = 0
tf = 15

params = params_vaccination()
z0 = z_init_vaccination()
```

Suppose we have a second vaccination time at `t = 10` days,

```
ts = 10
```

We first solve the ODE from `t0` up to `ts`:
```
sol1 = integrate.solve_ivp(model_vaccination, [t0, ts], z0, args = ([params]), method = "BDF", dense_output = True)
```

A second vaccination is modeled by reintroducing LNPs back into the system equal to the initial LNP concentration.

```
z1 = sol1.y[:,-1].copy() # Take the state of the system at time t=ts

# The first element of second_vacc_z and z0 corresponds to LNPs
# Add L0 to the LNPs of the system at t=ts to simulate a second vaccination
z1[0] += z0[0]
```
We can now run the integration again from `ts` up to `tf` with `z1` as the initial population
```
sol2 = integrate.solve_ivp(model_vaccination, [ts,tf], z1, args = ([params]), method = "BDF", dense_output = True)
```
To get the final trajectories of each compartment with a second vaccination, one can concatenate the solutions from the first and second integrations.

```
t = np.concatenate((sol1.t, sol2.t))

L = np.concatenate((sol1.y[0], sol2.y[0]))
S = np.concatenate((sol1.y[1], sol2.y[1]))
I = np.concatenate((sol1.y[2], sol2.y[2]))
Pf = np.concatenate((sol1.y[3], sol2.y[3]))
Pfdc = np.concatenate((sol1.y[4], sol2.y[4]))
Pssm = np.concatenate((sol1.y[5], sol2.y[5]))
BG = np.concatenate((sol1.y[6], sol2.y[6]))
BML = np.concatenate((sol1.y[7], sol2.y[7]))
BMH = np.concatenate((sol1.y[8], sol2.y[8]))
B1L = np.concatenate((sol1.y[9], sol2.y[9]))
B1H = np.concatenate((sol1.y[10], sol2.y[10]))
B2 = np.concatenate((sol1.y[11], sol2.y[11]))
A1 = np.concatenate((sol1.y[12], sol2.y[12]))
A2 = np.concatenate((sol1.y[13], sol2.y[13]))
Abcr = np.concatenate((sol1.y[14], sol2b.y[14]))
A = params["Abcrmin"]*A1 + params["Abcrmax"]*A2
```

## Calculating the protection time
The protection time is defined as the time at which the antibody titer is above critical value $T_c$ (Eq. A32 in the text). This section will show how the protection time is calculated using the functions in `model.py`. We use the same script as the previous section, before the integration of the ODE model after the second vaccination.

```
import scipy.integrate as integrate
import numpy as np

t0 = 0
tf = 15

params = params_vaccination()
z0 = z_init_vaccination()

ts = 10

sol1 = integrate.solve_ivp(model_vaccination, [t0, ts], z0, args = ([params]), method = "BDF", dense_output = True)

z1 = sol1.y[:,-1].copy()

z1[0] += z0[0]
```

We use the `events` feature within `scipy.integrate` to track at which value the antibody titer goes below $T_c$. The functions `Athresh_val` and `Athresh_event` is used as an argument within `scipy.integrate` to track this event.

```
sol2 = integrate.solve_ivp(model_vaccination, [ts,tf], z1, args = ([params]), method = "BDF", dense_output = True, events = [Athresh_event])
```

Taking `sol2.t_events` gives an array of times at which the antibody titer has crossed $T_c$. We assume that the antibody titer will start below the critical threshold, and while it evolves over time, it will cross the threshold upwards once and downwards once. Hence, taking the difference of this array will give the protection time.

```
prot_time = sol2.t_events[0][-1]-sol2.t_events[0][-2]
```

## Calculating the protection amplification
Protection amplification is defined as the quotient between the maximum antibody titer of the case with a second vaccination and without a second vaccination. This quantity is used extensively to generate Figure A3 in the main text.

$$
T_{\text{amp}} = \frac{\max(T_{\text{double vacc}})}{\max(T_{\text{single vacc}})}
$$

We begin by integrating the ODE for a case with only a single vaccination:

```
import scipy.integrate as integrate
import numpy as np

t0 = 0
tf = 15

params = params_vaccination()
z0 = z_init_vaccination()

ts = 10

sol0 = integrate.solve_ivp(model_vaccination, [t0, tf], z0, args = ([params]), method = "BDF", dense_output = True)
```

Then we compute for the maximum antibody titer for this case

```
Tmax0 = max((sol0.y[11]*params["Amin"]) + (sol0.y[11]*par["Amax"]))
```

Afterwards, we integrate the ODE for a case with a second vaccination.

```
sol1 = integrate.solve_ivp(model_vaccination, [t0, ts], z0, args = ([params]), method = "BDF", dense_output = True)

z1 = sol1.y[:,-1].copy()

z1[0] += z0[0]

sol2 = integrate.solve_ivp(model_vaccination, [ts,tf], z1, args = ([params]), method = "BDF", dense_output = True)
```

Then, we combine the titers from both solutions and take the maximum

```
A1 = np.concatenate((sol1.y[12], sol2.y[12]))
A2 = np.concatenate((sol1.y[13], sol2.y[13]))
Tmax = max(params["Abcrmin"]*A1 + params["Abcrmax"]*A2)
```
We can now compute for the protection amplification
```
prot_amp = Tmax/Tmax0
```

[^1]: [Modelling variability of the immunity build-up and waning following RNA-based vaccination](https://arxiv.org/abs/2511.05130)
> Written with [StackEdit](https://stackedit.io/).

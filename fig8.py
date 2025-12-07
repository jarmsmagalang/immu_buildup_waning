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

z0 = z_init_vaccination()

params_a = params_vaccination()
params_b = params_vaccination()
params_b["tauB1"] = 5/2
params_c = params_vaccination()
params_c["tauB1"] = 1/2

second_vaccination_time = 21
t_end = 300

sol1_a = integrate.solve_ivp(model_vaccination, [0,second_vaccination_time], z0, args = ([params_a]), method = "BDF", dense_output = True)
second_vacc_z_a = sol1_a.y[:,-1].copy()
second_vacc_z_a[0] += z0[0]
sol2_a = integrate.solve_ivp(model_vaccination, [second_vaccination_time,t_end], second_vacc_z_a, args = ([params_a]), method = "BDF", dense_output = True, events = [Athresh_event])
t_sim_a = np.concatenate((sol1_a.t, sol2_a.t))
B1H_sim_a = np.concatenate((sol1_a.y[10], sol2_a.y[10]))
B2_sim_a = np.concatenate((sol1_a.y[11], sol2_a.y[11]))
A1_sim_a = np.concatenate((sol1_a.y[12], sol2_a.y[12]))
A2_sim_a = np.concatenate((sol1_a.y[13], sol2_a.y[13]))
A_sim_a = (params_a["Abcrmin"]*A1_sim_a) + (params_a["Abcrmax"]*A2_sim_a)

sol1_b = integrate.solve_ivp(model_vaccination, [0,second_vaccination_time], z0, args = ([params_b]), method = "BDF", dense_output = True)
second_vacc_z_b = sol1_b.y[:,-1].copy()
second_vacc_z_b[0] += z0[0]
sol2_b = integrate.solve_ivp(model_vaccination, [second_vaccination_time,t_end], second_vacc_z_b, args = ([params_b]), method = "BDF", dense_output = True, events = [Athresh_event])
t_sim_b = np.concatenate((sol1_b.t, sol2_b.t))
B1H_sim_b = np.concatenate((sol1_b.y[10], sol2_b.y[10]))
B2_sim_b = np.concatenate((sol1_b.y[11], sol2_b.y[11]))
A1_sim_b = np.concatenate((sol1_b.y[12], sol2_b.y[12]))
A2_sim_b = np.concatenate((sol1_b.y[13], sol2_b.y[13]))
A_sim_b = (params_b["Abcrmin"]*A1_sim_b) + (params_b["Abcrmax"]*A2_sim_b)

sol1_c = integrate.solve_ivp(model_vaccination, [0,second_vaccination_time], z0, args = ([params_c]), method = "BDF", dense_output = True)
second_vacc_z_c = sol1_c.y[:,-1].copy()
second_vacc_z_c[0] += z0[0]
sol2_c = integrate.solve_ivp(model_vaccination, [second_vaccination_time,t_end], second_vacc_z_c, args = ([params_c]), method = "BDF", dense_output = True, events = [Athresh_event])
t_sim_c = np.concatenate((sol1_c.t, sol2_c.t))
B1H_sim_c = np.concatenate((sol1_c.y[10], sol2_c.y[10]))
B2_sim_c = np.concatenate((sol1_c.y[11], sol2_c.y[11]))
A1_sim_c = np.concatenate((sol1_c.y[12], sol2_c.y[12]))
A2_sim_c = np.concatenate((sol1_c.y[13], sol2_c.y[13]))
A_sim_c = (params_c["Abcrmin"]*A1_sim_c) + (params_c["Abcrmax"]*A2_sim_c)

fig1, (ax1a, ax1b) = plt.subplots(nrows = 2, sharex = True)
ax1a.plot(t_sim_a, B1H_sim_a, label = "$\\tau_{B1} = 120$ hr.", lw = 3, color = "black")
ax1a.plot(t_sim_b, B1H_sim_b, label = "$\\tau_{B1} = 60$ hr.", lw = 3, color = "red")
ax1a.plot(t_sim_c, B1H_sim_c, label = "$\\tau_{B1} = 12$ hr.", lw = 3, color = "green")
ax1b.plot(t_sim_a, B2_sim_a, label = "$\\tau_{B1} = 120$ hr.", lw = 3, color = "black")
ax1b.plot(t_sim_b, B2_sim_b, label = "$\\tau_{B1} = 60$ hr.", lw = 3, color = "red")
ax1b.plot(t_sim_c, B2_sim_c, label = "$\\tau_{B1} = 12$ hr.", lw = 3, color = "green")

ax1a.set_xlim(-5,100)
ax1b.set_xlim(-5,100)
ax1a.set_ylabel("$B_{1H}$ (copies/mL)")
ax1b.set_ylabel("$B_{2}$ (copies/mL)")
ax1b.set_xlabel("Days after 1st vaccination")
ax1a.legend()

fig2, ax2 = plt.subplots()
ax2.plot(t_sim_a, A_sim_a, label = "$\\tau_{B1} = 120$ hr.", lw = 3, color = "black")
ax2.plot(t_sim_b, A_sim_b, label = "$\\tau_{B1} = 60$ hr.", lw = 3, color = "red")
ax2.plot(t_sim_c, A_sim_c, label = "$\\tau_{B1} = 12$ hr.", lw = 3, color = "green")
ax2.axhline(Athresh_val(params_vaccination()), lw = 3, ls = "dashed", color = "black")
ax2.set_ylim(top = 6.2e5)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
ax2.set_xlabel("Days after 1st vaccination")
ax2.set_ylabel("$T$ (ng/mL)")
ax2.legend(loc = "upper right")
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from immuwane.model import params_vaccination, z_init_vaccination, Athresh_event, model_vaccination
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

params = params_vaccination()
z0 = z_init_vaccination()
L0 = z0[0]

end_time = 250
sol1 = integrate.solve_ivp(model_vaccination, [0, end_time], z0, args = ([params]), method = "BDF", dense_output = True)

t1 = sol1.t

L1 = sol1.y[0]
S1 = sol1.y[1]
I1 = sol1.y[2]
Pf1 = sol1.y[3]
Pfdc1 = sol1.y[4]
Pssm1 = sol1.y[5]
BG1 = sol1.y[6]
BML1 = sol1.y[7]
BMH1 = sol1.y[8]
B1L1 = sol1.y[9]
B1H1 = sol1.y[10]
B21 = sol1.y[11]
A11 = sol1.y[12]
A21 = sol1.y[13]
Abcr1 = sol1.y[14]
A1 = params["Abcrmin"]*A11 + params["Abcrmax"]*A21

second_vaccination_time = 21
sol2a = integrate.solve_ivp(model_vaccination, [0, second_vaccination_time], z0, args = ([params]), method = "BDF", dense_output = True)
second_vacc_z = sol2a.y[:,-1].copy()
second_vacc_z[0] += L0
sol2b = integrate.solve_ivp(model_vaccination, [second_vaccination_time,end_time], second_vacc_z, args = ([params]), method = "BDF", dense_output = True, events = [Athresh_event])

t2 = np.concatenate((sol2a.t, sol2b.t))

L2 = np.concatenate((sol2a.y[0], sol2b.y[0]))
S2 = np.concatenate((sol2a.y[1], sol2b.y[1]))
I2 = np.concatenate((sol2a.y[2], sol2b.y[2]))
Pf2 = np.concatenate((sol2a.y[3], sol2b.y[3]))
Pfdc2 = np.concatenate((sol2a.y[4], sol2b.y[4]))
Pssm2 = np.concatenate((sol2a.y[5], sol2b.y[5]))
BG2 = np.concatenate((sol2a.y[6], sol2b.y[6]))
BML2 = np.concatenate((sol2a.y[7], sol2b.y[7]))
BMH2 = np.concatenate((sol2a.y[8], sol2b.y[8]))
B1L2 = np.concatenate((sol2a.y[9], sol2b.y[9]))
B1H2 = np.concatenate((sol2a.y[10], sol2b.y[10]))
B22 = np.concatenate((sol2a.y[11], sol2b.y[11]))
A12 = np.concatenate((sol2a.y[12], sol2b.y[12]))
A22 = np.concatenate((sol2a.y[13], sol2b.y[13]))
Abcr2 = np.concatenate((sol2a.y[14], sol2b.y[14]))
A2 = params["Abcrmin"]*A12 + params["Abcrmax"]*A22

figsizes = (5,3)

fig1, ax1 = plt.subplots(figsize = figsizes) 
ax1.plot(t1, L1, lw = 5, ls = "dotted", color = "red")
ax1.plot(t2, L2, lw = 3, color = "black")
ax1.set_xlim(right = 100)
ax1.set_yticks(np.arange(0, 3e5, 1.2e5))
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("$L$ (copies/mL)")
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig2, ax2 = plt.subplots(figsize = figsizes) 
ax2.plot(t1, S1, lw = 5, ls = "dotted", color = "red")
ax2.plot(t2, S2, lw = 3, color = "black")
ax2.set_xlim(right = 100)
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("$S$ (cells/mL)")
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig3, ax3 = plt.subplots(figsize = figsizes) 
ax3.plot(t1, I1, lw = 5, ls = "dotted", color = "red")
ax3.plot(t2, I2, lw = 3, color = "black")
ax3.set_xlim(right = 100)
ax3.set_yticks(np.arange(0, 6e4, 2.5e4))
ax3.set_xlabel("Time (days)")
ax3.set_ylabel("$I$ (cells/mL)")
ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig4, ax4 = plt.subplots(figsize = figsizes) 
ax4.plot(t1, Pf1, lw = 5, ls = "dotted", color = "red")
ax4.plot(t2, Pf2, lw = 3, color = "black")
ax4.set_xlim(right = 100)
ax4.set_yticks(np.arange(0, 3e6, 1.2e6))
ax4.set_xlabel("Time (days)")
ax4.set_ylabel("$P_F$ (copies/mL)")
ax4.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig5, ax5 = plt.subplots(figsize = figsizes) 
ax5.plot(t1, Pfdc1, lw = 5, ls = "dotted", color = "red")
ax5.plot(t2, Pfdc2, lw = 3, color = "black")
ax5.set_xlim(right = 100)
ax5.set_ylim(0,4e5)
ax5.set_yticks(np.arange(0, 4e5, 1.5e5))
ax5.set_xlabel("Time (days)")
ax5.set_ylabel("$P_{FDC}$ (copies/mL)")
ax5.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig6, ax6 = plt.subplots(figsize = figsizes) 
ax6.plot(t1, Pssm1, lw = 5, ls = "dotted", color = "red")
ax6.plot(t2, Pssm2, lw = 3, color = "black")
ax6.set_xlim(right = 100)
ax6.set_ylim(0,4e5)
ax6.set_yticks(np.arange(0, 4e5, 1.5e5))
ax6.set_xlabel("Time (days)")
ax6.set_ylabel("$P_{SSM}$ (copies/mL)")
ax6.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig7, ax7 = plt.subplots(figsize = figsizes) 
ax7.plot(t1, BG1, lw = 5, ls = "dotted", color = "red")
ax7.plot(t2, BG2, lw = 3, color = "black")
ax7.set_xlim(right = 100)
ax7.set_xlabel("Time (days)")
ax7.set_ylabel("$B_G$ (cells/mL)")

fig8, ax8 = plt.subplots(figsize = figsizes) 
ax8.plot(t1, BML1, lw = 5, ls = "dotted", color = "red")
ax8.plot(t2, BML2, lw = 3, color = "black")
ax8.set_xlim(right = 100)
ax8.set_yticks(np.arange(0, 3e3, 1.2e3))
ax8.set_xlabel("Time (days)")
ax8.set_ylabel("$B_{ML}$ (cells/mL)")
ax8.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig9, ax9 = plt.subplots(figsize = figsizes) 
ax9.plot(t1, BMH1, lw = 5, ls = "dotted", color = "red")
ax9.plot(t2, BMH2, lw = 3, color = "black")
ax9.set_xlim(right = 100)
ax9.set_yticks(np.arange(0, 5e3, 1.5e3))
ax9.set_xlabel("Time (days)")
ax9.set_ylabel("$B_{MH}$ (cells/mL)")
ax9.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig10, ax10 = plt.subplots(figsize = figsizes) 
ax10.plot(t1, B1L1, lw = 5, ls = "dotted", color = "red")
ax10.plot(t2, B1L2, lw = 3, color = "black")
ax10.set_xlim(right = 100)
ax10.set_yticks(np.arange(0, 2e3, 8e2))
ax10.set_xlabel("Time (days)")
ax10.set_ylabel("$B_{1L}$ (cells/mL)")
ax10.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig11, ax11 = plt.subplots(figsize = figsizes) 
ax11.plot(t1, B1H1, lw = 5, ls = "dotted", color = "red")
ax11.plot(t2, B1H2, lw = 3, color = "black")
ax11.set_xlim(right = 100)
ax11.set_yticks(np.arange(0, 4e3, 1.2e3))
ax11.set_xlabel("Time (days)")
ax11.set_ylabel("$B_{1H}$ (cells/mL)")
ax11.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig12, ax12 = plt.subplots(figsize = figsizes) 
ax12.plot(t1, B21, lw = 5, ls = "dotted", color = "red")
ax12.plot(t2, B22, lw = 3, color = "black")
ax12.set_xlim(right = 150)
ax12.set_xlabel("Time (days)")
ax12.set_ylabel("$B_{2}$ (cells/mL)")

fig13, ax13 = plt.subplots(figsize = figsizes) 
ax13.plot(t1, A11, lw = 5, ls = "dotted", color = "red")
ax13.plot(t2, A12, lw = 3, color = "black")
ax13.set_xlabel("Time (days)")
ax13.set_ylabel("$A_{1}$ (copies/mL)")
ax13.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig14, ax14 = plt.subplots(figsize = figsizes) 
ax14.plot(t1, A21, lw = 5, ls = "dotted", color = "red")
ax14.plot(t2, A22, lw = 3, color = "black")
ax14.set_yticks(np.arange(0, 3e4, 1.2e4))
ax14.set_xlabel("Time (days)")
ax14.set_ylabel("$A_{2}$ (copies/mL)")
ax14.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig15, ax15 = plt.subplots(figsize = figsizes) 
ax15.plot(t1, Abcr1, lw = 5, ls = "dotted", color = "red")
ax15.plot(t2, Abcr2, lw = 3, color = "black")
ax15.set_xlim(right = 150)
ax15.set_xlabel("Time (days)")
ax15.set_ylabel("$A_{BCR}$")
ax15.set_title("BCR Affinity $A_{BCR}$")

fig16, ax16 = plt.subplots(figsize = figsizes) 
ax16.plot(t1, A1, lw = 5, ls = "dotted", color = "red")
ax16.plot(t2, A2, lw = 3, color = "black")
ax16.set_ylim(top = 6e5)
ax16.set_yticks(np.arange(0, 6.5e5, 1.5e5))
ax16.set_xlabel("Time (days)")
ax16.set_ylabel("$T$ (ng/mL)")
ax16.set_title("Antibody Titer $T$")
ax16.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
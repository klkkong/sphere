#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'})

import os
import numpy
import sphere
from permeabilitycalculator import *
import matplotlib.pyplot as plt

dp_list = numpy.array([1.0e3, 2.0e3, 4.0e3, 10.0e3, 20.0e3, 40.0e3])
cvals = [1.0, 0.1, 0.01]
c_phi = 1.0

K = [[], [], []]
dpdz = [[], [], []]
Q = [[], [], []]
phi_bar = [[], [], []]


c = 0
for c_grad_p in cvals:

    sids = []
    for dp in dp_list:
        sids.append('permeability-dp=' + str(dp) + '-c_phi=' + \
                str(c_phi) + '-c_grad_p=' + str(c_grad_p))

    K[c] = numpy.empty(len(sids))
    dpdz[c] = numpy.empty_like(K)
    Q[c] = numpy.empty_like(K)
    phi_bar[c] = numpy.empty_like(K)
    i = 0

    for sid in sids:
        pc = PermeabilityCalc(sid, plot_evolution=False, print_results=False,
                verbose=False)
        K[c][i] = pc.conductivity()
        pc.findPressureGradient()
        pc.findCrossSectionalFlux()
        dpdz[c][i] = pc.dPdL[2]
        Q[c][i] = pc.Q[2]
        pc.findMeanPorosity()
        phi_bar[c][i] = pc.phi_bar

        i += 1

    # produce VTK files
    #for sid in sids:
        #sim = sphere.sim(sid, fluid=True)
        #sim.writeVTKall()
    c += 1

fig = plt.figure()

#plt.subplot(3,1,1)
plt.xlabel('Pressure gradient $\\Delta p/\\Delta z$ [kPa m$^{-1}$]')
plt.ylabel('Hydraulic conductivity $K$ [ms$^{-1}$]')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
for c in range(len(c_vals)):
    dpdz /= 1000.0
    plt.plot(dpdz[c], K[c], 'o-k', label='$c$ = %.2f' % (c_vals[c]))
plt.grid()

#plt.subplot(3,1,2)
#plt.xlabel('Pressure gradient $\\Delta p/\\Delta z$ [Pa m$^{-1}$]')
#plt.ylabel('Hydraulic flux $Q$ [m$^3$s$^{-1}$]')
#plt.plot(dpdz, Q, '+')
#plt.grid()

#plt.subplot(3,1,3)
#plt.xlabel('Pressure gradient $\\Delta p/\\Delta z$ [Pa m$^{-1}$]')
#plt.ylabel('Mean porosity $\\bar{\\phi}$ [-]')
#plt.plot(dpdz, phi_bar, '+')
#plt.grid()

plt.tight_layout()
filename = 'permeability-dpdz-vs-K-vs-c.pdf'
plt.savefig(filename)
print(os.getcwd() + '/' + filename)
plt.savefig(filename)

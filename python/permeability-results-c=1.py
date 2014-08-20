#!/usr/bin/env python
import numpy
import sphere
from permeabilitycalculator import *
import matplotlib.pyplot as plt

sigma0_list = numpy.array([1.0e3, 2.0e3, 4.0e3, 10.0e3, 20.0e3, 40.0e3])

sids = []
for sigma0 in sigma0_list:
    sids.append('permeability-dp=' + str(sigma0))

K = numpy.empty(len(sids))
dpdz = numpy.empty_like(K)
Q = numpy.empty_like(K)
phi_bar = numpy.empty_like(K)
i = 0

for sid in sids:
    pc = PermeabilityCalc(sid, plot_evolution=False)
    K[i] = pc.conductivity()
    pc.findPressureGradient()
    pc.findCrossSectionalFlux()
    dpdz[i] = pc.dPdL[2]
    Q[i] = pc.Q[2]
    pc.findMeanPorosity()
    phi_bar[i] = pc.phi_bar

    i += 1

# produce VTK files
#for sid in sids:
    #sim = sphere.sim(sid, fluid=True)
    #sim.writeVTKall()

fig = plt.figure()

plt.subplot(3,1,1)
plt.xlabel('Pressure gradient $\\Delta p/\\Delta z$ [Pa m$^{-1}$]')
plt.ylabel('Hydraulic conductivity $K$ [ms$^{-1}$]')
plt.plot(dpdz, K, '+')
plt.grid()

plt.subplot(3,1,2)
plt.xlabel('Pressure gradient $\\Delta p/\\Delta z$ [Pa m$^{-1}$]')
plt.ylabel('Hydraulic flux $Q$ [m$^3$s$^{-1}$]')
plt.plot(dpdz, Q, '+')
plt.grid()

plt.subplot(3,1,3)
plt.xlabel('Pressure gradient $\\Delta p/\\Delta z$ [Pa m$^{-1}$]')
plt.ylabel('Mean porosity $\\bar{\\phi}$ [-]')
plt.plot(dpdz, phi_bar, '+')
plt.grid()

plt.tight_layout()
plt.savefig('permeability-dpdz-vs-K.png')

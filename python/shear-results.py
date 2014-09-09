#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'})
import shutil

import os
import numpy
import sphere
from permeabilitycalculator import *
import matplotlib.pyplot as plt

#sigma0_list = numpy.array([1.0e3, 2.0e3, 4.0e3, 10.0e3, 20.0e3, 40.0e3])
sigma0 = 10.0e3
cvals = [1.0, 0.1]
c_phi = 1.0

shear_strain = [[], [], []]
friction = [[], [], []]
dilation = [[], [], []]

fluid=True

# dry shear
sid = 'shear-sigma0=' + str(10.0e3)
sim = sphere.sim(sid)
sim.readlast()
sim.visualize('shear')
shear_strain[0] = sim.shear_strain
friction[0] = sim.tau/sim.sigma_eff
dilation[0] = sim.dilation

# wet shear
c = 1
for c in numpy.arange(1,len(cvals)+1):
    c_grad_p = cvals[c-1]

    sid = 'shear-sigma0=' + str(sigma0) + '-c_phi=' + \
                    str(c_phi) + '-c_grad_p=' + str(c_grad_p) + \
                    '-hi_mu-lo_visc'
    if os.path.isfile('../output/' + sid + '.status.dat'):

        sim = sphere.sim(sid, fluid=fluid)
        shear_strain[c] = numpy.zeros(sim.status())
        friction[c] = numpy.zeros_like(shear_strain[c])
        dilation[c] = numpy.zeros_like(shear_strain[c])

        sim.readlast()
        sim.visualize('shear')
        shear_strain[c] = sim.shear_strain
        friction[c] = sim.tau/sim.sigma_eff
        dilation[c] = sim.dilation
    else:
        print(sid + ' not found')

    # produce VTK files
    #for sid in sids:
        #sim = sphere.sim(sid, fluid=True)
        #sim.writeVTKall()
    c += 1

fig = plt.figure(figsize=(8,8))

#plt.subplot(3,1,1)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
mean_diameter = numpy.mean(sim.radius)*2.0

ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
ax1.plot(shear_strain[0], friction[0], label='dry')
ax2.plot(shear_strain[0], dilation[0]/mean_diameter, label='dry')

for c in numpy.arange(1,len(cvals)+1):
    ax1.plot(shear_strain[c][1:], friction[c][1:], \
            label='$c$ = %.2f' % (cvals[c-1]))
    ax2.plot(shear_strain[c][1:], dilation[c][1:]/mean_diameter, \
            label='$c$ = %.2f' % (cvals[c-1]))
    #plt.plot(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.semilogx(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.semilogy(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.loglog(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
ax2.set_xlabel('Shear strain [-]')
ax1.set_ylabel('Shear friction $\\tau/\\sigma\'$ [-]')
ax2.set_ylabel('Dilation $\\Delta h/(2r)$ [-]')
plt.setp(ax1.get_xticklabels(), visible=False)
#ax1.grid()
#ax2.grid()
ax1.legend(loc='lower right', prop={'size':18})
ax2.legend(loc='lower right', prop={'size':18})

plt.tight_layout()
filename = 'shear-10kPa-stress-dilation.pdf'
#print(os.getcwd() + '/' + filename)
plt.savefig(filename)
shutil.copyfile(filename, '/home/adc/articles/own/2-org/' + filename)
print(filename)

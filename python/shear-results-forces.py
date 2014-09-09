#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import shutil

import os
import numpy
import sphere
from permeabilitycalculator import *
import matplotlib.pyplot as plt

#steps = [5, 10, 100]
steps = [3]
nsteps_avg = 1 # no. of steps to average over

sigma0 = 10.0e3
c_grad_p = 1.0
c_phi = 1.0

sid = 'shear-sigma0=' + str(sigma0) + '-c_phi=' + \
                str(c_phi) + '-c_grad_p=' + str(c_grad_p) + '-hi_mu-lo_visc'
sim = sphere.sim(sid, fluid=True)
sim.readfirst(verbose=False)

# particle z positions
zpos_p = numpy.zeros((len(steps), sim.np))

# cell midpoint cell positions
zpos_c = numpy.zeros((len(steps), sim.num[2]))
dz = sim.L[2]/sim.num[2]
for i in numpy.arange(sim.num[2]):
    zpos_c[:,i] = i*dz + 0.5*dz

# particle x displacements
xdisp = numpy.zeros((len(steps), sim.np))

# particle-fluid force per particle
f_pf  = numpy.zeros_like(xdisp)

# pressure - hydrostatic pressure
dev_p = numpy.zeros((len(steps), sim.num[2]))

s = 0
for step in steps:

    if os.path.isfile('../output/' + sid + '.status.dat'):

        if step > sim.status():
            raise Exception(
                    'Simulation step %d not available (sim.status = %d).'
                    % (step, sim.status()))

        for substep in numpy.arange(nsteps_avg):

            sim.readstep(step + substep, verbose=False)

            zpos_p[s,:] += sim.x[:,2]/nsteps_avg

            xdisp[s,:] += sim.xyzsum[:,0]/nsteps_avg

            for i in numpy.arange(sim.np):
                f_pf[s,i] += \
                        sim.f_sum[i].dot(sim.f_sum[i])/nsteps_avg

            dev_p[s,:] += \
                    numpy.average(numpy.average(sim.p_f, axis=0), axis=0)\
                    /nsteps_avg

    else:
        print(sid + ' not found')
    s += 1

fig = plt.figure(figsize=(8,4*(len(steps))))

for s in numpy.arange(len(steps)):
    ax1 = plt.subplot((s+1)*100 + 31)
    ax2 = plt.subplot((s+1)*100 + 32, sharey=ax1)
    ax3 = plt.subplot((s+1)*100 + 33, sharey=ax1)

    ax1.plot(xdisp[s], zpos_p[s], '.')
    ax2.plot(f_pf[s],  zpos_p[s], '.')
    ax3.plot(dev_p[s], zpos_c[s])
    max_z = numpy.max(zpos_p)
    ax1.set_ylim([0, max_z])
    ax2.set_ylim([0, max_z])
    ax3.set_ylim([0, max_z])
    #plt.plot(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.semilogx(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.semilogy(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.loglog(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))

    ax1.set_ylabel('Vertical position $z$ [m]')
    #ax1.set_xlabel('Horizontal particle displacement [m]')
    #ax2.set_xlabel('Average fluid pressure $\\bar{p_f}$ [Pa]')
    #ax3.set_xlabel('Average fluid-particle interaction force '
            #+ '$||\\bar{\\boldsymbol{f_{pf}}}||$ [N]')
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    #ax1.grid()
    #ax2.grid()
    #ax1.legend(loc='lower right', prop={'size':18})
    #ax2.legend(loc='lower right', prop={'size':18})

plt.tight_layout()
filename = 'shear-10kPa-forces.pdf'
plt.savefig(filename)
shutil.copyfile(filename, '/home/adc/articles/own/2-org/' + filename)
print(filename)

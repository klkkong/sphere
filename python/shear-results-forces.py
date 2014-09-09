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
from matplotlib.ticker import MaxNLocator

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

# mean porosity
phi_bar = numpy.zeros((len(steps), sim.num[2]))

# mean per-particle values
xdisp_mean = numpy.zeros((len(steps), sim.num[2]))
f_pf_mean = numpy.zeros((len(steps), sim.num[2]))

shear_strain = numpy.zeros(len(steps))

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

            '''
            for i in numpy.arange(sim.np):
                f_pf[s,i] += \
                        sim.f_sum[i].dot(sim.f_sum[i])/nsteps_avg
                        '''
            f_pf[s,:] += sim.f_sum[:,2]

            dev_p[s,:] += \
                    numpy.average(numpy.average(sim.p_f, axis=0), axis=0)\
                    /nsteps_avg

            phi_bar[s,:] += \
                    numpy.average(numpy.average(sim.phi, axis=0), axis=0)\
                    /nsteps_avg

            shear_strain[s] += sim.shearStrain()/nsteps_avg

        # calculate mean values of xdisp and f_pf
        for iz in numpy.arange(sim.num[2]):
            z_bot = iz*dz
            z_top = (iz+1)*dz
            I = numpy.nonzero((zpos_p[s,:] >= z_bot) & (zpos_p[s,:] < z_top))
            if len(I) > 0:
                xdisp_mean[s,iz] = numpy.mean(xdisp[s,I])
                f_pf_mean[s,iz] = numpy.mean(f_pf[s,I])

    else:
        print(sid + ' not found')
    s += 1

fig = plt.figure(figsize=(8,4*(len(steps))))

for s in numpy.arange(len(steps)):
    ax1 = plt.subplot((s+1)*100 + 31)
    ax2 = plt.subplot((s+1)*100 + 32, sharey=ax1)
    ax3 = plt.subplot((s+1)*100 + 33, sharey=ax1)
    ax4 = ax3.twiny()

    ax1.plot(xdisp[s], zpos_p[s], '+', color = '#888888')
    ax1.plot(xdisp_mean[s], zpos_c[s], color = 'k')

    ax2.plot(f_pf[s],  zpos_p[s], '+', color = '#888888')
    ax2.plot(f_pf_mean[s], zpos_c[s], color = 'k')

    ax3.plot(dev_p[s]/1000.0, zpos_c[s], 'k')

    phicolor = '#666666'
    ax4.plot(phi_bar[s], zpos_c[s], '--', color = phicolor)
    for tl in ax4.get_xticklabels():
        tl.set_color(phicolor)

    max_z = numpy.max(zpos_p)
    ax1.set_ylim([0, max_z])
    ax2.set_ylim([0, max_z])
    ax3.set_ylim([0, max_z])
    #plt.plot(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.semilogx(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.semilogy(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.loglog(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))

    ax1.set_ylabel('Vertical position $z$ [m]')
    ax1.set_xlabel('$x^3_\\text{p}$ [m]')
    ax2.set_xlabel('$\\boldsymbol{f}_\\text{pf}$ [N]')
    ax3.set_xlabel('$\\bar{p_\\text{f}}$ [kPa]')
    ax4.set_xlabel('$\\bar{\\phi}$ [-]', color=phicolor)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)

    ax1.get_xaxis().set_major_locator(MaxNLocator(nbins=5))
    ax2.get_xaxis().set_major_locator(MaxNLocator(nbins=5))
    ax3.get_xaxis().set_major_locator(MaxNLocator(nbins=5))

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=90)

    fig.text(0.1, 0.9,
            'Shear strain $\\gamma = %.3f$' % (shear_strain[s]),
            horizontalalignment='left', fontsize=22)
    #ax1.grid()
    #ax2.grid()
    #ax1.legend(loc='lower right', prop={'size':18})
    #ax2.legend(loc='lower right', prop={'size':18})

plt.tight_layout()
plt.subplots_adjust(wspace = .05)
plt.MaxNLocator(nbins=4)
filename = 'shear-10kPa-forces.pdf'
plt.savefig(filename)
shutil.copyfile(filename, '/home/adc/articles/own/2-org/' + filename)
print(filename)

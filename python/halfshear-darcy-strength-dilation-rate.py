#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import shutil

import os
import sys
import numpy
import sphere
from permeabilitycalculator import *
import matplotlib.pyplot as plt

import seaborn as sns
#sns.set(style='ticks', palette='Set2')
sns.set(style='ticks', palette='colorblind')
#sns.set(style='ticks', palette='muted')
#sns.set(style='ticks', palette='pastel')
sns.despine() # remove right and top spines

pressures = True
zflow = False
contact_forces = False

#sigma0_list = numpy.array([1.0e3, 2.0e3, 4.0e3, 10.0e3, 20.0e3, 40.0e3])
sigma0 = 20000.0
#k_c_vals = [3.5e-13, 3.5e-15]
k_c = 3.5e-15
#k_c = 3.5e-13

# 5.0e-8 results present
#mu_f_vals = [1.797e-06, 1.204e-06, 5.0e-8, 1.797e-08]
mu_f_vals = [1.797e-06, 1.204e-06, 3.594e-07, 1.797e-08]
#mu_f_vals = [1.797e-06, 1.204e-06, 1.797e-08]
velfac = 1.0


shear_strain = [[], [], [], []]
friction = [[], [], [], []]
dilation = [[], [], [], []]
p_min = [[], [], [], []]
p_mean = [[], [], [], []]
p_max = [[], [], [], []]
f_n_mean = [[], [], [], []]
f_n_max  = [[], [], [], []]
v_f_z_mean  = [[], [], [], []]

fluid=True

# wet shear
for c in numpy.arange(0,len(mu_f_vals)):
#for c in numpy.arange(len(mu_f_vals)-1, -1, -1):
    mu_f = mu_f_vals[c]

    # halfshear-darcy-sigma0=20000.0-k_c=3.5e-13-mu=1.797e-06-velfac=1.0-shear
    sid = 'halfshear-darcy-sigma0=' + str(sigma0) + '-k_c=' + str(k_c) + \
            '-mu=' + str(mu_f) + '-velfac=' + str(velfac) + '-shear'
    #sid = 'halfshear-sigma0=' + str(sigma0) + '-c_v=' + str(c_v) +\
            #'-c_a=0.0-velfac=1.0-shear'
    if os.path.isfile('../output/' + sid + '.status.dat'):

        sim = sphere.sim(sid, fluid=fluid)
        n = sim.status()
        #n = 20
        shear_strain[c] = numpy.zeros(n)
        friction[c] = numpy.zeros_like(shear_strain[c])
        dilation[c] = numpy.zeros_like(shear_strain[c])

        '''
        sim.readlast(verbose=False)
        #sim.visualize('shear')
        shear_strain[c] = sim.shear_strain
        #shear_strain[c] = numpy.arange(sim.status()+1)
        #friction[c] = sim.tau/1000.0#/sim.sigma_eff
        friction[c] = sim.shearStress('effective')/sim.currentNormalStress('defined')
        dilation[c] = sim.dilation
        '''

        # fluid pressures and particle forces
        p_mean[c]   = numpy.zeros_like(shear_strain[c])
        p_min[c]    = numpy.zeros_like(shear_strain[c])
        p_max[c]    = numpy.zeros_like(shear_strain[c])
        f_n_mean[c] = numpy.zeros_like(shear_strain[c])
        f_n_max[c]  = numpy.zeros_like(shear_strain[c])

        for i in numpy.arange(n):

            sim.readstep(i, verbose=False)

            shear_strain[c][i] = sim.shearStrain()
            friction[c][i] = sim.shearStress('effective')/sim.currentNormalStress('defined')
            dilation[c][i] = sim.w_x[0]

            if pressures:
                iz_top = int(sim.w_x[0]/(sim.L[2]/sim.num[2]))-1
                p_mean[c][i] = numpy.mean(sim.p_f[:,:,0:iz_top])/1000
                p_min[c][i]  = numpy.min(sim.p_f[:,:,0:iz_top])/1000
                p_max[c][i]  = numpy.max(sim.p_f[:,:,0:iz_top])/1000

            if contact_forces:
                sim.findNormalForces()
                f_n_mean[c][i] = numpy.mean(sim.f_n_magn)
                f_n_max[c][i]  = numpy.max(sim.f_n_magn)

        if zflow:
            v_f_z_mean[c] = numpy.zeros_like(shear_strain[c])
            for i in numpy.arange(n):
                    v_f_z_mean[c][i] = numpy.mean(sim.v_f[:,:,:,2])

        dilation[c] =\
                (dilation[c] - dilation[c][0])/(numpy.mean(sim.radius)*2.0)

    else:
        print(sid + ' not found')

    # produce VTK files
    #for sid in sids:
        #sim = sphere.sim(sid, fluid=True)
        #sim.writeVTKall()


if zflow or pressures:
    #fig = plt.figure(figsize=(8,10))
    fig = plt.figure(figsize=(3.74, 2*3.74))
else:
    fig = plt.figure(figsize=(8,8)) # (w,h)
#fig = plt.figure(figsize=(8,12))
#fig = plt.figure(figsize=(8,16))
fig.subplots_adjust(hspace=0.0)

#plt.subplot(3,1,1)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

if zflow or pressures:
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313, sharex=ax1)
else:
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)
#ax3 = plt.subplot(413, sharex=ax1)
#ax4 = plt.subplot(414, sharex=ax1)
#alpha = 0.5
alpha = 1.0
#ax1.plot(shear_strain[0], friction[0], label='dry', linewidth=1, alpha=alpha)
#ax2.plot(shear_strain[0], dilation[0], label='dry', linewidth=1)
#ax4.plot(shear_strain[0], f_n_mean[0], '-', label='dry', color='blue')
#ax4.plot(shear_strain[0], f_n_max[0], '--', color='blue')

#color = ['b','g','r','c']
#color = ['g','r','c']
color = sns.color_palette()
#for c, mu_f in enumerate(mu_f_vals):
for c in numpy.arange(len(mu_f_vals)-1, -1, -1):
    mu_f = mu_f_vals[c]

    if numpy.isclose(mu_f, 1.797e-6):
        label = 'ref. shear velocity'
    #elif numpy.isclose(mu_f, 1.204e-6):
        #label = 'ref. shear velocity$\\times$0.67'
    #elif numpy.isclose(mu_f, 1.797e-8):
        #label = 'ref. shear velocity$\\times$0.01'
    else:
        #label = '$\\mu_\\text{{f}}$ = {:.3e} Pa s'.format(mu_f)
        label = 'ref. shear velocity$\\times${:.2}'.format(mu_f/mu_f_vals[0])

    ax1.plot(shear_strain[c], friction[c], \
            label=label, linewidth=1,
            alpha=alpha, color=color[c])

    ax2.plot(shear_strain[c], dilation[c], \
            label=label, linewidth=1,
            color=color[c])

    if zflow:
        ax3.plot(shear_strain[c], v_f_z_mean[c],
            label=label, linewidth=1)

    if pressures:
        #ax3.plot(shear_strain[c][1:], p_max[c][1:], '-', color=color[c],
                #alpha=0.5)
        ax3.plot(shear_strain[c], p_mean[c], '-', color=color[c], \
                label=label, linewidth=1)
        #ax3.plot(shear_strain[c][1:], p_min[c][1:], '-', color=color[c],
                #alpha=0.5)

        #ax3.fill_between(shear_strain[c][1:], p_min[c][1:], p_max[c][1:], 
                #where=p_min[c][1:]<=p_max[c][1:], facecolor=color[c],
                #interpolate=True, alpha=0.5)

        #ax4.plot(shear_strain[c][1:], f_n_mean[c][1:], '-' + color[c],
                #label='$c$ = %.2f' % (cvals[c-1]), linewidth=2)
        #ax4.plot(shear_strain[c][1:], f_n_max[c][1:], '--' + color[c])
            #label='$c$ = %.2f' % (cvals[c-1]), linewidth=2)

#ax4.set_xlabel('Shear strain $\\gamma$ [-]')
if zflow or pressures:
    ax3.set_xlabel('Shear strain $\\gamma$ [-]')
else:
    ax2.set_xlabel('Shear strain $\\gamma$ [-]')

ax1.set_ylabel('Shear friction $\\tau/\\sigma_0$ [-]')
#ax1.set_ylabel('Shear stress $\\tau$ [kPa]')
ax2.set_ylabel('Dilation $\\Delta h/(2r)$ [-]')
if zflow:
    ax3.set_ylabel('$\\boldsymbol{v}_\\text{f}^z h$ [ms$^{-1}$]')
if pressures:
    ax3.set_ylabel('Mean fluid pressure $\\bar{p}_\\text{f}$ [kPa]')
#ax4.set_ylabel('Particle contact force $||\\boldsymbol{f}_\\text{p}||$ [N]')

#ax1.set_xlim([200,300])
#ax3.set_ylim([595,608])

plt.setp(ax1.get_xticklabels(), visible=False)
if zflow or pressures:
    plt.setp(ax2.get_xticklabels(), visible=False)
#plt.setp(ax2.get_xticklabels(), visible=False)
#plt.setp(ax3.get_xticklabels(), visible=False)

'''
ax1.grid()
ax2.grid()
if zflow or pressures:
    ax3.grid()
#ax4.grid()
'''


# remove box at top and right
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['right'].set_visible(False)
#ax1.spines['left'].set_visible(True)
# remove ticks at top and right
ax1.get_xaxis().set_ticks_position('none')
ax1.get_yaxis().set_ticks_position('none')
ax1.get_yaxis().tick_left()
ax1.get_xaxis().grid(True, linestyle='--', linewidth=0.5)

# remove box at top and right
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
# remove ticks at top and right
ax2.get_xaxis().set_ticks_position('none')
ax2.get_yaxis().set_ticks_position('none')
ax2.get_yaxis().tick_left()
ax2.get_xaxis().grid(True, linestyle='--', linewidth=0.5)

# remove box at top and right
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
# remove ticks at top and right
ax3.get_xaxis().set_ticks_position('none')
ax3.get_yaxis().set_ticks_position('none')
ax3.get_xaxis().tick_bottom()
ax3.get_yaxis().tick_left()
ax3.get_xaxis().grid(True, linestyle='--', linewidth=0.5)

ax1.legend(loc='best')
#legend_alpha=0.5
#ax1.legend(loc='upper right', prop={'size':18}, fancybox=True,
        #framealpha=legend_alpha)
#ax2.legend(loc='lower right', prop={'size':18}, fancybox=True,
        #framealpha=legend_alpha)
#if zflow or pressures:
    #ax3.legend(loc='upper right', prop={'size':18}, fancybox=True,
            #framealpha=legend_alpha)
#ax4.legend(loc='best', prop={'size':18}, fancybox=True,
        #framealpha=legend_alpha)

#ax1.set_xlim([0.0, 0.09])
#ax2.set_xlim([0.0, 0.09])
#ax2.set_xlim([0.0, 0.2])

#ax1.set_ylim([-7, 45])
ax2.set_ylim([0.0, 0.8])
#ax1.set_ylim([0.0, 1.0])
#if pressures:
    #ax3.set_ylim([-1400, 900])
    #ax3.set_ylim([-200, 200])
    #ax3.set_xlim([0.0, 0.09])

plt.tight_layout()
#plt.subplots_adjust(hspace=0.05)
plt.subplots_adjust(hspace=0.15)
#filename = 'shear-' + str(int(sigma0/1000.0)) + 'kPa-stress-dilation.pdf'
filename = 'halfshear-darcy-rate.pdf'
#print(os.getcwd() + '/' + filename)
plt.savefig(filename)
shutil.copyfile(filename, '/home/adc/articles/own/2/graphics/' + filename)
plt.close()
print(filename)

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

sigma0 = 20000.0
#cvals = ['dry', 1.0, 0.1, 0.01]
cvals = ['dry', 3.5e-13, 3.5e-15]
#cvals = ['dry', 1.0]
#step = 1999

sim = sphere.sim('halfshear-sigma0=' + str(sigma0) + '-shear')
sim.readfirst(verbose=False)


# particle z positions
zpos_p = [[], [], []]

# cell midpoint cell positions
zpos_c = [[], [], []]

# particle x displacements
xdisp = [[], [], []]
xdisp_mean = [[], [], []]

s = 0
for c in cvals:

    if c == 'dry':
        fluid = False
        sid = 'halfshear-sigma0=' + str(sigma0) + '-shear'
    else:
        fluid = True
        sid = 'halfshear-darcy-sigma0=' + str(sigma0) + '-k_c=' + str(c) + \
        '-mu=1.797e-06-velfac=1.0-shear'

    sim = sphere.sim(sid, fluid=fluid)

    if os.path.isfile('../output/' + sid + '.status.dat'):

        sim.readlast(verbose=False)

        zpos_c[s] = numpy.zeros(sim.num[2]*2)
        dz = sim.L[2]/(sim.num[2]*2)
        for i in numpy.arange(sim.num[2]*2):
            zpos_c[s][i] = i*dz + 0.5*dz

        xdisp[s] = numpy.zeros(sim.np)
        xdisp_mean[s] = numpy.zeros(sim.num[2]*2)


        zpos_p[s][:] = sim.x[:,2]

        xdisp[s][:] = sim.xyzsum[:,0]

        #shear_strain[s] += sim.shearStrain()/nsteps_avg

        # calculate mean values of xdisp and f_pf
        for iz in numpy.arange(sim.num[2]*2):
            z_bot = iz*dz
            z_top = (iz+1)*dz
            I = numpy.nonzero((zpos_p[s][:] >= z_bot) & (zpos_p[s][:] < z_top))
            if len(I) > 0:
                xdisp_mean[s][iz] = numpy.mean(xdisp[s][I])

        # normalize distance
        max_dist = numpy.nanmax(xdisp_mean[s])
        xdisp_mean[s] /= max_dist

    else:
        print(sid + ' not found')
    s += 1


#fig = plt.figure(figsize=(8,4*(len(steps))+1))
#fig = plt.figure(figsize=(8,5*(len(steps))+1))
fig = plt.figure(figsize=(8,6))

ax = []
#linetype = ['-', '--', '-.']
linetype = ['-', '-', '-', '-']
#color = ['b','g','c','y']
color = ['b','g','r','y']
for s in numpy.arange(len(cvals)):

    ax.append(plt.subplot(111))
    #ax.append(plt.subplot(len(steps)*100 + 31 + s*3))
    #ax.append(plt.subplot(len(steps)*100 + 32 + s*3, sharey=ax[s*4+0]))
    #ax.append(plt.subplot(len(steps)*100 + 33 + s*3, sharey=ax[s*4+0]))
    #ax.append(ax[s*4+2].twiny())

    if cvals[s] == 'dry':
        legend = 'dry'
    elif cvals[s] == 3.5e-13:
        legend = 'wet, relatively permeable'
    elif cvals[s] == 3.5e-15:
        legend = 'wet, relatively impermeable'
    else:
        legend = 'wet, $k_c$ = ' + str(cvals[s]) + ' m$^2$'

    #ax[0].plot(xdisp[s], zpos_p[s], ',', color = '#888888')
    #ax[0].plot(xdisp[s], zpos_p[s], ',', color=color[s], alpha=0.5)
    ax[0].plot(xdisp_mean[s], zpos_c[s], linetype[s],
            color=color[s], label=legend, linewidth=1)

    ax[0].set_ylabel('Vertical position $z$ [m]')
    #ax[0].set_xlabel('$\\boldsymbol{x}^x_\\text{p}$ [m]')
    ax[0].set_xlabel('Normalized horizontal distance')

    #ax[s*4+0].get_xaxis().set_major_locator(MaxNLocator(nbins=5))
    #ax[s*4+1].get_xaxis().set_major_locator(MaxNLocator(nbins=5))
    #ax[s*4+2].get_xaxis().set_major_locator(MaxNLocator(nbins=5))

    #plt.setp(ax[s*4+0].xaxis.get_majorticklabels(), rotation=90)
    #plt.setp(ax[s*4+1].xaxis.get_majorticklabels(), rotation=90)
    #plt.setp(ax[s*4+2].xaxis.get_majorticklabels(), rotation=90)
    #plt.setp(ax[s*4+3].xaxis.get_majorticklabels(), rotation=90)

    #if s == 0:
        #y = 0.95
    #if s == 1:
        #y = 0.55

    #strain_str = 'Shear strain $\\gamma = %.3f$' % (shear_strain[s])
    #fig.text(0.1, y, strain_str, horizontalalignment='left', fontsize=22)
    #ax[s*4+0].annotate(strain_str, xytext=(0,1.1), textcoords='figure fraction',
            #horizontalalignment='left', fontsize=22)
    #plt.text(0.05, 1.06, strain_str, horizontalalignment='left', fontsize=22,
            #transform=ax[s*4+0].transAxes)
    #ax[s*4+0].set_title(strain_str)

    #ax[s*4+0].grid()
    #ax[s*4+1].grid()
    #ax[s*4+2].grid()
    #ax1.legend(loc='lower right', prop={'size':18})
    #ax2.legend(loc='lower right', prop={'size':18})

legend_alpha=0.5
ax[0].legend(loc='lower right', prop={'size':18}, fancybox=True, framealpha=legend_alpha)
ax[0].grid()
ax[0].set_xlim([-0.1,1.1])
plt.tight_layout()
plt.subplots_adjust(wspace = .05)
plt.MaxNLocator(nbins=4)

filename = 'halfshear-darcy-strain.pdf'
plt.savefig(filename)
shutil.copyfile(filename, '/home/adc/articles/own/2/graphics/' + filename)
print(filename)



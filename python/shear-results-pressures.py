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

matplotlib.rcParams['image.cmap'] = 'bwr'

sigma0 = float(sys.argv[1])
#c_grad_p = 1.0
c_grad_p = float(sys.argv[2])
c_phi = 1.0

#sid = 'shear-sigma0=' + str(sigma0) + '-c_phi=' + \
#                str(c_phi) + '-c_grad_p=' + str(c_grad_p) + '-hi_mu-lo_visc'
sid = 'halfshear-sigma0=' + str(sigma0) + '-c=' + str(c_grad_p) + '-shear'
sim = sphere.sim(sid, fluid=True)
sim.readfirst(verbose=False)

# cell midpoint cell positions
zpos_c = numpy.zeros(sim.num[2])
dz = sim.L[2]/sim.num[2]
for i in numpy.arange(sim.num[2]):
    zpos_c[i] = i*dz + 0.5*dz

shear_strain = numpy.zeros(sim.status())

dev_pres = numpy.zeros((sim.num[2], sim.status()))
pres_static = numpy.ones_like(dev_pres)*600.0e3
pres = numpy.zeros_like(dev_pres)

for i in numpy.arange(sim.status()):

    sim.readstep(i, verbose=False)

    pres[:,i] = numpy.average(numpy.average(sim.p_f, axis=0), axis=0)

    dz = sim.L[2]/sim.num[2]
    wall0_iz = int(sim.w_x[0]/dz)
    for z in numpy.arange(0, wall0_iz+1):
                #(wall0_iz*dz - zpos_c[z] + 0.5*dz)*sim.rho_f*numpy.abs(sim.g[2])\
        pres_static[z,i] = \
                (wall0_iz*dz - zpos_c[z])*sim.rho_f*numpy.abs(sim.g[2])\
                + sim.p_f[0,0,-1]
        #pres_static[z,i] = zpos_c[z]
        #pres_static[z,i] = z

    shear_strain[i] = sim.shearStrain()

dev_pres = pres - pres_static

#fig = plt.figure(figsize=(8,6))
#fig = plt.figure(figsize=(8,12))
fig = plt.figure(figsize=(8,15))

min_p = numpy.min(dev_pres)/1000.0
#max_p = numpy.min(dev_pres)
max_p = numpy.abs(min_p)

#cmap = matplotlib.colors.ListedColormap(['b', 'w', 'r'])
#bounds = [min_p, 0, max_p]
#norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

ax1 = plt.subplot(311)
#ax1 = plt.subplot(111)
#ax1 = plt.subplot(211)
#im1 = ax1.pcolormesh(shear_strain, zpos_c, dev_pres/1000.0, rasterized=True,
#        cmap=cmap, norm=norm)
im1 = ax1.pcolormesh(shear_strain, zpos_c, dev_pres/1000.0, vmin=min_p,
        vmax=max_p, rasterized=True)
#ax1.set_xlim([0, shear_strain[-1]])
#ax1.set_ylim([zpos_c[0], sim.w_x[0]])
ax1.set_xlabel('Shear strain $\\gamma$ [-]')
ax1.set_ylabel('Vertical position $z$ [m]')
cb1 = plt.colorbar(im1)
#cb1 = plt.colorbar(im1, cmap=cmap, norm=norm)
cb1.set_label('$p_\\text{f} - p^\\text{hyd}_\\text{f}$ [kPa]')
cb1.solids.set_rasterized(True)

# annotate plot
#ax1.text(0.02, 0.15, 'compressive',
        #bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

#ax1.text(0.12, 0.25, 'dilative',
        #bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

#'''
ax2 = plt.subplot(312)
im2 = ax2.pcolormesh(shear_strain, zpos_c, pres/1000.0, rasterized=True)
#ax2.set_xlim([0, shear_strain[-1]])
#ax2.set_ylim([zpos_c[0], sim.w_x[0]])
ax2.set_xlabel('Shear strain $\\gamma$ [-]')
ax2.set_ylabel('Vertical position $z$ [m]')
cb2 = plt.colorbar(im2)
cb2.set_label('Pressure $p_\\text{f}$ [kPa]')
cb2.solids.set_rasterized(True)

#'''
ax3 = plt.subplot(313)
im3 = ax3.pcolormesh(shear_strain, zpos_c, pres_static/1000.0, rasterized=True)
#ax3.set_xlim([0, shear_strain[-1]])
#ax3.set_ylim([zpos_c[0], sim.w_x[0]])
ax3.set_xlabel('Shear strain $\\gamma$ [-]')
ax3.set_ylabel('Vertical position $z$ [m]')
cb3 = plt.colorbar(im3)
cb3.set_label('Static Pressure $p_\\text{f}$ [kPa]')
cb3.solids.set_rasterized(True)
#'''


#plt.MaxNLocator(nbins=4)
plt.tight_layout()
plt.subplots_adjust(wspace = .05)
#plt.MaxNLocator(nbins=4)

filename = 'shear-' + str(int(sigma0/1000.0)) + 'kPa-pressures.pdf'
plt.savefig(filename)
shutil.copyfile(filename, '/home/adc/articles/own/2-org/' + filename)
print(filename)

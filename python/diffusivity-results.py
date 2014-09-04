#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'})
import os

import sphere
import numpy
import matplotlib.pyplot as plt
#import diffusivitycalc

c_phi = 1.0
c_grad_p = 1.0
#sigma0_list = numpy.array([5.0e3, 10.0e3, 20.0e3, 40.0e3, 80.0e3, 160.0e3])
sigma0_list = numpy.array([10.0e3, 20.0e3, 40.0e3, 80.0e3, 160.0e3])
alpha = numpy.empty_like(sigma0_list)
phi_bar = numpy.empty_like(sigma0_list)

#dc = diffusivitycalc.DiffusivityCalc()

i = 0
for sigma0 in sigma0_list:

    sid = 'cons-sigma0=' + str(sigma0) + '-c_phi=' + \
                     str(c_phi) + '-c_grad_p=' + str(c_grad_p)
    if os.path.isfile('../output/' + sid + '.status.dat'):
        sim = sphere.sim(sid, fluid=True)

        sim.visualize('walls')
        sim.plotLoadCurve()
        alpha[i] = sim.c_v
        phi_bar[i] = sim.phi_bar
        #sim.writeVTKall()

    else:
        print(sid + ' not found')

    i += 1

fig, ax1 = plt.subplots()
sigma0_list /= 1000.0


#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.plot(sigma0_list, alpha, '.-k')
ax1.set_xlabel('Normal stress $\\sigma_0$ [kPa]')
ax1.set_ylabel('Hydraulic diffusivity $\\alpha$ [m$^2$s$^{-1}$]')
#ax1.grid()

ax2 = ax1.twinx()
color = 'b'
ax2.plot(sigma0_list, phi_bar, '.--' + color)
ax2.set_ylabel('Mean porosity $\\bar{\\phi}$ [-]')
for tl in ax2.get_yticklabels():
    tl.set_color(color)

filename = 'diffusivity-sigma0-vs-alpha.pdf'
plt.tight_layout()
plt.savefig(filename)
print(os.getcwd() + '/' + filename)

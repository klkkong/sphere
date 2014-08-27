#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'})
import os

import sphere
import numpy
import matplotlib.pyplot as plt
import diffusivitycalc


c_phi = 1.0
c_grad_p = 1.0
sigma0_list = numpy.array([5.0e3, 10.0e3, 20.0e3, 40.0e3, 80.0e3, 160.0e3])
alpha = numpy.empty(len(sigma0_list))

dc = diffusivitycalc.DiffusivityCalc()

i = 0
for sigma0 in sigma0_list:

    sim = sphere.sim('cons-sigma0=' + str(sigma0) + '-c_phi=' + \
                     str(c_phi) + '-c_grad_p=' + str(c_grad_p), fluid=True)

    sim.visualize('walls')
    sim.plotLoadCurve()
    #sim.writeVTKall()

    i += 1


plt.xlabel('Normal stress $\\sigma_0$ [kPa]')
plt.ylabel('Hydraulic diffusivity $\\alpha$ [m$^2$s$^{-1}$]')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
sigma0 /= 1000.0
plt.plot(sigma0, alpha, 'o-k')
plt.grid()

plt.tight_layout()
filename = 'diffusivity-sigma0-vs-alpha.pdf'
plt.savefig(filename)
print(os.getcwd() + '/' + filename)
plt.savefig(filename)

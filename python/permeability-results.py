#!/usr/bin/env python
import numpy
import matplotlib.pyplot as plt
from permeabilitycalculator import *

sids = [
    'permeability-dp=1000.0',
    'permeability-dp=1000.0-c_phi=1.0-c_grad_p=0.01',
    'permeability-dp=1000.0-c_phi=1.0-c_grad_p=0.5',
    'permeability-dp=20000.0-c_phi=1.0-c_grad_p=0.01',
    'permeability-dp=20000.0-c_phi=1.0-c_grad_p=0.1',
    'permeability-dp=20000.0-c_phi=1.0-c_grad_p=0.5',
    'permeability-dp=4000.0-c_phi=1.0-c_grad_p=0.01',
    'permeability-dp=4000.0-c_phi=1.0-c_grad_p=0.1',
    'permeability-dp=4000.0-c_phi=1.0-c_grad_p=0.5',
    'permeability-dp=4000.0']

K = numpy.array([])
c_grad_p = numpy.array([])

for sid in sids:
    pc = PermeabilityCalc(sid)
    K.append(pc.conductivity())
    c_grad_p.append(pc.c_grad_p())
        
fig = plt.figure()
plt.xlabel('Pressure gradient coefficient $c$ [-]')
plt.ylabel('Hydraulic conductivity $K$ [m/s]')
plt.plot(c_grad_p, K)
plt.grid()
plt.savefig('c_grad_p-vs-K.png')

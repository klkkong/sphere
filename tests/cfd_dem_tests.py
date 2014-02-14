#!/usr/bin/env python
from pytestutils import *

import sphere
import sys
import numpy
import matplotlib.pyplot as plt

print("### Coupled CFD-DEM tests ###")
'''
## Stokes drag
# Iteration and conservation of mass test
# No gravity, no pressure gradients => no flow
orig = sphere.Spherebin(sid = "cfddemtest", fluid = True)
cleanup(orig)
orig.defaultParams()
orig.addParticle([0.5,0.5,0.5], 0.05)
orig.defineWorldBoundaries([1.0,1.0,1.0])
orig.initFluid(nu = 8.9e-4)
#orig.initTemporal(total = 0.2, file_dt = 0.01)
orig.initTemporal(total = 0.1, file_dt = 0.01)
orig.g[2] = -10.0
orig.writebin(verbose=False)
orig.run(dry=True)
orig.run(verbose=True)
py = Spherebin(sid = orig.sid, fluid = True)

ones = numpy.ones((orig.num))
py.readlast(verbose = False)
py.plotConvergence()
py.writeVTKall()
#compareNumpyArrays(ones, py.p_f, "Conservation of pressure:")

it = numpy.loadtxt("../output/" + orig.sid + "-conv.log")
#test((it[:,1] < 2000).all(), "Convergence rate:\t\t")

t = numpy.empty(py.status())
acc = numpy.empty(py.status())
vel = numpy.empty(py.status())
pos = numpy.empty(py.status())
for i in range(py.status()):
    py.readstep(i+1, verbose=False)
    t[i] = py.time_current[0]
    acc[i] = py.force[0,2]/(V_sphere(py.radius[0])*py.rho[0])
    vel[i] = py.vel[0,2]
    pos[i] = py.x[0,2]

fig = plt.figure()
#plt.title('Convergence evolution in CFD solver in "' + self.sid + '"')
plt.xlabel('Time [s]')
plt.ylabel('$z$ value')
plt.plot(t, acc, label='Acceleration')
plt.plot(t, vel, label='Velocity')
plt.plot(t, pos, label='Position')
plt.grid()
plt.legend()
format = 'png'
plt.savefig('./' + py.sid + '-stokes.' + format)
plt.clf()
plt.close(fig)
#cleanup(orig)
'''


## Sedimentation of many particles
orig = sphere.Spherebin("sedimentation", np = 2000, fluid = True)
orig.radius[:] = 0.05
orig.initRandomGridPos(gridnum = [20, 20, 9000])
orig.initFluid()
orig.initTemporal(total = 3.0, file_dt = 0.01)
orig.g[2] = -9.81
orig.writebin()
orig.run(dry=True)
orig.run()
orig.writeVTKall()

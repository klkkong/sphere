#!/usr/bin/env python
from pytestutils import *

import sphere
import sys
import numpy
import matplotlib.pyplot as plt

print("### CFD tests - Dirichlet/Neumann BCs ###")

# Iteration and conservation of mass test
# No gravity, no pressure gradients => no flow
orig = sphere.sim("neumann", fluid = True)
cleanup(orig)
orig.defaultParams(mu_s = 0.4, mu_d = 0.4)
orig.defineWorldBoundaries([0.4, 0.4, 1], dx = 0.1)
orig.addParticle([0.2, 0.2, 0.9], 0.02)
orig.initFluid(mu = 8.9e-4)
orig.initTemporal(total = 1.0, file_dt = 0.05, dt = 1.0e-4)
py = sphere.sim(sid = orig.sid, fluid = True)
#orig.time_total[0] = 1.0e-2
#orig.time_total = orig.time_dt*10
#orig.time_file_dt = orig.time_total/20
orig.p_f[:,:,-1] = 1.1
orig.g[2] = -10.0
orig.bc_bot[0] = 1      # No-flow BC at bottom (Neumann)
#orig.run(dry=True)
orig.run(verbose=False)
#orig.writeVTKall()
py.readlast(verbose = False)
ideal_grad_p_z = numpy.linspace(orig.p_f[0,0,0], orig.p_f[0,0,-1], orig.num[2])
compareNumpyArraysClose(numpy.zeros((1,orig.num[2])),\
        ideal_grad_p_z - py.p_f[0,0,:],\
        "Pressure gradient:\t", tolerance=1.0e-1)

# Fluid flow along z should be very small
if ((py.v_f[:,:,:,2] < 1.0e-3).all()):
    print("Flow field:\t\t" + passed())
else:
    print("Flow field:\t\t" + failed())


#for i in range(py.num[2]-1,-1,-1):
    #print(str(i) + ': ' + str(numpy.mean(py.v_f[:,:,i,2])))
#cleanup(orig)

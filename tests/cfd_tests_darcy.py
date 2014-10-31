#!/usr/bin/env python
from pytestutils import *

import sphere
import sys
import numpy
import matplotlib.pyplot as plt

print("### CFD tests - Dirichlet BCs ###")

# Iteration and conservation of mass test
# No gravity, no pressure gradients => no flow
orig = sphere.sim(np = 0, nd = 3, nw = 0, sid = "cfdtest", fluid = True)
cleanup(orig)
orig.defaultParams()
orig.addParticle([0.5,0.5,0.5], 0.05)
orig.defineWorldBoundaries([1.0,1.0,1.0])
orig.initFluid(cfd_solver = 1)
#orig.initFluid(mu = 8.9e-4)
orig.initTemporal(total = 0.2, file_dt = 0.01)
#orig.g[2] = -10.0
orig.time_file_dt = orig.time_dt*0.99
orig.time_total = orig.time_dt*10
#orig.run(dry=True)
orig.run(verbose=False)
#orig.run(verbose=True)
py = sphere.sim(sid = orig.sid, fluid = True)

ones = numpy.ones((orig.num))
py.readlast(verbose = False)
compareNumpyArrays(ones, py.p_f, "Conservation of pressure:")

# Convergence rate (1/2)
it = numpy.loadtxt("../output/" + orig.sid + "-conv.log")
compare(it[:,1].sum(), 0.0, "Convergence rate (1/2):\t")

# Fluid flow should be very small
if ((numpy.abs(py.v_f[:,:,:,:]) < 1.0e-6).all()):
    print("Flow field:\t\t" + passed())
else:
    print("Flow field:\t\t" + failed())
    print(numpy.min(py.v_f))
    print(numpy.mean(py.v_f))
    print(numpy.max(py.v_f))
    raise Exception("Failed")


# Add pressure gradient
# This test passes with BETA=0.0 and tolerance=1.0e-9
orig.deleteAllParticles()
#orig.defineWorldBoundaries([0.3,0.3,0.3], dx=0.1)
orig.initFluid(cfd_solver = 1)
orig.p_f[:,:,-1] = 1.1
#orig.time_file_dt = orig.time_dt*0.99
#orig.time_total = orig.time_dt*1
#orig.setMaxIterations(10)
orig.run(verbose=False)
#orig.run(verbose=True)
py.readlast(verbose = False)
ideal_grad_p_z = numpy.linspace(orig.p_f[0,0,0], orig.p_f[0,0,-1], orig.num[2])
#orig.writeVTKall()
compareNumpyArraysClose(numpy.zeros((1,orig.num[2])),\
        ideal_grad_p_z - py.p_f[0,0,:],\
        "Pressure gradient:\t", tolerance=1.0e-1)
        #"Pressure gradient:\t", tolerance=1.0e-2)

# Fluid flow direction, opposite of gradient (i.e. towards -z)
if ((py.v_f[:,:,:,2] < 0.0).all() and (py.v_f[:,:,:,0:1] < 1.0e-7).all()):
    print("Flow field:\t\t" + passed())
else:
    print("Flow field:\t\t" + failed())
    raise Exception("Failed")

# Convergence rate (2/2)
# This test passes with BETA=0.0 and tolerance=1.0e-9
it = numpy.loadtxt("../output/" + orig.sid + "-conv.log")
if ((it[0:6,1] < 1000).all() and (it[6:,1] < 20).all()):
    print("Convergence rate (2/2):\t" + passed())
else:
    print("Convergence rate (2/2):\t" + failed())

# Long test
orig.p_f[:,:,-1] = 1.1
orig.time_total[0] = 0.1
orig.time_file_dt[0] = orig.time_total[0]/10.0
orig.run(verbose=False)
py.readlast(verbose = False)
ideal_grad_p_z = numpy.linspace(orig.p_f[0,0,0], orig.p_f[0,0,-1], orig.num[2])
#py.writeVTKall()
compareNumpyArraysClose(numpy.zeros((1,orig.num[2])),\
        ideal_grad_p_z - py.p_f[0,0,:],\
        "Pressure gradient (long test):", tolerance=1.0e-2)

# Fluid flow direction, opposite of gradient (i.e. towards -z)
if ((py.v_f[:,:,:,2] < 0.0).all() and (py.v_f[:,:,:,0:1] < 1.0e-7).all()):
    print("Flow field:\t\t" + passed())
else:
    print("Flow field:\t\t" + failed())

# Convergence rate (2/2)
# This test passes with BETA=0.0 and tolerance=1.0e-9
it = numpy.loadtxt("../output/" + orig.sid + "-conv.log")
if (it[0,1] < 700 and it[1,1] < 250 and (it[2:,1] < 20).all()):
    print("Convergence rate (2/2):\t" + passed())
else:
    print("Convergence rate (2/2):\t" + failed())

# Slow pressure modulation test
orig.cleanup()
orig.time_total[0] = 1.0e-1
orig.time_file_dt[0] = 0.101*orig.time_total[0]
orig.setFluidPressureModulation(A=1.0, f=1.0/orig.time_total[0])
#orig.plotPrescribedFluidPressures()
orig.run(verbose=False)
#py.readlast()
#py.plotConvergence()
#py.plotFluidDiffAdvPresZ()
#py.writeVTKall()
for it in range(1,py.status()): # gradient should be smooth in all output files
    py.readstep(it, verbose=False)
    ideal_grad_p_z =\
            numpy.linspace(py.p_f[0,0,0], py.p_f[0,0,-1], py.num[2])
    compareNumpyArraysClose(numpy.zeros((1,py.num[2])),\
            ideal_grad_p_z - py.p_f[0,0,:],\
            'Slow pressure modulation (' + 
            str(it+1) + '/' + str(py.status()) + '):', tolerance=1.0e-1)

'''
# Fast pressure modulation test
orig.time_total[0] = 1.0e-2
orig.time_file_dt[0] = 0.101*orig.time_total[0]
orig.mu[0] = 0.0 # dont let diffusion add transient effects
orig.setFluidPressureModulation(A=1.0, f=1.0/orig.time_total[0])
#orig.plotPrescribedFluidPressures()
orig.run(verbose=False)
#py.plotConvergence()
#py.plotFluidDiffAdvPresZ()
#py.writeVTKall()
for it in range(1,py.status()+1): # gradient should be smooth in all output files
    py.readstep(it, verbose=False)
    #py.plotFluidDiffAdvPresZ()
    ideal_grad_p_z =\
            numpy.linspace(py.p_f[0,0,0], py.p_f[0,0,-1], py.num[2])
    compareNumpyArraysClose(numpy.zeros((1,py.num[2])),\
            ideal_grad_p_z - py.p_f[0,0,:],\
            'Fast pressure modulation (' + 
            str(it) + '/' + str(py.status()) + '):', tolerance=5.0e-1)
'''

cleanup(orig)

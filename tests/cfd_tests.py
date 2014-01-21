#!/usr/bin/env python
from pytestutils import *

import sphere
import sys
import numpy
import matplotlib.pyplot as plt

print("### CFD tests ###")

# Iteration and conservation of mass test
# No gravity, no pressure gradients => no flow
orig = sphere.Spherebin(np = 1e4, nd = 3, nw = 0, sid = "cfdtest", fluid = True)
orig.generateRadii(radius_mean = 0.05, histogram=False)
orig.defaultParams(mu_s = 0.4, mu_d = 0.4, nu = 8.9e-4)
orig.initRandomGridPos(gridnum = numpy.array([40, 40, 1000]), periodic = 1, contactmodel = 1)
orig.initTemporal(total = 0.002, file_dt = 0.001)
orig.time_file_dt = orig.time_dt*0.99
orig.time_total = orig.time_dt*10
orig.initFluid(nu = 0.0)
orig.g[2] = 0.0
orig.writebin(verbose=False)
orig.run(verbose=False)
py = Spherebin(sid = orig.sid, fluid = True)
ones = numpy.ones((orig.num))
py.readlast(verbose = False)
compareNumpyArrays(ones, py.p_f, "Conservation of pressure:")

# Convergence rate (1/2)
it = numpy.loadtxt("../output/" + orig.sid + "-conv.log")
compare(it[:,1].sum(), 0.0, "Convergence rate (1/2):\t")

# Add pressure gradient
# This test passes with BETA=0.0 and tolerance=1.0e-9
orig.p_f[:,:,-1] = 1.1
orig.writebin(verbose=False)
orig.run(verbose=False)
py.readlast(verbose = False)
ideal_grad_p_z = numpy.linspace(orig.p_f[0,0,0], orig.p_f[0,0,-1], orig.num[2])
#py.writeVTKall()
compareNumpyArraysClose(numpy.zeros((1,orig.num[2])),\
        ideal_grad_p_z - py.p_f[0,0,:],\
        "Pressure gradient:\t", tolerance=1.0e-2)

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

# Add viscosity which will limit the fluid flow. Used to test the stress tensor
# in the fluid velocity prediction
#print(numpy.mean(py.v_f[:,:,:,2]))
orig.time_file_dt[0] = 1.0e-4
orig.time_total[0] = 1.0e-3
orig.initFluid(nu = 0.0)
#orig.nu[0] = 4.0
#orig.nu[0] = 0.0
orig.nu[0] = 8.9e-4     # water
orig.p_f[:,:,-1] = 2.0
#orig.time_total[0] = 0.01
#orig.time_file_dt[0] = 0.001
orig.writebin(verbose=False)
orig.run(verbose=False)
#orig.writeVTKall()

#py.plotConvergence()

py.readsecond(verbose=False)
#py.plotFluidDiffAdvPresZ()

# The v_z values are read from sb.v_f[0,0,:,2]
dz = py.L[2]/py.num[2]
rho = 1000.0 # fluid density

# Central difference gradients
dvz_dz = (py.v_f[0,0,1:,2] - py.v_f[0,0,:-1,2])/(2.0*dz)
dvzvz_dz = (py.v_f[0,0,1:,2]**2 - py.v_f[0,0,:-1,2]**2)/(2.0*dz)

# Diffusive contribution to velocity change
dvz_diff = 2.0*py.nu/rho*dvz_dz*py.time_dt

# Advective contribution to velocity change
dvz_adv = dvzvz_dz*py.time_dt

# Diffusive and advective terms should have opposite terms
if ((numpy.sign(dvz_diff) == numpy.sign(-dvz_adv)).all()):
    print("Diffusion-advection (1/2):" + passed())
else:
    print("Diffusion-advection (1/2):" + failed())


py.readlast(verbose=False)
#py.plotFluidDiffAdvPresZ()

# The v_z values are read from sb.v_f[0,0,:,2]
dz = py.L[2]/py.num[2]
rho = 1000.0 # fluid density

# Central difference gradients
dvz_dz = (py.v_f[0,0,1:,2] - py.v_f[0,0,:-1,2])/(2.0*dz)
dvzvz_dz = (py.v_f[0,0,1:,2]**2 - py.v_f[0,0,:-1,2]**2)/(2.0*dz)

# Diffusive contribution to velocity change
dvz_diff = 2.0*py.nu/rho*dvz_dz*py.time_dt

# Advective contribution to velocity change
dvz_adv = dvzvz_dz*py.time_dt

# Diffusive and advective terms should have opposite terms
if ((numpy.sign(dvz_diff) == numpy.sign(-dvz_adv)).all()):
    print("Diffusion-advection (2/2):" + passed())
else:
    print("Diffusion-advection (2/2):" + failed())


# Slow pressure modulation test
orig.time_total[0] = 1.0e-1
orig.time_file_dt[0] = 0.101*orig.time_total[0]
orig.nu[0] = 0.0 # dont let diffusion add transient effects
orig.setFluidPressureModulation(A=1.0, f=1.0/orig.time_total[0])
orig.plotPrescribedFluidPressures()
orig.writebin(verbose=False)
orig.run(verbose=False)
#py.readlast()
#py.plotConvergence()
#py.plotFluidDiffAdvPresZ()
#py.writeVTKall()
for it in range(py.status()): # gradient should be smooth in all output files
    py.readstep(it)
    ideal_grad_p_z =\
            numpy.linspace(py.p_f[0,0,0], py.p_f[0,0,-1], py.num[2])
    compareNumpyArraysClose(numpy.zeros((1,py.num[2])),\
            ideal_grad_p_z - py.p_f[0,0,:],\
            'Slow pressure modulation (' + 
            str(it+1) + '/' + str(py.status()) + '):', tolerance=1.0e-1)

# Fast pressure modulation test
orig.time_total[0] = 1.0e-2
orig.time_file_dt[0] = 0.101*orig.time_total[0]
orig.nu[0] = 0.0 # dont let diffusion add transient effects
orig.setFluidPressureModulation(A=1.0, f=1.0/orig.time_total[0])
orig.plotPrescribedFluidPressures()
orig.writebin(verbose=False)
orig.run(verbose=False)
#py.readlast()
#py.plotConvergence()
#py.plotFluidDiffAdvPresZ()
#py.writeVTKall()
for it in range(py.status()): # gradient should be smooth in all output files
    py.readstep(it)
    ideal_grad_p_z =\
            numpy.linspace(py.p_f[0,0,0], py.p_f[0,0,-1], py.num[2])
    compareNumpyArraysClose(numpy.zeros((1,py.num[2])),\
            ideal_grad_p_z - py.p_f[0,0,:],\
            'Fast pressure modulation (' + 
            str(it+1) + '/' + str(py.status()) + '):', tolerance=1.0e-1)

cleanup(orig)

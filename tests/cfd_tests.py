#!/usr/bin/env python
from pytestutils import *

import sphere
import sys
import numpy

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
orig.time_file_dt[0] = 1.0e-2
orig.time_total[0] = 1.0e-1
orig.initFluid(nu = 0.0)
orig.nu[0] = 4.0
#orig.nu[0] = 0.0
orig.p_f[:,:,-1] = 2.0
#orig.time_total[0] = 0.01
#orig.time_file_dt[0] = 0.001
orig.writebin(verbose=False)
orig.run(verbose=True)
#orig.writeVTKall()
py.readlast(verbose=False)
print(numpy.mean(py.v_f[:,:,:,2]))


# Compare contributions to the velocity from diffusion and advection at top
# boundary, assuming the flow is 1D along the z-axis, phi = 1, and dphi = 0.
# This solution is analog to the predicted velocity and not constrained by the
# conservation of mass.

# The v_z values are read from py.v_f[0,0,:,2]
dz = py.L[2]/py.num[2]

# Central difference gradients
dvz_dz = (py.v_f[0,0,1:,2] - py.v_f[0,0,:-1,2])/(2.0*dz)
dvzvz_dz = (py.v_f[0,0,1:,2]**2 - py.v_f[0,0,:-1,2]**2)/dz  # denominator maybe wrong!

# Diffusive contribution to velocity change
dvz_diff = 2.0*py.nu/1000.0*dvz_dz*py.time_dt

# Advective contribution to velocity change
dvz_adv = dvzvz_dz*py.time_dt



#cleanup(orig)

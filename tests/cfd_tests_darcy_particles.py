#!/usr/bin/env python
from pytestutils import *
import sphere
import numpy


print("### Steady state, no gravity, no forcing, Dirichlet+Dirichlet BCs")
orig = sphere.sim('darcy_particles', np = 1000)
orig.cleanup()
#orig.generateRadii(histogram=False, psd='uni', radius_mean=5.0e-4, radius_variance=5.0e-5)
orig.defaultParams()
orig.generateRadii(psd='uni', mean=5.0e-2, variance=5.0e-5)
orig.initRandomGridPos([20, 20, 200])
orig.initTemporal(total=0.005, file_dt=0.001)
orig.initFluid(cfd_solver=1)
#orig.p_f[5,3,2] *= 1.5
#orig.k_c[0] = 4.6e-15
orig.k_c[0] = 4.6e-10
#orig.g[2] = -10.0
orig.setStiffnessNormal(36.4e9)
orig.setStiffnessTangential(36.4e9/3.0)
orig.run(verbose=False)
#orig.writeVTKall()
py = sphere.sim(sid = orig.sid, fluid = True)
py.readlast(verbose=False)

ones = numpy.ones((orig.num))
py.readlast(verbose = False)
compareNumpyArrays(ones, py.p_f, "Conservation of pressure:")

# Fluid flow should be very small
if ((numpy.abs(py.v_f[:,:,:,:]) < 1.0e-6).all()):
    print("Flow field:\t\t" + passed())
else:
    print("Flow field:\t\t" + failed())
    print(numpy.min(py.v_f))
    print(numpy.mean(py.v_f))
    print(numpy.max(py.v_f))
    raise Exception("Failed")



print("### Steady state, no gravity, no forcing, Neumann+Dirichlet BCs")
orig = sphere.sim('darcy_particles', np = 1000)
orig.cleanup()
#orig.generateRadii(histogram=False, psd='uni', radius_mean=5.0e-4, radius_variance=5.0e-5)
orig.defaultParams()
orig.generateRadii(psd='uni', mean=5.0e-2, variance=5.0e-5)
orig.initRandomGridPos([20, 20, 200])
orig.initTemporal(total=0.005, file_dt=0.001)
orig.initFluid(cfd_solver=1)
#orig.p_f[5,3,2] *= 1.5
#orig.k_c[0] = 4.6e-15
orig.k_c[0] = 4.6e-10
orig.setFluidBottomNoFlow()
#orig.g[2] = -10.0
orig.setStiffnessNormal(36.4e9)
orig.setStiffnessTangential(36.4e9/3.0)
orig.run(verbose=False)
#orig.writeVTKall()
py = sphere.sim(sid = orig.sid, fluid = True)
py.readlast(verbose=False)

ones = numpy.ones((orig.num))
py.readlast(verbose = False)
compareNumpyArrays(ones, py.p_f, "Conservation of pressure:")

# Fluid flow should be very small
if ((numpy.abs(py.v_f[:,:,:,:]) < 1.0e-6).all()):
    print("Flow field:\t\t" + passed())
else:
    print("Flow field:\t\t" + failed())
    print(numpy.min(py.v_f))
    print(numpy.mean(py.v_f))
    print(numpy.max(py.v_f))
    raise Exception("Failed")



print("### Steady state, no gravity, no forcing, Neumann+Neumann BCs")
orig = sphere.sim('darcy_particles', np = 1000)
orig.cleanup()
#orig.generateRadii(histogram=False, psd='uni', radius_mean=5.0e-4, radius_variance=5.0e-5)
orig.defaultParams()
orig.generateRadii(psd='uni', mean=5.0e-2, variance=5.0e-5)
orig.initRandomGridPos([20, 20, 200])
orig.initTemporal(total=0.005, file_dt=0.001)
orig.initFluid(cfd_solver=1)
#orig.p_f[5,3,2] *= 1.5
#orig.k_c[0] = 4.6e-15
orig.k_c[0] = 4.6e-10
orig.setFluidBottomNoFlow()
orig.setFluidTopNoFlow()
#orig.g[2] = -10.0
orig.setStiffnessNormal(36.4e9)
orig.setStiffnessTangential(36.4e9/3.0)
orig.run(verbose=False)
#orig.writeVTKall()
py = sphere.sim(sid = orig.sid, fluid = True)
py.readlast(verbose=False)

ones = numpy.ones((orig.num))
py.readlast(verbose = False)
compareNumpyArrays(ones, py.p_f, "Conservation of pressure:")

# Fluid flow should be very small
if ((numpy.abs(py.v_f[:,:,:,:]) < 1.0e-6).all()):
    print("Flow field:\t\t" + passed())
else:
    print("Flow field:\t\t" + failed())
    print(numpy.min(py.v_f))
    print(numpy.mean(py.v_f))
    print(numpy.max(py.v_f))
    raise Exception("Failed")




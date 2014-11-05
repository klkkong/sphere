#!/usr/bin/env python
from pytestutils import *
import sphere
import numpy

'''
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
'''


print("### Fluidization test: Transient, gravity, Dirichlet+Dirichlet BCs")
#orig = sphere.sim('diffusivity-relax', fluid=False)
orig = sphere.sim('cube-init', fluid=False)
orig.readlast(verbose=False)
orig.id('darcy_fluidization')
orig.cleanup()
orig.initTemporal(total=0.005, file_dt=0.001)
orig.initFluid(cfd_solver=1)
orig.g[2] = -10.0

mean_porosity = orig.bulkPorosity()
fluidize_pressure = -(orig.rho - orig.rho_f) \
        *(1.0 - mean_porosity)*numpy.abs(orig.g[2])

fluid_pressure_gradient = numpy.array([0.1, 0.9, 1.1, 2.0])

for i in numpy.arange(fluid_pressure_gradient.size):

    # set pressure gradient
    dpdz = fluid_pressure_gradient[i] * fluidize_pressure
    dp = dpdz * orig.L[2]
    base_p = 0.0
    orig.p_f[:,:,0] = base_p
    orig.p_f[:,:,-1] = base_p + dp

    orig.run(verbose=True)
    #orig.writeVTKall()
    py = sphere.sim(sid = orig.sid, fluid = True)
    py.readlast(verbose=False)

    print('Mean particle velocity: '
            + str(numpy.mean(py.vel[:,0])) + ', '
            + str(numpy.mean(py.vel[:,1])) + ', '
            + str(numpy.mean(py.vel[:,2])) + ' m/s')

    z_vel_threshold = 0.001
    if fluid_pressure_gradient[i] < 1.0:
        test('Fluidization (' + str(i) + '):\t',
                numpy.mean(py.vel[:,2]) < z_vel_threshold)
    elif fluid_pressure_gradient[i] > 1.0:
        test('Fluidization (' + str(i) + '):\t',
                numpy.mean(py.vel[:,2]) > z_vel_threshold)



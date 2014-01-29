#!/usr/bin/env python
'''
Validate the implemented contact models by observing the behavior of one or two
particles.
'''

import sphere
import numpy
import pytestutils

### Wall-particle interaction ##################################################

## Linear elastic collisions

# Normal impact: Check for conservation of momentum (sum(v_i*m_i))
orig = sphere.Spherebin(np=1, nw=0, sid='contactmodeltest')
orig.radius[:] = 1.0
orig.x[0,:] = [5.0, 5.0, 1.05]
orig.vel[0,2] = -0.1
orig.defineWorldBoundaries(L=[10,10,10])
orig.gamma_wn[0] = 0.0  # Disable wall viscosity
orig.gamma_wt[0] = 0.0  # Disable wall viscosity
orig.initTemporal(total = 1.0, file_dt = 0.01)
orig.writebin(verbose=False)
moment_before = orig.totalMomentum()
orig.run(verbose=False)
#orig.writeVTKall()
orig.readlast(verbose=False)
moment_after = orig.totalMomentum()
print(moment_before)
print(moment_after)
pytestutils.compareFloats(moment_before, moment_after, "Normal wall collision:")

''' This test isn't useful unless there is a tangential force component
# Oblique impact: Check for conservation of momentum (sum(v_i*m_i))
orig = sphere.Spherebin(np=1, nw=0, sid='contactmo')
orig.radius[:] = 1.0
orig.x[0,:] = [5.0, 5.0, 1.05]
orig.vel[0,1] =  0.1
orig.vel[0,2] = -0.1
orig.defineWorldBoundaries(L=[10,10,10])
orig.gamma_wn[0] = 0.0  # Disable wall viscosity
orig.gamma_wt[0] = 0.0  # Disable wall viscosity
orig.initTemporal(total = 1.0, file_dt = 0.01)
orig.writebin(verbose=False)
moment_before = orig.totalMomentum()
orig.run(verbose=False)
#orig.writeVTKall()
orig.readlast(verbose=False)
moment_after = orig.totalMomentum()
pytestutils.compareFloats(moment_before, moment_after,\
        "45 deg. wall collision:")
'''


## Visco-elastic collisions

# Normal impact: Check for conservation of momentum (sum(v_i*m_i))
orig = sphere.Spherebin(np=1, nw=0, sid='contactmodeltest')
orig.radius[:] = 1.0
orig.x[0,:] = [5.0, 5.0, 1.05]
orig.vel[0,2] = -0.1
orig.defineWorldBoundaries(L=[10,10,10])
orig.gamma_wn[0] = 1.0e6
orig.gamma_wt[0] = 0.0
orig.initTemporal(total = 1.0, file_dt = 0.01)
orig.writebin(verbose=False)
moment_before = orig.totalMomentum()
orig.run(verbose=False)
#orig.writeVTKall()
orig.readlast(verbose=False)
moment_after = orig.totalMomentum()
pytestutils.compareFloats(moment_before, moment_after, "Normal wall collision:")
print(moment_before)
print(moment_after)

# Oblique impact: Check for conservation of momentum (sum(v_i*m_i))
orig = sphere.Spherebin(np=1, nw=0, sid='contactmodeltest')
orig.radius[:] = 1.0
orig.x[0,:] = [5.0, 5.0, 1.05]
orig.vel[0,1] =  0.1
orig.vel[0,2] = -0.1
orig.defineWorldBoundaries(L=[10,10,10])
orig.gamma_wn[0] = 1.0e6
orig.gamma_wt[0] = 1.0e6
orig.initTemporal(total = 1.0, file_dt = 0.01)
orig.writebin(verbose=False)
moment_before = orig.totalMomentum()
orig.run(verbose=False)
#orig.writeVTKall()
orig.readlast(verbose=False)
moment_after = orig.totalMomentum()
pytestutils.compareFloats(moment_before, moment_after,\
        "45 deg. wall collision:")
print(moment_before)
print(moment_after)



#sphere.cleanup(orig)

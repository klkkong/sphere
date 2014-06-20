#!/usr/bin/env python
'''
Validate the implemented contact models by observing the behavior of two
particles.
'''

import sphere
import numpy
import pytestutils

### Particle-particle interaction ##############################################

## Linear elastic collisions

# Normal impact: Check for conservation of momentum (sum(v_i*m_i))
orig = sphere.sim(np=2, sid='contactmodeltest')
after = sphere.sim(np=2, sid='contactmodeltest')
sphere.cleanup(orig)
#orig.radius[:] = [1.0, 2.0]
orig.radius[:] = [1.0, 1.0]
orig.x[0,:] = [5.0, 5.0, 2.0]
orig.x[1,:] = [5.0, 5.0, 4.05]
v_orig = 1
orig.vel[0,2] = v_orig
orig.defineWorldBoundaries(L=[10,10,10])
orig.initTemporal(total = 0.1, file_dt = 0.01)

orig.run(verbose=False)
after.readlast(verbose=False)
pytestutils.compareFloats(orig.vel[0,2], after.vel[1,2],\
        "Elastic normal collision (1/4):")
#print(orig.totalKineticEnergy())
#print(after.totalKineticEnergy())
pytestutils.compareFloats(orig.totalKineticEnergy(), after.totalKineticEnergy(),\
        "Elastic normal collision (2/4):")

# Normal impact with different sizes: Check for conservation of momentum
orig = sphere.sim(np=2, sid='contactmodeltest')
after = sphere.sim(np=2, sid='contactmodeltest')
sphere.cleanup(orig)
orig.radius[:] = [2.0, 1.0]
orig.x[0,:] = [5.0, 5.0, 2.0]
orig.x[1,:] = [5.0, 5.0, 5.05]
orig.vel[0,2] = 1.0
orig.defineWorldBoundaries(L=[10,10,10])
orig.initTemporal(total = 0.1, file_dt = 0.01)

orig.run(verbose=False)
after.readlast(verbose=False)
pytestutils.compareFloats(orig.totalKineticEnergy(), after.totalKineticEnergy(),\
        "Elastic normal collision (3/4):")

# Normal impact with different sizes: Check for conservation of momentum
orig = sphere.sim(np=2, sid='contactmodeltest')
after = sphere.sim(np=2, sid='contactmodeltest')
sphere.cleanup(orig)
orig.radius[:] = [1.0, 2.0]
orig.x[0,:] = [5.0, 5.0, 2.0]
orig.x[1,:] = [5.0, 5.0, 5.05]
orig.vel[0,2] = 1.0
orig.defineWorldBoundaries(L=[10,10,10])
orig.initTemporal(total = 0.1, file_dt = 0.01)

orig.run(verbose=False)
after.readlast(verbose=False)
pytestutils.compareFloats(orig.totalKineticEnergy(), after.totalKineticEnergy(),\
        "Elastic normal collision (4/4):")




#orig.cleanup()

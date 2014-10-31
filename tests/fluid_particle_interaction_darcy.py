#!/usr/bin/env python
import sphere
from pytestutils import *

sim = sphere.sim('fluid_particle_interaction', fluid=True)
sim.cleanup()

sim.defineWorldBoundaries([1.0, 1.0, 1.0], dx = 0.1)
sim.initFluid(cfd_solver = 1)


# No gravity, pressure gradient enforced by Dirichlet boundaries.
# The particle should be sucked towards the low pressure
print('# Test 1: Test pressure gradient force')
sim.p_f[:,:,0]  = 1.0
sim.p_f[:,:,-1] = 1.1
sim.addParticle([0.5, 0.5, 0.5], 0.05)
sim.initTemporal(total=0.01, file_dt=0.001)
#sim.time_file_dt[0] = sim.time_dt[0]
#sim.time_total[0] = sim.time_dt[0]

#sim.run(verbose=False)
sim.run()
#sim.run(dry=True)
#sim.run(cudamemcheck=True)
#sim.writeVTKall()

sim.readlast()
test(sim.vel[0,2] < 0.0, 'Particle velocity:')

sim.cleanup()

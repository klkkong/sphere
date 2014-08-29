#!/usr/bin/env python

# This script simulates the effect of capillary cohesion on a sand pile put on a
# desk.

# start with
# $ python capillary-cohesion.py <DEVICE> <COHESION> <GRAVITY>
# where DEVICE specifies the index of the GPU (0 is the most common value).
# COHESION should have the value of 0 or 1. 0 denotes a dry simulation without
# cohesion, 1 denotes a wet simulation with capillary cohesion.
# GRAVITY toggles gravitational acceleration. Without it, the particles are
# placed in the middle of a volume. With it enabled, the particles are put on
# top of a flat wall.

import sphere
import numpy
import sys

device = sys.argv[1]
cohesion = sys.argv[2]
gravity = sys.argv[3]

# Create packing
sim = sphere.sim('cap-cohesion=' + str(cohesion) + '-init', np=2000)
sim.mu_s[0] = 0.0
sim.mu_d[0] = 0.0
sim.generateRadii(psd='logn', radius_mean=1.0e-3, radius_variance=1.0e-4)
sim.contactModel(1)
sim.initRandomGridPos([12, 12, 10000])
sim.initTemporal(5.0, file_dt=0.01, epsilon=0.07)
sim.g[2] = -10.0
sim.run()

sim.readlast()
sim.sid = 'cap-cohesion=' + str(cohesion)
sim.defaultParams(capillaryCohesion=cohesion)
sim.adjustUpperWall()
init_lx = sim.L[0]
init_ly = sim.L[1]
sim.L[0] *= 5
sim.L[1] *= 5
sim.num[0] *= 5
sim.num[1] *= 5
sim.x[:,0] += 0.5*sim.L[0] - 0.5*init_lx
sim.x[:,1] += 0.5*sim.L[1] - 0.5*init_ly

if gravity == 0:
    init_lz = sim.L[2]
    sim.L[2] *= 5
    sim.num[2] *= 5
    sim.w_x[0] = sim.L[2]
    sim.x[:,2] += 0.5*sim.L[2] - 0.5*init_lz
    sim.g[2] = 0.0

sim.initTemporal(2.0, file_dt=0.01, epsilon=0.07)
sim.run()

sim.writeVTKall()

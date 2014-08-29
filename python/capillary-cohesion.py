#!/usr/bin/env python

# This script simulates the effect of capillary cohesion on a sand pile put on a
# desk.

# start with
# $ python capillary-cohesion.py <DEVICE> <COHESION>
# where DEVICE specifies the index of the GPU (0 is the most common value).
# COHESION should have the value of 0 or 1. 0 denotes a dry simulation without
# cohesion, 1 denotes a wet simulation with capillary cohesion.

import sphere
import numpy
import sys

device = sys.argv[1]
cohesion = sys.argv[2]

sim = sphere.sim('cap-cohesion=' + str(cohesion), np=2000)
sim.defaultParams(capillaryCohesion = cohesion)
sim.generateRadii(psd='logn', radius_mean=1.0e-3, radius_variance=1.0e-4)
sim.contactModel(1)
sim.initRandomGridPos([12, 12, 10000])
sim.initTemporal(2.0, file_dt=0.01, epsilon=0.07)
sim.run()

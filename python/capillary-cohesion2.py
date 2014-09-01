#!/usr/bin/env python

# This script simulates the effect of capillary cohesion on a sand pile put on a
# desk.

# start with
# $ python capillary-cohesion.py <DEVICE> <COHESION>
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

cube = sphere.sim('cube-init')
cube.readlast()
cube.adjustUpperWall(z_adjust=1.0)

# Fill out grid with cubic packages
grid = numpy.array((
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]))

# World dimensions and cube grid
nx = 1                # horizontal (thickness) cubes
ny = grid.shape[1]    # horizontal cubes
nz = grid.shape[0]    # vertical cubes
dx = cube.L[0]
dy = cube.L[1]
dz = cube.L[2]
Lx = dx*nx
Ly = dy*ny
Lz = dz*nz

sim = sphere.sim('cap2-cohesion=' + str(cohesion), nw=0)

for z in range(nz):
    for y in range(ny):
        for x in range(nx):

            if (grid[z,y] == 0):
                continue # skip to next iteration

            for i in range(cube.np):
                # x=x, y=y, z=z
                pos = [ cube.x[i,0] + x*dx,
                        cube.x[i,1] + y*dy,
                        cube.x[i,2] + z*dz ]
                sim.addParticle(pos, radius=cube.radius[i], color=grid[z,y])

sim.checkerboardColors()
sim.defaultParams(capillaryCohesion=cohesion)
sim.g[2] = -10.0
sim.run(device=device)

sim.writeVTKall()

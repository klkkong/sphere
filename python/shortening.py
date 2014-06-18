#!/usr/bin/env python
import sphere
import numpy

cube = sphere.sim('cube-init')
cube.readlast()
cube.adjustUpperWall(z_adjust=1.0)

# Fill out grid with cubic packages
grid = numpy.array((
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ))

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

sim = sphere.sim('shortening', nw=1)

# insert particles into each cube in 90 degree CCW rotated coordinate system
# around y
for z in range(nz):
    for y in range(ny):
        for x in range(nx):

            if (grid[z,y] == 0):
                continue # skip to next iteration

            for i in range(cube.np):
                # x=x, y=Ly-z, z=y
                pos = [ cube.x[i,0] + x*dx,
                        Ly - ((dz - cube.x[i,2]) + z*dz),
                        cube.x[i,1] + y*dy ]
                sim.addParticle(pos, radius=cube.radius[i])

sim.defineWorldBoundaries(L=[Lx, Lz*3, Ly])

sim.k_t[0] = 2.0/3.0*sim.k_n[0]

sim.writeVTK()
print(sim.np[0])


'''
## Relaxation
# Choose the tangential contact model
# 1) Visco-frictional (somewhat incorrect, fast computations)
# 2) Elastic-viscous-frictional (more correct, slow computations in dense
# packings)
sim.contactmodel[0] = 2

# Add gravitational acceleration
# Flip geometry so the upper wall pushes downwards
sim.g[1] = -9.81

sim.periodicBoundariesX()

# Set duration of simulation, automatically determine timestep, etc.
sim.initTemporal(total=3.0, file_dt = 0.1)

sim.zeroKinematics()
sim.run(dry=True)
sim.run()
sim.writeVTKall()
'''

'''
## Shortening
sim.readlast()
sim.initTemporal(current=0.0, total=5.0, file_dt = 0.01)

# push down upper wall
compressional_strain = 0.5
sim.uniaxialStrainRate(wvel = compressional_strain*Lx/sim.time_total[0])

sim.run(dry=True)
sim.run()
sim.writeVTKall()
'''

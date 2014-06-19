#!/usr/bin/env python
import sphere
import numpy

cube = sphere.sim('cube-init')
cube.readlast()
cube.adjustUpperWall(z_adjust=1.0)

# Fill out grid with cubic packages
grid = numpy.array((
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
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

sim = sphere.sim('shortening-relaxation', nw=0)

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
                sim.addParticle(pos, radius=cube.radius[i], color=grid[z,y])

# move to y=0
min_y = numpy.min(sim.x[:,1] - sim.radius[:])
sim.x[:,1] = sim.x[:,1] - min_y 

# move to z=0
min_z = numpy.min(sim.x[:,2] - sim.radius[:])
sim.x[:,2] = sim.x[:,2] - min_z 

sim.defineWorldBoundaries(L=[Lx, Lz*3, Ly])
sim.k_t[0] = 2.0/3.0*sim.k_n[0]

sim.cleanup()
sim.writeVTK()
print(sim.np[0])


## Relaxation

# Add gravitational acceleration
# Flip geometry so the upper wall pushes downwards
sim.g[0] = 0
sim.g[1] = -9.81
sim.g[2] = 0

sim.gamma_wn[0] = 1.0e4
sim.mu_ws[0] = 0.0
sim.mu_wd[0] = 0.0

sim.gamma_n[0] = 1.0e2
sim.mu_s[0] = 0.0
sim.mu_d[0] = 0.0

sim.periodicBoundariesX()
sim.uniaxialStrainRate(wvel = 0.0)

# Set duration of simulation, automatically determine timestep, etc.
sim.initTemporal(total=3.0, file_dt = 0.01)
sim.zeroKinematics()

sim.run(dry=True)
sim.run()
sim.writeVTKall()


'''
## Shortening
sim = sphere.sim('shortening-relaxation', nw=1)
sim.readlast()
sim.sid = 'shortening'
sim.cleanup()
sim.initTemporal(current=0.0, total=5.0, file_dt = 0.01)

# set colors again
color_ny = 6
y_max = numpy.max(sim.x[:,1])
color_dy = y_max/color_ny
color_y = numpy.arange(0.0, y_max, ny)
for i in range(ny-1):
    I = numpy.nonzero((sim.x[:,1] >= color_y[i]) & (sim.x[:,1] <= color_y[i+1]))
    sim.color[I] = i%2 + 1

sim.mu_s[0] = 0.5
sim.mu_d[0] = 0.5

# push down upper wall
compressional_strain = 0.5
sim.uniaxialStrainRate(wvel = -compressional_strain*Lx/sim.time_total[0])

sim.run(dry=True)
sim.run()
sim.writeVTKall()
'''

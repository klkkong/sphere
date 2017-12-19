#!/usr/bin/env python

# Import sphere functionality
import sphere

### EXPERIMENT SETUP ###
initialization = True
consolidation  = True
shearing       = True
rendering      = False
plots          = True

# Number of particles
np = 2e4

# Common simulation id
sim_id = "shear-test-ocr"

# Effective normal stresses during consolidation [Pa]
Nlist = [5e3, 10e3, 25e3, 50e3, 100e3, 250e3, 500e3]

# Effective normal stresses during relaxation and shear [Pa]
Nshear = 10e3

### INITIALIZATION ###

# New class
init = sphere.sim(np = np, nd = 3, nw = 0, sid = sim_id + "-init")

# Save radii with uniform size distribution
init.generateRadii(mean = 9e-4, variance = 3e-4, histogram = True)

# Use default params
init.defaultParams(gamma_n = 1e2, mu_s = 0.5, mu_d = 0.5)
init.setYoungsModulus(7e8)

# Add gravity
init.g[2] = -9.81

# Periodic x and y boundaries
init.periodicBoundariesXY()

# Initialize positions in random grid (also sets world size)
hcells = np**(1.0/3.0)
init.initRandomGridPos(gridnum = [hcells, hcells, 1e9])

# Set duration of simulation
init.initTemporal(total = 10.0)

if (initialization == True):

    # Run sphere
    init.run(dry = True)
    init.run()

    if (plots == True):
        # Make a graph of energies
        init.visualize('energy')

    init.writeVTKall()

    if (rendering == True):
        # Render images with raytracer
        init.render(method = "angvel", max_val = 0.3, verbose = False)


# For each normal stress, consolidate and subsequently shear the material
for N in Nlist:

    ### CONSOLIDATION ###

    # New class
    cons = sphere.sim(np = init.np, nw = 1, sid = sim_id +
                      "-cons-N{}".format(N))

    # Read last output file of initialization step
    lastf = status(sim_id + "-init")
    cons.readbin("../output/" + sim_id + "-init.output{:0=5}.bin".format(lastf), verbose=False)
    cons.gamma_n[0] = 0.

    # Periodic x and y boundaries
    cons.periodicBoundariesXY()

    # Setup consolidation experiment
    cons.consolidate(normal_stress = N, periodic = init.periodic)
    cons.adaptiveGrid()

    # Set duration of simulation
    cons.initTemporal(total = 1.5)

    if (consolidation == True):

        # Run sphere
        cons.run(dry = True) # show values, don't run
        cons.run() # run

        if (plots == True):
            # Make a graph of energies
            cons.visualize('energy')
            cons.visualize('walls')

        cons.writeVTKall()

        if (rendering == True):
            # Render images with raytracer
            cons.render(method = "pres", max_val = 2.0*N, verbose = False)


    ### RELAXATION at Nshear ###
    relax = sphere.sim(np = cons.np, nw = cons.nw, sid = sim_id +
                       "-relax-from-N{}".format(N))
    lastf = status(sim_id + "-cons-N{}".format(N))
    relax.readbin("../output/" + sim_id +
                  "-cons-N{}.output{:0=5}.bin".format(N, lastf))

    relax.periodicBoundariesXY()

    # Setup relaxation experiment
    relax.consolidate(normal_stress = Nshear, periodic = init.periodic)
    relax.adaptiveGrid()

    # Set duration of simulation
    relax.initTemporal(total = 1.0)

    if (relaxation == True):

        # Run sphere
        relax.run(dry = True) # show values, don't run
        relax.run() # run

        if (plots == True):
            # Make a graph of energies
            relax.visualize('energy')
            relax.visualize('walls')

        relax.writeVTKall()

        if (rendering == True):
            # Render images with raytracer
            relax.render(method = "pres", max_val = 2.0*Nshear, verbose = False)

    ### SHEARING ###

    # New class
    shear = sphere.sim(np = relax.np, nw = relax.nw, sid = sim_id +
                       "-shear-N{}-OCR{}".format(Nshear, N/Nshear))

    # Read last output file of initialization step
    lastf = status(sim_id + "-relax-from-N{}".format(N))
    shear.readbin("../output/" + sim_id +
                  "-relax-from-N{}.output{:0=5}.bin".format(N, lastf),
                  verbose = False)

    # Periodic x and y boundaries
    shear.periodicBoundariesXY()

    # Setup shear experiment
    shear.shear(shear_strain_rate = 0.05, periodic = init.periodic)
    shear.adaptiveGrid()

    # Set duration of simulation
    shear.initTemporal(total = 20.0)

    if (shearing == True):

        # Run sphere
        shear.run(dry = True)
        shear.run()

        if (plots == True):
            # Make a graph of energies
            shear.visualize('energy')
            shear.visualize('shear')

        shear.writeVTKall()

        if (rendering == True):
            # Render images with raytracer
            shear.render(method = "pres", max_val = 2.0*N, verbose = False)

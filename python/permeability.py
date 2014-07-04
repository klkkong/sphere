#!/usr/bin/env python
import sphere
import numpy


for dp in [1.0e3, 2.0e3, 4.0e3, 10.0e3, 20.0e3, 40.0e3, 80.0e3]:
    # Read initial configuration
    sim.sid = 'diffusivity-relax'
    sim.readlast()

    sim.sid = 'permeability-dp=' + str(dp)
    sim.cleanup()

    sim.g[2] = -9.81
    sim.nw[0] = 0
    sim.initGrid()
    sim.zeroKinematics()
    sim.initFluid(mu = 17.87e-4, p = 1.0e5, hydrostatic=True)  # mu = water at 0 deg C
    sim.setFluidTopFixedPressure()
    sim.p_f[:,:,-1] = dp
    #sim.setDEMstepsPerCFDstep(100)
    sim.setDEMstepsPerCFDstep(10)
    sim.initTemporal(total = 2.0, file_dt = 0.01, epsilon=0.07)
    sim.run(dry=True)
    sim.run()
    sim.writeVTKall()

#!/usr/bin/env python
import sphere

fluid = True

sim = sphere.sim('cons2-20kPa')
sim.readlast()
sim.id('shear2-20kPa-c=1.0')
sim.shear(1.0/20.0)

if fluid:
    sim.num[2] *= 2
    sim.L[2] *= 2.0
    sim.initFluid(mu=1.797e-6, p=600.0e3, hydrostatic=True)
    sim.setFluidBottomNoFlow()
    sim.setFluidTopFixedPressure()
    sim.setDEMstepsPerCFDstep(100)
    sim.setMaxIterations(2e5)
    sim.c_grad_p[0] = 1.0

sim.checkerboardColors()
sim.initTemporal(20.0, epsilon=0.07)
sim.run()
sim.writeVTKall()

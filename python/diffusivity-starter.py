#!/usr/bin/env python
import sphere
import numpy
import sys

# launch with:
# $ python diffusivity-starter <DEVICE> <C_PHI> <C_GRAD_P> <DP_1, DP_2, ...>

for sigma0_str in sys.argv[4:]:

    sigma0 = float(sigma0_str)
    device = int(sys.argv[1])
    c_phi = float(sys.argv[2])
    c_grad_p = float(sys.argv[3])

    sim = sphere.sim('diffusivity-relax')
    sim.readlast()

    sim.sid = 'permeability-dp=' + str(dp) + '-c_phi=' + str(c_phi) + \
            '-c_grad_p=' + str(c_grad_p)
    print(sim.sid)

    # Checkerboard colors
    x_min = numpy.min(sim.x[:,0])
    x_max = numpy.max(sim.x[:,0])
    y_min = numpy.min(sim.x[:,1])
    y_max = numpy.max(sim.x[:,1])
    z_min = numpy.min(sim.x[:,2])
    z_max = numpy.max(sim.x[:,2])
    color_nx = 6
    color_ny = 6
    color_nz = 6
    for i in range(sim.np):
        ix = numpy.floor((sim.x[i,0] - x_min)/(x_max/color_nx))
        iy = numpy.floor((sim.x[i,1] - y_min)/(y_max/color_ny))
        iz = numpy.floor((sim.x[i,2] - z_min)/(z_max/color_nz))
        sim.color[i] = (-1)**ix + (-1)**iy + (-1)**iz

    sim.cleanup()
    sim.adjustUpperWall()
    sim.zeroKinematics()
    sim.consolidate(normal_stress = 10.0e3)
    sim.initFluid(mu = 17.87e-4, p = 1.0e5, hydrostatic = True)
    sim.setFluidBottomNoFlow()
    sim.setFluidTopFixedPressure()
    sim.setDEMstepsPerCFDstep(10)
    sim.setMaxIterations(2e5)
    sim.initTemporal(total = 5.0, file_dt = 0.01, epsilon=0.07)
    sim.run(dry=True)
    sim.run(device=0)
    #sim.writeVTKall()
    sim.visualize('walls')
    sim.visualize('fluid-pressure')

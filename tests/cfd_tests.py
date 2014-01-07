#!/usr/bin/env python
from pytestutils import *

# Import sphere functionality
import sphere
from sphere import visualize, status
import sys
import numpy

#### Input/output tests ####
print("### CFD tests ###")

# Steady-state test
orig = sphere.Spherebin(np = 1e4, nd = 3, nw = 0, sid = "cfdtest")
orig.generateRadii(radius_mean = 0.05, histogram=False)
orig.defaultParams(mu_s = 0.4, mu_d = 0.4, nu = 8.9e-4)
orig.initRandomGridPos(gridnum = numpy.array([40, 40, 1000]), periodic = 1, contactmodel = 1)
orig.initTemporal(total = 0.002, file_dt = 0.001)
orig.initFluid(nu = 8.9e-4)
orig.g[2] = 0.0
orig.writebin()
orig.run(verbose=False, cfd=True)
#orig.writeVTKall()
py = Spherebin(sid=orig.sid)
ones = numpy.ones((orig.num))
py.readlast()
compareNumpyArrays(ones, py.p_f, "Fluid pressure conservation:")


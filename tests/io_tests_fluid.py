#!/usr/bin/env python
from pytestutils import *
import sphere

#### Input/output tests ####
print("### Fluid input/output tests ###")

# Generate data in python
orig = sphere.sim(np=100, sid="test-initgrid-fluid", fluid=True)
orig.generateRadii(histogram=False, radius_mean=1.0)
orig.defaultParams()
orig.initRandomGridPos()
orig.initFluid()
orig.initTemporal(current=0.0, total=0.0)
orig.time_total=2.0*orig.time_dt
orig.time_file_dt = orig.time_dt
orig.writebin(verbose=False)

# Test Python IO routines
py = sphere.sim(fluid=True)
py.readbin("../input/" + orig.sid + ".bin", verbose=False)
compare(orig, py, "Python IO:")

# Test C++ IO routines
#orig.run(verbose=True, hideinputfile=True)
orig.run(verbose=True, hideinputfile=False)
cpp = sphere.sim(fluid=True)
cpp.readbin("../output/" + orig.sid + ".output00000.bin", verbose=False)
compare(orig, cpp, "C++ IO:   ")

# Test CUDA IO routines
cuda = sphere.sim(fluid=True)
cuda.readbin("../output/" + orig.sid + ".output00001.bin", verbose=False)
cuda.time_current = orig.time_current
cuda.time_step_count = orig.time_step_count
compareNumpyArraysClose(orig.v_f, cuda.v_f, "cuda.v_f:", tolerance=1e-5)
cuda.v_f = orig.v_f
compareNumpyArraysClose(orig.p_f, cuda.p_f, "cuda.p_f:", tolerance=0.1)
cuda.p_f = orig.p_f
compare(orig, cuda, "CUDA IO:  ")

# Remove temporary files
cleanup(orig)

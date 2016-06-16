#!/usr/bin/env python
import sphere

jobname_prefix = 'rs0-'

# Simulation parameter values
effective_stresses = [10e3, 20e3, 100e3, 200e3, 1000e3, 2000e3]
velfacs = [0.1, 1.0, 10.0]
mu_s_vals = [0.5]
mu_d_vals = [0.5]

# Loop through parameter values
for effective_stress in effective_stresses:
    for velfac in velfacs:
        for mu_s in mu_s_vals:
            for mu_d in mu_s_vals:

                jobname = jobname_prefix + '{}Pa-v={}-mu_s={}-mu_d={}'.format(
                    effective_stress,
                    velfac,
                    mu_s,
                    mu_d)

                print(jobname)
                sim = sphere.sim(jobname, fluid=False)
                #sim.readlast()
                sim.visualize('shear')

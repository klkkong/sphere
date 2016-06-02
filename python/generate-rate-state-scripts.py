#!/usr/bin/env python
import subprocess

# Account and cluster information
# https://portal.xsede.org/sdsc-comet
# https://www.sdsc.edu/support/user_guides/comet.html
account = 'csd492'  # from `show_accounts`
jobname_prefix = 'rs0-'
walltime = '2-0'   # hours:minutes:seconds or days-hours
partition = 'gpu-shared'
no_gpus = 1
no_nodes = 1
ntasks_per_node = 1

# Simulation parameter values
effective_stresses = [10e3, 20e3, 100e3, 200e3, 1000e3, 2000e3]
velfacs = [0.1, 1.0, 10.0]

for effective_stress in effective_stresses:
    for velfac in velfacs:

        jobname = 'rs0-' + str(effective_stress) + 'Pa-v=' + str(velfac)

        # Generate scripts for queue manager, submit with `sbatch <script>`
        generate_slurm_script(jobname,
                              effective_stress,
                              velfac)

        generate_slurm_continue_script(jobname,
                                       effective_stress,
                                       velfac)

        generate_simulation_script(jobname,
                                   effective_stress,
                                   velfac)

        generate_simulation_continue_script(jobname,
                                            effective_stress,
                                            velfac)

def generate_slurm_script(effective_stress, velfac):

    slurm_script = '''#!/bin/bash
    #SBATCH -A {account}
    #SBATCH --job-name
    '''.format(account)

slurm_continue_script = '''
'''

# Generate scripts for sphere
simulation_script = '''#!/usr/bin/env python
'''

simulation_continue_script = '''#!/usr/bin/env python
'''

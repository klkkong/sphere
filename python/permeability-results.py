#!/usr/bin/env python
from permeability-calculator import *
        
pc = PermeabilityCalc('permeability-dp=1000.0')
pc = PermeabilityCalc('permeability-dp=1000.0-c_phi=1.0-c_grad_p=0.01')
pc = PermeabilityCalc('permeability-dp=1000.0-c_phi=1.0-c_grad_p=0.5')

pc = PermeabilityCalc('permeability-dp=20000.0-c_phi=1.0-c_grad_p=0.01')
pc = PermeabilityCalc('permeability-dp=20000.0-c_phi=1.0-c_grad_p=0.1')
pc = PermeabilityCalc('permeability-dp=20000.0-c_phi=1.0-c_grad_p=0.5')

pc = PermeabilityCalc('permeability-dp=4000.0')
pc = PermeabilityCalc('permeability-dp=4000.0-c_phi=1.0-c_grad_p=0.01')
pc = PermeabilityCalc('permeability-dp=4000.0-c_phi=1.0-c_grad_p=0.1')
pc = PermeabilityCalc('permeability-dp=4000.0-c_phi=1.0-c_grad_p=0.5')

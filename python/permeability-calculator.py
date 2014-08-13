#!/usr/bin/env python
import sys
import sphere
import numpy
import matplotlib.pyplot as plt

class PermeabilityCalc:
    ''' Darcy's law: Q = -k*A/mu * dP '''

    def __init__(self, sid):
        self.sid = sid
        self.readfile()
        self.findPermeability()
        self.findConductivity()
        self.findMeanPorosity()

    def readfile(self):
        self.sim = sphere.sim(self.sid, fluid=True)
        self.sim.readlast()

    def findPermeability(self):
        self.findCellSpacing()
        self.findCrossSectionalArea()
        self.findCrossSectionalFlux()
        self.findPressureGradient()
        self.k = -self.Q*self.sim.mu/(self.A*self.dP) # m^2

    def findConductivity(self):
        # hydraulic conductivity
        self.K = self.k/self.sim.mu # m/s

    def findMeanPorosity(self):
        ''' calculate mean porosity in cells beneath the top wall '''

        if (self.sim.nw > 0):
            wall_iz = int(self.sim.w_x[0]/self.dx[2])
            self.phi_bar = numpy.mean(self.sim.phi[:,:,0:wall_iz-1])
        else:
            self.phi_bar = numpy.mean(self.sim.phi[:,:,0:-3])
                

    def findCrossSectionalArea(self):
        ''' Cross sectional area normal to each axis '''
        self.A = numpy.array([
            self.sim.L[1]*self.sim.L[2],
            self.sim.L[0]*self.sim.L[2],
            self.sim.L[0]*self.sim.L[1]])

    def findCellSpacing(self):
        self.dx = numpy.array([
            self.sim.L[0]/self.sim.num[0],
            self.sim.L[1]/self.sim.num[1],
            self.sim.L[2]/self.sim.num[2]])

    def findCrossSectionalFlux(self):
        ''' Flux along each axis, measured at the outer boundaries '''
        #self.Q = numpy.array([
            #numpy.mean(self.sim.v_f[-1,:,:]),
            #numpy.mean(self.sim.v_f[:,-1,:]),
            #numpy.mean(self.sim.v_f[:,:,-1])])*self.A

        self.Q = numpy.zeros(3)

        self.A_cell = numpy.array([
            self.dx[1]*self.dx[2],
            self.dx[0]*self.dx[2],
            self.dx[0]*self.dx[1]])

        # x axis (0)
        for y in numpy.arange(self.sim.num[1]):
            for z in numpy.arange(self.sim.num[2]):
                self.Q[0] += self.sim.v_f[-1,y,z,0] * self.A_cell[0]

        # y axis (1)
        for x in numpy.arange(self.sim.num[0]):
            for z in numpy.arange(self.sim.num[2]):
                self.Q[1] += self.sim.v_f[x,-1,z,1] * self.A_cell[1]

        # z axis (2)
        for x in numpy.arange(self.sim.num[0]):
            for y in numpy.arange(self.sim.num[1]):
                self.Q[2] += self.sim.v_f[x,y,-1,2] * self.A_cell[2]

    def findPressureGradient(self):
        ''' Determine pressure gradient by finite differencing the
        mean values at the outer boundaries '''
        self.dP = numpy.array([
            numpy.mean(self.sim.p_f[-1,:,:]) - numpy.mean(self.sim.p_f[0,:,:]),
            numpy.mean(self.sim.p_f[:,-1,:]) - numpy.mean(self.sim.p_f[:,0,:]),
            numpy.mean(self.sim.p_f[:,:,-1]) - numpy.mean(self.sim.p_f[:,:,0])
            ])/self.sim.L

    def printResults(self):
        print('\n### Permeability resuts for "' + self.sid + '" ###')
        print('Pressure gradient: dP = ' + str(self.dP) + ' Pa/m')
        print('Flux: Q = ' + str(self.Q) + ' m^3/s')
        print('Intrinsic permeability: k = ' + str(self.k) + ' m^2')
        print('Saturated hydraulic conductivity: K = ' + str(self.K) + ' m/s')
        print('Mean porosity: phi_bar = ' + str(self.phi_bar) + '\n')

    def plotEvolution(self, axis=2, outformat='png'):
        ''' Plot temporal evolution of parameters on the selected axis '''
        t = numpy.linspace(0.0, self.sim.time_total, self.sim.status())
        Q = numpy.empty((self.sim.status(), 3))
        phi_bar = numpy.empty(self.sim.status())
        k = numpy.empty((self.sim.status(), 3))
        K = numpy.empty((self.sim.status(), 3))

        print('Reading ' + str(self.sim.status()) + ' output files... '),
        sys.stdout.flush()
        for i in numpy.arange(1, self.sim.status()):
            self.sim.readstep(i, verbose=False)

            t[i] = self.sim.time_current[0]

            self.findCrossSectionalFlux()
            Q[i,:] = self.Q

            self.findMeanPorosity()
            phi_bar[i] = self.phi_bar

            self.findPermeability()
            k[i,:] = self.k

            self.findConductivity()
            K[i,:] = self.K
        print('Done')

        fig = plt.figure()

        plt.subplot(1,4,1)
        plt.xlabel('Time $t$ [s]')
        plt.ylabel('Flux $Q$ [m^3/s]')
        plt.plot(t, Q[:,0], label='$x$')
        plt.plot(t, Q[:,1], label='$y$')
        plt.plot(t, Q[:,2], label='$z$')
        plt.legend()
        plt.grid()

        plt.subplot(1,4,2)
        plt.xlabel('Time $t$ [s]')
        plt.ylabel('Porosity $\phi$ [-]')
        plt.plot(t, phi_bar)
        plt.grid()

        plt.subplot(1,4,3)
        plt.xlabel('Time $t$ [s]')
        plt.ylabel('Permeability $k$ [m^2]')
        plt.plot(t, k[:,0], label='$x$')
        plt.plot(t, k[:,1], label='$y$')
        plt.plot(t, k[:,2], label='$z$')
        plt.legend()
        plt.grid()

        plt.subplot(1,4,4)
        plt.xlabel('Time $t$ [s]')
        plt.ylabel('Conductivity $K$ [m/s]')
        plt.plot(t, K[:,0], label='$x$')
        plt.plot(t, K[:,1], label='$y$')
        plt.plot(t, K[:,2], label='$z$')
        plt.legend()
        plt.grid()

        filename = self.sid + '-permeability.' + outformat
        plt.savefig(filename)
        print('Figure saved as "' + filename + '"')
        plt.show()
        
# Simulation ID
pc = PermeabilityCalc('permeability-dp=1000.0')
pc.printResults()
pc.plotEvolution()

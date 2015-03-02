#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'})
matplotlib.rcParams.update({'font.size': 7, 'font.family': 'sans-serif'})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import shutil

import os
import sys
import numpy
import sphere
import matplotlib.pyplot as plt
import scipy.optimize

sids =\
        ['halfshear-darcy-sigma0=80000.0-k_c=3.5e-13-mu=1.04e-07-ss=10000.0-A=70000.0-f=0.2']
        #['halfshear-darcy-sigma0=80000.0-k_c=3.5e-13-mu=1.797e-06-ss=10000.0-A=70000.0-f=0.2']
outformat = 'pdf'
fluid = True
#threshold = 100.0 # [N]


def creep_rheology(friction, n):
    ''' return strain rate from friction (tau/N) value '''
    return friction**n


for sid in sids:

    sim = sphere.sim(sid, fluid=fluid)

    #nsteps = 2
    #nsteps = 10
    nsteps = sim.status()

    t = numpy.empty(nsteps)

    tau = numpy.empty(sim.status())
    N = numpy.empty(sim.status())
    #v = numpy.empty(sim.status())
    shearstrainrate = numpy.empty(sim.status())
    shearstrain = numpy.empty(sim.status())
    for i in numpy.arange(sim.status()):
        sim.readstep(i+1, verbose=False)
        #tau = sim.shearStress()
        tau[i] = sim.w_tau_x # defined shear stress
        #tau[i] = sim.shearStress()[0] # measured shear stress along x
        N[i] = sim.currentNormalStress() # defined normal stress
        #v[i] = sim.shearVel()
        shearstrainrate[i] = sim.shearStrainRate()
        shearstrain[i] = sim.shearStrain()
        t[i] = sim.currentTime()

    # remove nonzero sliding velocities and their associated values
    #idx = numpy.nonzero(v)
    idx = numpy.nonzero(shearstrainrate)
    #v_nonzero = v[idx]
    shearstrainrate_nonzero = shearstrainrate[idx]
    tau_nonzero = tau[idx]
    N_nonzero = N[idx]
    shearstrain_nonzero = shearstrain[idx]
    t_nonzero = t[idx]

    ### "Critical" state fit
    # The algorithm uses the Levenberg-Marquardt algorithm through leastsq
    idxfit = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            (shearstrainrate_nonzero < 0.1) &
            (((t_nonzero >  6.0) & (t_nonzero <  8.0)) |
            ((t_nonzero > 11.0) & (t_nonzero < 13.0)) |
            ((t_nonzero > 16.0) & (t_nonzero < 18.0))))
    #popt, pvoc = scipy.optimize.curve_fit(
            #creep_rheology, tau/N, shearstrainrate)
    popt, pvoc = scipy.optimize.curve_fit(
            creep_rheology, tau_nonzero[idxfit]/N_nonzero[idxfit],
            shearstrainrate_nonzero[idxfit])
    print '# Critical state'
    print popt
    print pvoc
    n = popt[0] # stress exponent

    friction = tau_nonzero/N_nonzero
    x_min = numpy.floor(numpy.min(friction))
    x_max = numpy.ceil(numpy.max(friction)) + 0.05
    friction_fit =\
            numpy.linspace(numpy.min(tau_nonzero[idxfit]/N_nonzero[idxfit]),
                    numpy.max(tau_nonzero[idxfit]/N_nonzero[idxfit]),
                    100)
    strainrate_fit = friction_fit**n

    ### Consolidated state fit
    idxfit2 = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            (shearstrainrate_nonzero < 0.1) &
            ((t_nonzero > 0.0) & (t_nonzero < 3.0)))
    #popt, pvoc = scipy.optimize.curve_fit(
            #creep_rheology, tau/N, shearstrainrate)
    popt2, pvoc2 = scipy.optimize.curve_fit(
            creep_rheology, tau_nonzero[idxfit2]/N_nonzero[idxfit2],
            shearstrainrate_nonzero[idxfit2])
    print '# Consolidatet state'
    print popt2
    print pvoc2
    n2 = popt2[0] # stress exponent

    friction_fit2 =\
            numpy.linspace(numpy.min(tau_nonzero[idxfit2]/N_nonzero[idxfit2]),
                    numpy.max(tau_nonzero[idxfit2]/N_nonzero[idxfit2]),
                    100)
    strainrate_fit2 = friction_fit2**n2


    ### Plotting
    fig = plt.figure(figsize=(3.5,2.5))
    ax1 = plt.subplot(111)
    #ax1.semilogy(N/1000., v)
    #ax1.semilogy(tau_nonzero/N_nonzero, v_nonzero, '+k')
    #ax1.plot(tau/N, v, '.')
    #CS = ax1.scatter(friction, v_nonzero, c=shearstrain_nonzero,
            #linewidth=0)
    CS = ax1.scatter(friction, shearstrainrate_nonzero,
            #c=t_nonzero, linewidth=0.2,
            c=shearstrain_nonzero, linewidth=0.05,
            cmap=matplotlib.cm.get_cmap('afmhot'))
    #CS = ax1.scatter(friction[idxfit], shearstrainrate_nonzero[idxfit],
            #c=shearstrain_nonzero[idxfit], linewidth=0.2,
            #cmap=matplotlib.cm.get_cmap('afmhot'))

    ## plastic limit
    x = [0.3, 0.3]
    y = ax1.get_ylim()
    limitcolor = '#333333'
    ax1.plot(x, y, '--', linewidth=2, color=limitcolor)
    ax1.text(x[0]+0.03, 2.0e-4,
            'Yield strength', rotation=90, color=limitcolor,
            bbox={'fc':'#ffffff', 'pad':3, 'alpha':0.7})


    ## Fit
    ax1.plot(friction_fit, strainrate_fit)
    #ax1.plot(friction_fit2, strainrate_fit2)
    ax1.annotate('$\\dot{\\gamma} = (\\tau/N)^{6.40}$',
            xy = (friction_fit[40], strainrate_fit[40]),
            xytext = (0.32+0.05, 2.0e-9),
            arrowprops=dict(facecolor='blue', edgecolor='blue', shrink=0.1,
                width=1, headwidth=4, frac=0.2),)
            #xytext = (friction_fit[50]+0.15, strainrate_fit[50]-1.0e-5))#,
            #arrowprops=dict(facecolor='black', shrink=0.05),)

    ax1.set_yscale('log')
    ax1.set_xlim([x_min, x_max])
    #y_min = numpy.min(v_nonzero)*0.5
    #y_max = numpy.max(v_nonzero)*2.0
    y_min = numpy.min(shearstrainrate_nonzero)*0.5
    y_max = numpy.max(shearstrainrate_nonzero)*2.0
    ax1.set_ylim([y_min, y_max])
    #ax1.set_xlim([friction_fit[0], friction_fit[-1]])
    #ax1.set_ylim([strainrate_fit[0], strainrate_fit[-1]])

    cb = plt.colorbar(CS)
    cb.set_label('Shear strain $\\gamma$ [-]')

    #ax1.set_xlabel('Effective normal stress [kPa]')
    ax1.set_xlabel('Friction $\\tau/N$ [-]')
    #ax1.set_ylabel('Shear velocity [m/s]')
    ax1.set_ylabel('Shear strain rate $\\dot{\\gamma}$ [s$^{-1}$]')

    plt.tight_layout()
    filename = sid + '-rate-dependence.' + outformat
    plt.savefig(filename)
    plt.close()
    shutil.copyfile(filename, '/home/adc/articles/own/3/graphics/' + filename)
    print(filename)

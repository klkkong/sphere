#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import shutil

import os
import sys
import numpy
import sphere
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors
import matplotlib.ticker
import matplotlib.cm

matplotlib.rcParams.update({'font.size': 7, 'font.family': 'sans-serif'})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

#import seaborn as sns
#sns.set(style='ticks', palette='Set2')
#sns.set(style='ticks', palette='colorblind')
#sns.set(style='ticks', palette='muted')
#sns.set(style='ticks', palette='pastel')
#sns.set(style='ticks', palette='pastel')
#sns.set(style='ticks')
#sns.despine() # remove right and top spines

outformat='pdf'


sids = ['halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=2000.0-A=4000.0-f=0.2']
timescalings=[1.157e-4]

steps = numpy.arange(1625, 1875)
plotsteps = numpy.array([1670,1750,1850]) # slow creep, fast creep, slip
#plotsteps = numpy.array([1670])
contactfigs = []
contactidx = []

datalists = [[], [], []]
strikelists = [[], [], []]
diplists = [[], [], []]
forcemagnitudes = [[], [], []]
alphas = [[], [], []]
f_n_maxs = [[], [], []]
taus = [[], [], []]
ts = [[], [], []]
Ns = [[], [], []]

#f_min = 1.0
#f_max = 1.0e16
lower_limit = 0.3
upper_limit = 0.5
f_n_max = 50 # for force chain plots

N = numpy.zeros_like(steps, dtype=numpy.float64)
t = numpy.zeros_like(steps, dtype=numpy.float64)


s=0
for sid in sids:

    sim = sphere.sim(sid, fluid=True)
    t_DEM_to_t_real = timescalings[s]

    i=0
    i_scatter = 0
    for step in steps:

        sim.readstep(step, verbose=False)
        if i == 0:
            L = sim.L

        N[i] = sim.currentNormalStress('defined')
        t[i] = sim.currentTime()
        #sim.plotContacts(outfolder='../img_out/')

        if (step == plotsteps).any():
            #contactdata.append(sim.plotContacts(return_data=True))
            datalists[i_scatter], strikelists[i_scatter], diplists[i_scatter],\
                forcemagnitudes[i_scatter], alphas[i_scatter], \
                f_n_maxs[i_scatter] = sim.plotContacts(return_data=True,
                                                       lower_limit=lower_limit,
                                                       upper_limit=upper_limit)
                                                                  #f_min=f_min,
                                                                  #f_max=f_max)
            #contactfigs.append(
                #sim.plotContacts(return_fig=True,
                                 #f_min=f_min,
                                 #f_max=f_max))
            #datalists.append(data)
            #strikelists.append(strikelist)
            #diplists.append(diplist)
            #forcemagnitudes.append(forcemagnitude)
            #alphas.append(alpha)
            #f_n_maxs.append(f_n_max)
            ts[i_scatter] = t[i]
            Ns[i_scatter] = N[i]
            #taus.append(sim.shearStress('defined'))
            taus[i_scatter] = sim.shearStress('defined')

            #contactidx.append(step)
            i_scatter += 1


        i += 1
    s += 1



    ## PLOTTING ######################################################

    # Time in days
    scalingfactor = 1./t_DEM_to_t_real / (24.*3600.)
    t_scaled = t*scalingfactor


    ## Normal stress plot
    fig = plt.figure(figsize=[3.5, 3.5])

    ax1 = plt.subplot(1, 1, 1)

    ax1.plot(t_scaled, N/1000., '-k', label='$N$', clip_on=False)
    ax1.plot([0,10],[taus[0]/.3/1000., taus[0]/.3/1000.], '--', color='gray')
    ax1.set_xlabel('Time [d]')
    ax1.set_ylabel('Effective normal stress $N$ [kPa]')

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    ax1.set_xlim([numpy.min(t_scaled), numpy.max(t_scaled)])
    #ax1.locator_params(axis='x', nbins=5)
    ax1.locator_params(axis='y', nbins=4)

    ## Contact scatter plots
    dx=.23; dy=.23

    # Scatter plot 1
    sc=0
    Lx=.17; Ly=.3;
    #xytext=(Lx+.5*dx, Ly+.5*dy)
    xytext=(Lx+.15*dx, Ly+dy+.07)
    xy=(ts[0]*scalingfactor, Ns[0]/1000.)
    #print xytext
    #print xy
    ax1.annotate('',
                 xytext=xytext, textcoords='axes fraction',
                 xy=xy, xycoords='data',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))

    axsc1 = fig.add_axes([Lx, Ly, dx, dy], polar=True)
    cs = axsc1.scatter(strikelists[sc], 90. - diplists[sc], marker='o',
                       c=forcemagnitudes[sc],
                       s=forcemagnitudes[sc]/f_n_maxs[sc]*5.,
                       alpha=alphas[sc],
                       edgecolors='none',
                       vmin=f_n_maxs[sc]*lower_limit,
                       vmax=f_n_maxs[sc]*upper_limit,
                       cmap=matplotlib.cm.get_cmap('afmhot_r'))#,
                       #norm=matplotlib.colors.LogNorm())
    # tick locations
    #thetaticks = numpy.arange(0,360,90)
    # set ticklabels location at 1.3 times the axes' radius
    #ax.set_thetagrids(thetaticks, frac=1.3)
    axsc1.set_xticklabels([])
    axsc1.set_yticklabels([])

    axsc1.set_title('\\textbf{Slow creep}', fontsize=7)
    if upper_limit < 1.0:
        cbar = plt.colorbar(cs, extend='max', fraction=0.035, pad=0.04)
    else:
        cbar = plt.colorbar(cs, fraction=0.045, pad=0.04)
    #cbar.set_label('$||\\boldsymbol{f}_n||$')
    cbar.set_label('$\\boldsymbol{f}_\\text{n}$ [N]')
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=4)
    cbar.update_ticks()

    # plot defined max compressive stress from tau/N ratio
    axsc1.scatter(0., numpy.degrees(numpy.arctan(taus[sc]/Ns[sc])),
            marker='+', c='none', edgecolor='red', s=100)
    axsc1.scatter(0., numpy.degrees(numpy.arctan(taus[sc]/Ns[sc])),
            marker='o', c='none', edgecolor='red', s=100)
    '''
    ax.sc1scatter(0., # actual stress
            numpy.degrees(numpy.arctan(
                self.shearStress('effective')/
                self.currentNormalStress('effective'))),
            marker='+', color='blue', s=300)
    '''

    axsc1.set_rmax(90)
    axsc1.set_rticks([])
    #axsc1.grid(False)

    # force chain plot

    axfc1 = fig.add_axes([Lx-0.007, Ly-0.7*dy, dx, dy*0.7])

    data = datalists[sc]

    # find the max. value of the normal force
    #f_n_max = numpy.amax(data[:,6])

    # specify the lower limit of force chains to do statistics on
    f_n_lim = lower_limit * f_n_max

    # find the indexes of these contacts
    I = numpy.nonzero(data[:,6] >= f_n_lim)

    #color = matplotlib.cm.spectral(data[:,6]/f_n_max)
    for i in I[0]:

        x1 = data[i,0]
        #y1 = data[i,1]
        z1 = data[i,2]
        x2 = data[i,3]
        #y2 = data[i,4]
        z2 = data[i,5]
        f_n = data[i,6]

        lw_max = 1.0
        if f_n >= f_n_max:
            lw = lw_max
        else:
            lw = (f_n - f_n_lim)/(f_n_max - f_n_lim)*lw_max

        #print lw
        axfc1.plot([x1,x2], [z1,z2], '-k', linewidth=lw)
        #axfc1.plot([x1,x2], [z1,z2], '-', linewidth=lw, color=forcemagnitudes[sc])
        #axfc1.plot([x1,x2], [z1,z2], '-', linewidth=lw, color=color)

    axfc1.spines['right'].set_visible(False)
    axfc1.spines['left'].set_visible(False)
    # Only show ticks on the left and bottom spines
    axfc1.xaxis.set_ticks_position('none')
    axfc1.yaxis.set_ticks_position('none')
    axfc1.set_xticklabels([])
    axfc1.set_yticklabels([])
    axfc1.set_xlim([numpy.min(data[I[0],0]), numpy.max(data[I[0],0])])
    axfc1.set_ylim([numpy.min(data[I[0],2]), numpy.max(data[I[0],2])])
    axfc1.set_aspect('equal')



    # Scatter plot 2
    sc=1
    Lx=.37; Ly=.7;
    #xytext=(Lx+.5*dx, Ly+.5*dy)
    xytext=(Lx+.5*dx, Ly+.05)
    xy=(ts[sc]*scalingfactor, Ns[sc]/1000.)
    #print xytext
    #print xy
    ax1.annotate('',
                 xytext=xytext, textcoords='axes fraction',
                 xy=xy, xycoords='data',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))

    axsc2 = fig.add_axes([Lx, Ly, dx, dy], polar=True)

    cs = axsc2.scatter(strikelists[sc], 90. - diplists[sc], marker='o',
                       c=forcemagnitudes[sc],
                       s=forcemagnitudes[sc]/f_n_max*5.,
                       alpha=alphas[sc],
                       edgecolors='none',
                       vmin=f_n_maxs[sc]*lower_limit,
                       vmax=f_n_maxs[sc]*upper_limit,
                       cmap=matplotlib.cm.get_cmap('afmhot_r'))#,
                       #norm=matplotlib.colors.LogNorm())
    # tick locations
    #thetaticks = numpy.arange(0,360,90)
    # set ticklabels location at 1.3 times the axes' radius
    #ax.set_thetagrids(thetaticks, frac=1.3)
    axsc2.set_xticklabels([])
    axsc2.set_yticklabels([])

    axsc2.set_title('\\textbf{Fast creep}', fontsize=7)
    if upper_limit < 1.0:
        cbar = plt.colorbar(cs, extend='max', fraction=0.035, pad=0.04)
    else:
        cbar = plt.colorbar(cs, fraction=0.045, pad=0.04)
    cbar.set_label('$\\boldsymbol{f}_\\text{n}$ [N]')
    #cbar.set_label('Contact force [N]')
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=4)
    cbar.update_ticks()

    # plot defined max compressive stress from tau/N ratio
    axsc2.scatter(0., numpy.degrees(numpy.arctan(taus[sc]/Ns[sc])),
            marker='+', c='none', edgecolor='red', s=100)
    axsc2.scatter(0., numpy.degrees(numpy.arctan(taus[sc]/Ns[sc])),
            marker='o', c='none', edgecolor='red', s=100)
    '''
    ax.sc2scatter(0., # actual stress
            numpy.degrees(numpy.arctan(
                self.shearStress('effective')/
                self.currentNormalStress('effective'))),
            marker='+', color='blue', s=300)
    '''

    axsc2.set_rmax(90)
    axsc2.set_rticks([])
    #axsc2.grid(False)

    # force chain plot

    #axfc2 = fig.add_axes([Lx+dx+0.05, Ly+0.03, dx, dy*0.7])
    axfc2 = fig.add_axes([Lx+dx+0.10, Ly+0.03, dx, dy*0.7])

    data = datalists[sc]

    # find the max. value of the normal force
    #f_n_max = numpy.amax(data[:,6])

    # specify the lower limit of force chains to do statistics on
    f_n_lim = lower_limit * f_n_max

    # find the indexes of these contacts
    I = numpy.nonzero(data[:,6] >= f_n_lim)

    #color = matplotlib.cm.spectral(data[:,6]/f_n_max)
    for i in I[0]:

        x1 = data[i,0]
        #y1 = data[i,1]
        z1 = data[i,2]
        x2 = data[i,3]
        #y2 = data[i,4]
        z2 = data[i,5]
        f_n = data[i,6]

        lw_max = 1.0
        if f_n >= f_n_max:
            lw = lw_max
        else:
            lw = (f_n - f_n_lim)/(f_n_max - f_n_lim)*lw_max

        #print lw
        axfc2.plot([x1,x2], [z1,z2], '-k', linewidth=lw)
        #axfc2.plot([x1,x2], [z1,z2], '-', linewidth=lw, color=forcemagnitudes[sc])
        #axfc2.plot([x1,x2], [z1,z2], '-', linewidth=lw, color=color)

    axfc2.spines['right'].set_visible(False)
    axfc2.spines['left'].set_visible(False)
    # Only show ticks on the left and bottom spines
    axfc2.xaxis.set_ticks_position('none')
    axfc2.yaxis.set_ticks_position('none')
    axfc2.set_xticklabels([])
    axfc2.set_yticklabels([])
    axfc2.set_xlim([numpy.min(data[I[0],0]), numpy.max(data[I[0],0])])
    axfc2.set_ylim([numpy.min(data[I[0],2]), numpy.max(data[I[0],2])])
    axfc2.set_aspect('equal')




    # Scatter plot 3
    sc=2
    #Lx=.50; Ly=.40;
    Lx=.65; Ly=.40;
    #xytext=(Lx+.5*dx, Ly+.5*dy)
    #xytext=(Lx+1.75*dx, Ly+0.01)
    xytext=(Lx+1.15*dx, 0.2)
    xy=(ts[sc]*scalingfactor, Ns[sc]/1000.)
    #print xytext
    #print xy
    ax1.annotate('',
                 xytext=xytext, textcoords='axes fraction',
                 xy=xy, xycoords='data',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))

    axsc2 = fig.add_axes([Lx, Ly, dx, dy], polar=True)

    cs = axsc2.scatter(strikelists[sc], 90. - diplists[sc], marker='o',
                       c=forcemagnitudes[sc],
                       s=forcemagnitudes[sc]/f_n_max*5.,
                       alpha=alphas[sc],
                       edgecolors='none',
                       vmin=f_n_maxs[sc]*lower_limit,
                       vmax=f_n_maxs[sc]*upper_limit,
                       cmap=matplotlib.cm.get_cmap('afmhot_r'))#,
                       #norm=matplotlib.colors.LogNorm())
    # tick locations
    #thetaticks = numpy.arange(0,360,90)
    # set ticklabels location at 1.3 times the axes' radius
    #ax.set_thetagrids(thetaticks, frac=1.3)
    axsc2.set_xticklabels([])
    axsc2.set_yticklabels([])

    axsc2.set_title('\\textbf{Slip}', fontsize=7)
    if upper_limit < 1.0:
        cbar = plt.colorbar(cs, extend='max', fraction=0.035, pad=0.04)
    else:
        cbar = plt.colorbar(cs, fraction=0.045, pad=0.04)
    cbar.set_label('$\\boldsymbol{f}_\\text{n}$ [N]')
    #cbar.set_label('Contact force [N]')
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=4)
    cbar.update_ticks()

    # plot defined max compressive stress from tau/N ratio
    axsc2.scatter(0., numpy.degrees(numpy.arctan(taus[sc]/Ns[sc])),
            marker='+', c='none', edgecolor='red', s=100)
    axsc2.scatter(0., numpy.degrees(numpy.arctan(taus[sc]/Ns[sc])),
            marker='o', c='none', edgecolor='red', s=100)
    '''
    ax.sc2scatter(0., # actual stress
            numpy.degrees(numpy.arctan(
                self.shearStress('effective')/
                self.currentNormalStress('effective'))),
            marker='+', color='blue', s=300)
    '''

    axsc2.set_rmax(90)
    axsc2.set_rticks([])
    #axsc2.grid(False)

    # force chain plot

    axfc2 = fig.add_axes([Lx-0.007, Ly-0.7*dy, dx, dy*0.7])
    #axfc2 = fig.add_axes([Lx+dx+0.05, Ly+0.03, dx, dy*0.7])

    data = datalists[sc]

    # find the max. value of the normal force
    #f_n_max = numpy.amax(data[:,6])

    # specify the lower limit of force chains to do statistics on
    f_n_lim = lower_limit * f_n_max

    # find the indexes of these contacts
    I = numpy.nonzero(data[:,6] >= f_n_lim)

    #color = matplotlib.cm.spectral(data[:,6]/f_n_max)
    for i in I[0]:

        x1 = data[i,0]
        #y1 = data[i,1]
        z1 = data[i,2]
        x2 = data[i,3]
        #y2 = data[i,4]
        z2 = data[i,5]
        f_n = data[i,6]

        lw_max = 1.0
        if f_n >= f_n_max:
            lw = lw_max
        else:
            lw = (f_n - f_n_lim)/(f_n_max - f_n_lim)*lw_max

        #print lw
        axfc2.plot([x1,x2], [z1,z2], '-k', linewidth=lw)
        #axfc2.plot([x1,x2], [z1,z2], '-', linewidth=lw, color=forcemagnitudes[sc])
        #axfc2.plot([x1,x2], [z1,z2], '-', linewidth=lw, color=color)

    axfc2.spines['right'].set_visible(False)
    axfc2.spines['left'].set_visible(False)
    # Only show ticks on the left and bottom spines
    axfc2.xaxis.set_ticks_position('none')
    axfc2.yaxis.set_ticks_position('none')
    axfc2.set_xticklabels([])
    axfc2.set_yticklabels([])
    axfc2.set_xlim([numpy.min(data[:,0]), numpy.max(data[:,0])])
    axfc2.set_ylim([numpy.min(data[:,2]), numpy.max(data[:,2])])
    axfc2.set_aspect('equal')







    #fig.tight_layout()
    #plt.subplots_adjust(hspace=0.05)
    plt.subplots_adjust(right=0.82)

    filename = sid + '-creep-dynamics.' + outformat
    plt.savefig(filename)
    plt.close()
    shutil.copyfile(filename, '/home/adc/articles/own/3/graphics/' + filename)
    print(filename)

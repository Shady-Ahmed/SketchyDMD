"""
Plotting scripts for the Shallow Water Equations on Spherical Coordinates problem

This code was used to generate the results accompanying the following paper:
    "Dynamic mode decomposition with core sketch"
    Authors: Shady E Ahmed, Pedram H Dabaghian, Omer San, Diana Bistrian, and Ionel Navon
    Published in the Physics of Fluids journal
For any questions and/or comments, please email me at: shady.ahmed@okstate.edu
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi#, sin, cos, tan, exp

import cartopy.crs as ccrs
import matplotlib
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

############################### Plotting ######################################
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}
mpl.rc('font', **font)

import sys

import cmasher as cmr

#%%   
    
# Inputs

output_interval_mins = 15    #Time between outputs (minutes)
forecast_length_days = 6     #Total simulation length (days)
output_interval = output_interval_mins*60.0 #Time between outputs (s)
forecast_length = forecast_length_days*24.0*3600.0 #Forecast length (s)

noutput = int(np.round(forecast_length/output_interval)) #Number of output frames

nstart_day = 3 #truncate the first 3 days
nstart = nstart_day*24.0*3600.0

nout_start = int(np.round(nstart/output_interval))

dtrom = output_interval
ns    = noutput-nout_start 
   
#%% read data

filename  = "./Data/FOM.npz"
data      = np.load(filename)
x         = data['Phi'] * 180/pi 
y         = data['Theta'] * 180/pi
vorticity = data['vorticity']
time      = data['t_save']

wfom      = vorticity[:,:,nout_start:noutput+1] #take data from end of day 3 to end of day 6
[nx,ny]   = x.shape

#%% FOM PLOT

fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                      subplot_kw={'projection': ccrs.Orthographic(central_longitude=-90.0, central_latitude=10.0) })

ax = ax.flat
vv = np.linspace(-1e-4,1e-4,100, endpoint=True)
ctick = np.linspace(-1e-4, 1e-4, 5, endpoint=True)

mapp = 'RdGy'
for i in range(3):
    kk = int(96)*(i+1)

    cs = ax[i].contourf(x, y, wfom[:,:,kk],vv,  transform=ccrs.PlateCarree(),
            cmap=mapp, extend='both')

    for c in cs.collections:
        c.set_rasterized(True)

    cs.set_clim([-1e-4,1e-4])

    ax[i].coastlines()
    gl = ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=1, color='k', alpha=0.5, linestyle='-')
    
    ax[i].set_title(r"\bf{Day " + str(int(time[nout_start+kk]/3600/24)) + "}", fontsize = 12)
    ax[i].set_global()

fig.subplots_adjust(hspace=0.15, wspace=0.2)

# plt.savefig('./Figures/Fig3.png', dpi = 300, bbox_inches = 'tight')
# plt.savefig('./Figures/Fig3.eps', dpi = 300, bbox_inches = 'tight')
# plt.savefig('./Figures/Fig3.pdf', dpi = 300, bbox_inches = 'tight')

plt.show()

       
#%% Deterministic DMD with different number of modes

#% DMD plot

svd = None 
sort = None 
        

nrs = [20,40,60]
fig,ax = plt.subplots(nrows=3, ncols=3, figsize=(8, 9),
                          subplot_kw={'projection': ccrs.Orthographic(central_longitude=-90.0, central_latitude=10.0) })
ax = ax.flat
vv = np.linspace(-1e-4,1e-4,100, endpoint=True)
ctick = np.linspace(-1e-4, 1e-4, 5, endpoint=True)
    
for jj, nr in enumerate(nrs):

    filename = "./Results/DMD_nr="+str(nr) + "_svd=" + str(svd) + "_isort=" + str(sort) + ".npz"
    data = np.load(filename)
    wdmd = data['wdmd']
                
    for i in range(3):
        kk = int(96)*(i+1) 
    
        cs = ax[i+jj*3].contourf(x, y, wdmd[:,:,kk],vv,  transform=ccrs.PlateCarree(),
                cmap=mapp, extend='both')
        
        for c in cs.collections:
            c.set_rasterized(True)
    
        cs.set_clim([-1e-4,1e-4])
    
        ax[i+jj*3].coastlines()
        gl = ax[i+jj*3].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=1, color='k', alpha=0.5, linestyle='-')
        
        if jj == 0:
            ax[i].set_title(r"\bf{Day " + str(int(time[nout_start+kk]/3600/24)) + "}", fontsize = 18, pad=15)
    
        ax[i+jj*3].set_global()
    
    ax[jj*3].text(-0.7, 0.5, r"$r = " + str(nr) +"$", va='center',fontsize=18, transform=ax[jj*3].transAxes)
    
    
        
rect = plt.Rectangle(
    # (lower-left corner), width, height
    (0.1, 0.65), 0.83, 0.24, fill=False, color="k", lw=2, 
    zorder=1000, transform=fig.transFigure, figure=fig, ls='--')
fig.patches.extend([rect])

rect = plt.Rectangle(
    # (lower-left corner), width, height
    (0.1, 0.382), 0.83, 0.24, fill=False, color="k", lw=2, 
    zorder=1000, transform=fig.transFigure, figure=fig, ls='--')
fig.patches.extend([rect])

rect = plt.Rectangle(
    # (lower-left corner), width, height
    (0.1, 0.111), 0.83, 0.24, fill=False, color="k", lw=2, 
    zorder=1000, transform=fig.transFigure, figure=fig, ls='--')
fig.patches.extend([rect])


fig.subplots_adjust(hspace=0.25, wspace=0.15)


# plt.savefig('./Figures/Fig4.png', dpi = 300, bbox_inches = 'tight')
# plt.savefig('./Figures/Fig4.eps', dpi = 300, bbox_inches = 'tight')
# plt.savefig('./Figures/Fig4.pdf', dpi = 300, bbox_inches = 'tight')

plt.show()              
    

       
#%% Deterministic DMD with different sorting criteria

#% DMD plot

svd = None ## None=standard svd, bistrian, erichson, sketching1, sketching2
        

nr = 20

sorts = [None,2] ## None=early truncation, 0=amplitudes, 1, 2

fig,ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 6),
                          subplot_kw={'projection': ccrs.Orthographic(central_longitude=-90.0, central_latitude=10.0) })
ax = ax.flat
vv = np.linspace(-1e-4,1e-4,100, endpoint=True)
ctick = np.linspace(-1e-4, 1e-4, 5, endpoint=True)
    
for jj, sort in enumerate(sorts):

    filename = "./Results/DMD_nr="+str(nr) + "_svd=" + str(svd) + "_isort=" + str(sort) + ".npz"
    data = np.load(filename)
    wdmd = data['wdmd']
                
    for i in range(3):
        kk = int(96)*(i+1) 
    
        cs = ax[i+jj*3].contourf(x, y, wdmd[:,:,kk],vv,  transform=ccrs.PlateCarree(),
                cmap=mapp, extend='both')
        
        for c in cs.collections:
            c.set_rasterized(True)
    
        cs.set_clim([-1e-4,1e-4])
    
        ax[i+jj*3].coastlines()
        gl = ax[i+jj*3].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=1, color='k', alpha=0.5, linestyle='-')
        
        if jj == 0:
            ax[i].set_title(r"\bf{Day " + str(int(time[nout_start+kk]/3600/24)) + "}", fontsize = 14, pad=15)
    
        ax[i+jj*3].set_global()
    
ax[0].text(-0.65, 0.5, r"\bf{Early}", va='center',fontsize=14, transform=ax[0].transAxes)
ax[0].text(-0.8, 0.4, r"\bf{Truncation}", va='center',fontsize=14, transform=ax[0].transAxes)

ax[3].text(-0.7, 0.5, r"\bf{Sorting}", va='center',fontsize=14, transform=ax[3].transAxes)
ax[3].text(-0.74, 0.4, r"\bf{Criterion}", va='center',fontsize=14, transform=ax[3].transAxes)

            
rect = plt.Rectangle(
    # (lower-left corner), width, height
    (0.1, 0.53), 0.83, 0.355, fill=False, color="k", lw=2, 
    zorder=1000, transform=fig.transFigure, figure=fig, ls='--')
fig.patches.extend([rect])

rect = plt.Rectangle(
    # (lower-left corner), width, height
    (0.1, 0.115), 0.83, 0.355, fill=False, color="k", lw=2, 
    zorder=1000, transform=fig.transFigure, figure=fig, ls='--')
fig.patches.extend([rect])


fig.subplots_adjust(hspace=0.25, wspace=0.15)

# plt.savefig('./Figures/Fig5.png', dpi = 300, bbox_inches = 'tight')
# plt.savefig('./Figures/Fig5.eps', dpi = 300, bbox_inches = 'tight')
# plt.savefig('./Figures/Fig5.pdf', dpi = 300, bbox_inches = 'tight')

plt.show()
    
#%% Sketching DMD

#% DMD plot

sort = 2 ## None=early truncation, 0=amplitudes, 1, 2  
nr = 20

svds = [None,'CoreX1']  

fig,ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 6),
                          subplot_kw={'projection': ccrs.Orthographic(central_longitude=-90.0, central_latitude=10.0) })
ax = ax.flat
vv = np.linspace(-1e-4,1e-4,100, endpoint=True)
ctick = np.linspace(-1e-4, 1e-4, 5, endpoint=True)

for jj, ssvd in enumerate(svds):

    filename = "./Results/DMD_nr="+str(nr) + "_svd=" + str(svd) + "_sort=" + str(sort) + ".npz"
    data = np.load(filename)
    wdmd = data['wdmd']
                
    for i in range(3):
        kk = int(96)*(i+1) 
    
        cs = ax[i+jj*3].contourf(x, y, wdmd[:,:,kk],vv,  transform=ccrs.PlateCarree(),
                cmap=mapp, extend='both')
        
        for c in cs.collections:
            c.set_rasterized(True)
    
        cs.set_clim([-1e-4,1e-4])
    
        ax[i+jj*3].coastlines()
        gl = ax[i+jj*3].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=1, color='k', alpha=0.5, linestyle='-')
        
        if jj == 0:
            ax[i].set_title(r"\bf{Day " + str(int(time[nout_start+kk]/3600/24)) + "}", fontsize = 14, pad=15)
        ax[i+jj*3].set_global()
    
ax[0].text(-0.85, 0.5, r"\bf{Determinstic}", va='center',fontsize=14, transform=ax[0].transAxes)
ax[0].text(-0.65, 0.4, r"\bf{DMD}", va='center',fontsize=14, transform=ax[0].transAxes)
ax[3].text(-0.85, 0.5, r"\bf{SketchyDMD}", va='center',fontsize=14, transform=ax[3].transAxes)
    
            
rect = plt.Rectangle(
    # (lower-left corner), width, height
    (0.1, 0.53), 0.83, 0.355, fill=False, color="k", lw=2, 
    zorder=1000, transform=fig.transFigure, figure=fig, ls='--')
fig.patches.extend([rect])

rect = plt.Rectangle(
    # (lower-left corner), width, height
    (0.1, 0.115), 0.83, 0.355, fill=False, color="k", lw=2, 
    zorder=1000, transform=fig.transFigure, figure=fig, ls='--')
fig.patches.extend([rect])

fig.subplots_adjust(hspace=0.25, wspace=0.15)


# plt.savefig('./Figures/Fig6.png', dpi = 300, bbox_inches = 'tight')
# plt.savefig('./Figures/Fig6.eps', dpi = 300, bbox_inches = 'tight')
# plt.savefig('./Figures/Fig6.pdf', dpi = 300, bbox_inches = 'tight')

plt.show()              


#%%

nr = 100
sort = 2 ## None=early truncation, 0=amplitudes, 1, 2
    
symbols = ['o','x','^','s']
i = 0
   
lamda_svd = []
for svd in [None, 'RangeX1', 'RangeX', 'CoreX1']:    
    filename = "./Results/DMD_nr="+str(nr) + "_svd=" + str(svd) + "_sort=" + str(sort) + ".npz"
    data = np.load(filename)
    lamda = data['lam']
    lamda_svd.append(lamda)     
    
    
xx = np.cos(np.linspace(0,2*np.pi,100))
yy = np.sin(np.linspace(0,2*np.pi,100))

fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(14,4))      
ax = ax.flat
    
axin1 = []
axin2 = []

for i in range(3):    
    lamda = lamda_svd[0]
    ax[i].plot(np.real(lamda),np.imag(lamda),'x',fillstyle='none',markeredgecolor='C0')
    ax[i].set_xlabel(r'$real(\lambda)$')
    ax[i].set_ylabel(r'$imag(\lambda)$')
    ax[i].plot(xx,yy,'k--',linewidth=2)

    # inset axes....
    axins = ax[i].inset_axes([0.25, 0.5, 0.4, 0.3])
    axins.plot(np.real(lamda),np.imag(lamda),'x',fillstyle='none',markeredgecolor='C0')
    #axins.set_facecolor((1.0, 0.47, 0.42))

    # # sub region of the original image
    x1, x2, y1, y2 = -0.7, 0, 0.75, 1.05
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.set_xticks([])
    axins.set_yticks([])
    ax[i].indicate_inset_zoom(axins, edgecolor="black",facecolor='black',alpha=0.1)
    axins.patch.set_facecolor('black')
    axins.patch.set_alpha(0.1)


    axin1.append(axins)

    # inset axes....
    axins = ax[i].inset_axes([0.45, 0.08, 0.35, 0.3])
    axins.plot(np.real(lamda),np.imag(lamda),'x',fillstyle='none',markeredgecolor='C0')

    # # sub region of the original image
    x1, x2, y1, y2 = 0.98, 1.01, -0.3, 0.3
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.set_xticks([])
    axins.set_yticks([])
    ax[i].indicate_inset_zoom(axins, edgecolor="black",facecolor='black',alpha=0.1)
    axins.patch.set_facecolor('black')
    axins.patch.set_alpha(0.1)    
    axin2.append(axins)
    

lamda = lamda_svd[3]
ax[0].plot(np.real(lamda),np.imag(lamda),'o',fillstyle='none',markeredgecolor='C1')
ax[0].set_title(r'\textbf{DMD with $X_1$ core sketch}')
# inset axes....
axins = axin1[0]
axins.plot(np.real(lamda),np.imag(lamda),'o',fillstyle='none')
ax[0].indicate_inset_zoom(axins, edgecolor="black")
axins = axin2[0]
axins.plot(np.real(lamda),np.imag(lamda),'o',fillstyle='none')
ax[0].indicate_inset_zoom(axins, edgecolor="black")


lamda = lamda_svd[1]
ax[1].plot(np.real(lamda),np.imag(lamda),'o',fillstyle='none')
ax[1].set_title(r'\textbf{DMD with $X_1$ range sketch}')
# inset axes....
axins = axin1[1]
axins.plot(np.real(lamda),np.imag(lamda),'o',fillstyle='none')
ax[1].indicate_inset_zoom(axins, edgecolor="black")
axins = axin2[1]
axins.plot(np.real(lamda),np.imag(lamda),'o',fillstyle='none')
ax[1].indicate_inset_zoom(axins, edgecolor="black")


lamda = lamda_svd[2]
ax[2].plot(np.real(lamda),np.imag(lamda),'o',fillstyle='none')
ax[2].set_title(r'\textbf{DMD with $X$ range sketch}')
# inset axes....
axins = axin1[2]
axins.plot(np.real(lamda),np.imag(lamda),'o',fillstyle='none')
ax[2].indicate_inset_zoom(axins, edgecolor="black")
axins = axin2[2]
axins.plot(np.real(lamda),np.imag(lamda),'o',fillstyle='none')
ax[2].indicate_inset_zoom(axins, edgecolor="black")


fig.subplots_adjust(wspace=0.3)
# plt.savefig('./Figures/Fig7.png', dpi = 300, bbox_inches = 'tight')
# plt.savefig('./Figures/Fig7.eps', dpi = 300, bbox_inches = 'tight')
# plt.savefig('./Figures/Fig7.pdf', dpi = 300, bbox_inches = 'tight')

plt.show()              





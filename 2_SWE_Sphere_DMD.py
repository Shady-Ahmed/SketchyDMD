# -*- coding: utf-8 -*-
"""
DMD Implementation with Different Algorithms and Sorting Criteria
for the Shallow Water Equations on Spherical Coordinates problem

This code was used to generate the results accompanying the following paper:
    "Dynamic mode decomposition with core sketch"
    Authors: Shady E Ahmed, Pedram H Dabaghian, Omer San, Diana Bistrian, and Ionel Navon
    Published in the Physics of Fluids journal
For any questions and/or comments, please email me at: shady.ahmed@okstate.edu
"""


#%% Import Libraries
import numpy as np
from numpy.random import default_rng

import numpy.linalg as LA
import matplotlib.pyplot as plt
#from numpy import pi, sin, cos, tan, exp
import os,sys

#%% Define Functions

#==================================================================================#
# DMD Implementation
def DMD(u, nr, dt, svd=None, sort=None, eps=None, Exact=False, s=0, p=0, k=0, l=0): 
#==================================================================================#

    [nx,ny,ns] = np.shape(u)
    X  = np.reshape(u,[nx*ny,ns])
    X1 = np.copy(X[:,:-1])
    X2 = np.copy(X[:,1:])
    
    if svd == 'RangeX1':
        Q, X1 = rand_DMD_RX1(X,nr,p=0)
        
    elif svd == 'RangeX':
        Q, X1, X2 = rand_DMD_RX(X,nr,s,p=0)
        
    elif svd =='CoreX1':
        Q, P, X1 = rand_DMD_CX1(X,nr,k,l)

   
    U, s, Vh = LA.svd(X1, full_matrices=False)
    V = Vh.conj().T
    S = np.diag(s)
 
    if sort == None:
        r = nr
    else:
        r = LA.matrix_rank(S,tol=eps)
    
    if svd == 'RangeX1':
        U  = Q @ U
        X1 = np.copy(X[:,:-1])       
        
    elif svd == 'CoreX1':
        U  = Q @ U        
        V  = P @ V
        X1 = np.copy(X[:,:-1])
        r  = int(2.5*nr) #increase rank- based on empirical observation
        
    Ur = U[:,:r]
    Sr = S[:r,:r]
    Vr = V[:,:r]

    Atilde = (Ur.conj().T) @ X2 @ Vr @ LA.inv(Sr)
    lam,Wr = LA.eig(Atilde)

    if Exact:  
        Phi = X2 @ Vr @ LA.inv(Sr) @ Wr
    else:
        Phi = Ur @ Wr     
        
    omega = np.log(lam)/dt
    amp = LA.lstsq(Phi,X1[:,0],rcond=None)[0]
        
    if svd == 'RangeX':
        Phi = Q @ Phi
        
        
    if sort is not None:
        ind = DMD_sort(Phi,omega,lam,amp,nr,ns,dt,sort)
        Phi   = np.copy(Phi[:,ind])
        omega = np.copy(omega[ind])
        lam   = np.copy(lam[ind])
        amp   = np.copy(amp[ind])

    return Phi, omega, lam, amp


#=============================================#
# Sorting criteria
def DMD_sort(Phi,omega,lam,amp,nr,ns,dt,sort): 
#=============================================#
    
    if sort == 0:
        I = np.abs(amp)
        
    if sort == 1:
    #Ahmed, S. E., San, O., Bistrian, D. A., & Navon, I. M. (2020). 
    #Sampling and resolution characteristics in reduced order models of shallow water equations: Intrusive vs nonintrusive. 
    #International Journal for Numerical Methods in Fluids, 92(8), 992-1036.
        sigma = np.real(omega) #growth rate
        I = np.abs(amp) * ( np.exp(sigma) + np.exp(-sigma) )
               
    elif sort == 4:
    #Ahmed, S. E., San, O., Bistrian, D., & Navon, I. (2022).
    #Sketching Methods for Dynamic Mode Decomposition in Spherical Shallow Water Equations.
    #In AIAA SCITECH 2022 Forum (p. 2325).
        
        sigma = np.real(omega) #growth rate
        T = (ns-1)*dt
        I = np.abs(amp) * ( np.exp(sigma*T) - 1)/(sigma*T)

    ind = np.argsort(I)[::-1][:nr]
    
    return ind

#==================================#
# DMD solution/reconstruction
def DMD_solve(Phi,omega,amp,ns,dt):
#==================================#
    
    Tn = ns*dt
    t = np.linspace(0,Tn,ns+1)
    nr = np.shape(Phi)[1]
    
    time_dynamics = np.zeros([nr,ns+1],dtype=np.complex128)
    
    for i in range(ns+1):
        time_dynamics[:,i] = amp*np.exp(omega*t[i])
    
    Xdmd = Phi @ time_dynamics
    
    return np.real(Xdmd)

#==========================#
# Sketching range of X1
def rand_DMD_RX1(X,nr,p=0):
#==========================#
    #Bistrian, D. A., & Navon, I. M. (2017).
    #Randomized dynamic mode decomposition for nonintrusive reduced order modelling.
    #International Journal for Numerical Methods in Engineering, 112(1), 3-25.
    
    rng = default_rng(irand)

    # Stage A: Computing a near-optimal basis for X1 such that (X1 ~ Q Q* X1)
    [n,m] = X.shape
    X1 = np.copy(X[:,:-1])
    
    #over-sampling
    l = np.min([2*nr,m-1]) 
    G = rng.standard_normal([m-1,l])
    Y = X1 @ G #this cost can be improved by using strucutured random matrix
    
    #power-iterations
    for i in range(p):
        Q, _ = LA.qr(Y) #orthonomalization for stability
        Z, _ = LA.qr(X1.conj().T @ Q)
        Y = X1 @ Z
    Q, _ = LA.qr(Y)
    
    #Stage B: Computing a low-dimensional matrix
    B1 = Q.conj().T @ X1
    return Q, B1



#=============================#
# Sketching range of X
def rand_DMD_RX(X,nr,s=0,p=0):
#=============================#  
    #Erichson, N. B., Mathelin, L., Kutz, J. N., & Brunton, S. L. (2019).
    #Randomized dynamic mode decomposition.
    #SIAM Journal on Applied Dynamical Systems, 18(4), 1867-1891.
    
    rng = default_rng(irand)

    # Stage A: Computing a near-optimal basis for X such that (X ~ Q Q* X)
    [n,m] = X.shape
    
    #over-sampling
    l = nr+s 
    G = rng.standard_normal([m,l])
    Y = X @ G #this cost can be improved by using strucutured random matrix
    
    #power-iterations
    for i in range(p):
        Q, _ = LA.qr(Y) #orthonomalization for stability
        Z, _ = LA.qr(X.conj().T @ Q)
        Y = X @ Z
    Q, _ = LA.qr(Y)
    
    #Stage B: Computing a low-dimensional matrix
    B = Q.conj().T @ X
    
    B1 = np.copy(B[:,:-1])
    B2 = np.copy(B[:,1:])    
    return Q, B1, B2


#==========================#
# Sketching core of X
def rand_DMD_CX1(X,nr,k,l):   
#==========================#     
    rng = default_rng(irand)

    # Stage A: Computing a near-optimal basis for X1 such that (X1 ~ Q Q* X1)
    [n,m] = X.shape
    X1 = np.copy(X[:,:-1])
       
    Gamma = rng.standard_normal([k,n])
    Omega = rng.standard_normal([k,m-1])
    Phi = rng.standard_normal([l,n])
    Psi = rng.standard_normal([l,m-1])
 
    
    #multiplication cost can be improved by using strucutured random matrix
    U = Gamma @ X1              #this matrix captures the co-range of the data matrix X1
    Y = X1 @ Omega.conj().T     #this matrix captures the range of the data matrix X1
    Z = Phi @ X1 @ Psi.conj().T #the core sketch contains fresh information that improves
                                #our estimates of the singular values and singular vectors of X1
                                #it is responsible for the superior performance of the new method.
    
    P, _ = LA.qr(U.conj().T) #orthonormal matrix P spanning sketch U.T -- for co-range
    Q, _ = LA.qr(Y)          #orthonormal matrix Q spanning sketch Y -- for range

    #solve two small least-square problems to get  the core approximation C = (Phi Q\Z)/(Psi P).T,
    #which describes how X1 acts between range(P) and range(Q)
    C1 = LA.lstsq(Phi @ Q, Z,rcond=None)[0]
    Ch = LA.lstsq(Psi @ P, C1.conj().T,rcond=None)[0]
    B1 = Ch.conj().T #Computing a low-dimensional matrix

    return Q, P, B1


#%% Main Program

def main(nr, svd=None, sort=None, eps=None, Exact=False, s=0, p=0, k=0, l=0):
    
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
    ns = noutput-nout_start 
       
    #%% read data
    
    filename = "./Data/FOM.npz"
    data = np.load(filename)
    x = data['Phi']
    y = data['Theta']
    vorticity = data['vorticity']
    t_time = data['t_save']
    t_time = t_time[nout_start:noutput+1]/3600-72
    
    w = vorticity[:,:,nout_start:noutput+1] #take data from end of day 3 to end of day 6
        
    #%% compute dmd
    Phi, omega, lam, amp = DMD(w, nr, dtrom, svd, sort, eps, Exact, s, p, k, l)
        
    
    #%% solve dmd
    Xdmd = DMD_solve(Phi,omega,amp,ns,dtrom)
    [nx,ny] = x.shape
    wdmd = np.reshape(Xdmd,[nx,ny,ns+1])
    # err = np.zeros(ns+1)
    # for i in range(ns+1):
    #     err[i] = LA.norm(wdmd[:,:,i]-w[:,:,i])/LA.norm(w[:,:,i])
    
    #%% save results
    filename = "./Results/DMD_nr="+str(nr) + "_svd=" + str(svd) + "_sort=" + str(sort) + ".npz"
    np.savez(filename, Phi=Phi, omega=omega, lam=lam, amp=amp, wdmd=wdmd)
    return w, wdmd

if __name__ == "__main__":
    
    nr = 20 #number of DMD modes
        
    svd = None
    sort = None
    eps = None
    Exact = False
    s = 0 #over-sampling for range sketch
    p = 0 #power iteration for range sketch
    k = 0 #over-sampling for core sketch
    l = 0 #over-sampling for core sketch
    irand = 0 #seed for random number generator
    
    w, wdmd = main(nr,svd=svd, sort=sort, eps=eps, Exact=Exact, s=s, p=p, k=k, l=l)
        
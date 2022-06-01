# -*- coding: utf-8 -*-
"""
Forward solver for the Shallow Water Equations on Spherical Coordinates

This code was used to generate the results accompanying the following paper:
    "Dynamic mode decomposition with core sketch"
    Authors: Shady E Ahmed, Pedram H Dabaghian, Omer San, Diana Bistrian, and Ionel Navon
    Published in the Physics of Fluids journal
For any questions and/or comments, please email me at: shady.ahmed@okstate.edu

"""

#%% Import Libraries
import numpy as np
from numpy import pi, sin, cos, tan, exp
import os

#%% Define Functions

def lax_wendroff_sphere(dphi,dtheta,dt,g,u,v,h,H,R,Theta,u_accel,v_accel):
    
    #This function performs one timestep of the Lax-Wendroff scheme applied to the spherical shallow water equations
    #Richtmyer two-step Lax–Wendroff method
    [nx,ny] = np.shape(u) 

    dx=dphi*R*cos(Theta)
    dx = np.reshape(dx,[nx,ny,1])
    dy=dtheta*R
    
    q = np.zeros([nx,ny,3]) #h,hu,hv -- conserved quantity
    q[:,:,0] = h
    q[:,:,1] = h*u
    q[:,:,2] = h*v
    
    F, G = flux(q)
    
    #step 1: compute predictor at i+1/2, j+1/2, n+1/2
    #work out mid-point values in time and space
    i = np.arange(0,nx-1)
    q_mid_xt = 0.5*(q[i+1,:,:]+q[i,:,:]) - 0.5*dt/dx[i,:]*(F[i+1,:,:]-F[i,:,:])
    F_mid_xt,tmp = flux(q_mid_xt)
    
    j = np.arange(0,ny-1)
    q_mid_yt = 0.5*(q[:,j+1,:]+q[:,j,:]) - 0.5*dt/dy*(G[:,j+1,:]-G[:,j,:])
    tmp,G_mid_yt = flux(q_mid_yt)


    #Step 2: use the mid-point values to predict the values at the next timestep
    i = np.arange(1,nx-1)
    j = np.arange(1,ny-1)
    [ii,jj] = np.meshgrid(i,j, indexing='ij')
    qnew = q[ii,jj] - dt/dx[ii,jj]*(F_mid_xt[ii,jj,:]-F_mid_xt[ii-1,jj,:]) - dt/dy*(G_mid_yt[ii,jj,:]-G_mid_yt[ii,jj-1,:])
    hnew = np.copy(qnew[:,:,0])
    
    #add source term contribution
    S = np.zeros([nx-2,ny-2,3]) #Source term
    S[:,:,0] = tan(Theta[ii,jj])*v[ii,jj]/(R+H[ii,jj]) * 0.5*(h[ii,jj]+hnew)
    qnew[:,:,0] = np.copy(qnew[:,:,0]) + np.copy(S[:,:,0])*dt
    
    hnew = np.copy(qnew[:,:,0]) #update hnew
    S[:,:,1] = u_accel* 0.5*(h[ii,jj]+hnew)
    S[:,:,2] = v_accel* 0.5*(h[ii,jj]+hnew)

    qnew[:,:,1:] = np.copy(qnew[:,:,1:]) + np.copy(S[:,:,1:])*dt

    hnew = np.copy(qnew[:,:,0])
    unew = np.copy(qnew[:,:,1]/qnew[:,:,0])
    vnew = np.copy(qnew[:,:,2]/qnew[:,:,0])
    return hnew,unew,vnew


def flux(q):
    
    [nx,ny,neq] = np.shape(q) #neq=3
    if neq != 3:
         print('Error: Incorrect Dimension of q')
    #q = ([nx,ny,3]) #h,hu,hv -- conserved quantity
    h = np.copy(q[:,:,0])
    u = np.copy(q[:,:,1]/q[:,:,0])
    v = np.copy(q[:,:,2]/q[:,:,0])

    F = np.zeros([nx,ny,3]) #hu,hu^2+1/2gh^2,huv -- Flux in x
    F[:,:,0] = u*h
    F[:,:,1] = u**2*h + 0.5*g*h**2
    F[:,:,2] = u*v*h

    G = np.zeros([nx,ny,3]) #hv,huv,hv^2+1/2gh^2 -- Flux in y
    G[:,:,0] = v*h
    G[:,:,1] = u*v*h
    G[:,:,2] = v**2*h + 0.5*g*h**2
    
    return F,G



#%% Main Program

# Inputs

# user settings
add_random_height_noise = 'true'
initially_geostrophic = 'true'

dt_mins              = 0.5    #Timestep (minutes)
output_interval_mins = 15     #Time between outputs (minutes)
forecast_length_days = 10     #Total simulation length (days)
dt = dt_mins*60.0             #Timestep (s)
output_interval = output_interval_mins*60.0            #Time between outputs (s)
forecast_length = forecast_length_days*24.0*3600.0     #Forecast length (s)

nt = np.int(np.round(forecast_length/dt))              #Number of timesteps
timesteps_between_outputs = np.int(np.round(output_interval/dt))
noutput = np.int(np.round(forecast_length/output_interval)) #Number of output frames



# constants
g   = 9.8              #gravity
rho = 1.2              #density
R   = 6.4e6            #radius of planet
f   = 2*pi/(24*3600)   #radians per second - earth's rotation rate

# set-up of model grid
dtheta = 1*pi/180;     #the north-south (latitude coordinate)
dphi   = 1*pi/180;     #the east-west (longitude coordinate)

phi   = np.arange(0,2*pi,dphi)   #the longitude mesh from 0 to 2pi-dphi because 2pi is the same as 0
theta = np.arange(-80*pi/180+dtheta/2,80*pi/180+dtheta/2,dtheta)    # the latitude mesh from -80+dtheta/2 to 80-dtheta/2 (-79.5 to 79.5 -- cell centers)
                                                                    # the regions near the poles are excluded because sin 90 = 0
nx = len(phi)   #number of points in east-west direction
ny = len(theta) #number of points in north-south direction

[Phi,Theta] = np.meshgrid(phi,theta, indexing='ij') #mesh


F = 2*f*sin(Theta)     #Coriolis parameter

H = np.zeros([nx,ny])  #terrain height -- for flat: bottom H = 0

#%% Define initial conditions

height = 1e4 - 60*cos(4*pi*Theta)*exp(-1*(Theta)**2)

if add_random_height_noise:
    height = height + 1.0*np.random.randn(nx,ny)*(dtheta*(ny-1)/pi)*(np.abs(F)*1e4)*cos(Theta)
    
    
u=np.zeros([nx,ny])
v=np.zeros([nx,ny])
if initially_geostrophic:
    
   #Centred spatial differences to compute geostrophic wind
   j = np.arange(1,ny-1)
   u[:,j]= -0.5*g*(height[:,j+1]-height[:,j-1])/dtheta / ((R+H[:,j])*(F[:,j]-1e-60))

   i = np.arange(0,nx) #with periodic BC in x
   v[i,:] = 0.5*g*(height[(i+1)%nx,:]-height[i-1,:])/dphi / ((R+H[i,:])*cos(Theta[i,:])*(F[i,:]-1e-60))
 
   v[:,0]  = 0
   v[:,-1] = 0

   u[:,0]  = u[:,1]
   u[:,-1] = u[:,-2]

   
   u[:,np.int(np.ceil(ny/2))] = u[:,np.int(np.ceil(ny/2))+1]
   v[:,np.int(np.ceil(ny/2))] = v[:,np.int(np.ceil(ny/2))+1]
   
   #%%
   # Don't allow the initial wind speed to exceed 200 m/s anywhere
   max_wind = 200
   u[u>max_wind]  =  max_wind
   u[u<-max_wind] = -max_wind
   v[v>max_wind]  =  max_wind
   v[v<-max_wind] = -max_wind



# Define h as the depth of the fluid (whereas "height" is the height of the upper surface)
h = height - H

# Initialize the 3D arrays where the output data will be stored
u_save = np.zeros([nx,ny,noutput+1])
v_save = np.zeros([nx,ny,noutput+1])
h_save = np.zeros([nx,ny,noutput+1])
t_save = np.zeros(noutput+1)


#%% Main loop -- time integration
# Index to stored data
i_save = 0

for n in range(nt):
    #Every fixed number of timesteps we store the fields
    if n%timesteps_between_outputs == 0:
      max_u = np.sqrt(np.max(u*u+v*v))
      print('Time = ' + str((n)*dt/3600) + ' hours (max ' + str(forecast_length_days*24) + '); max(|u|) = ' +str(max_u))
      u_save[:,:,i_save] = u
      v_save[:,:,i_save] = v
      h_save[:,:,i_save] = h
      t_save[i_save] = n*dt
      i_save = i_save+1
          
    #Compute the accelerations - terms NOT done by lax-wendroff, on current time-step
    i = np.arange(1,nx-1)
    j = np.arange(1,ny-1)
    [ii,jj] = np.meshgrid(i,j,indexing='ij')
    u_accel = F[ii,jj]*v[ii,jj] \
            - g/((R+H[ii,jj])*cos(Theta[ii,jj])) * (H[ii+1,jj]-H[ii-1,jj])/(2*dphi) \
            + u[ii,jj]*v[ii,jj]*tan(Theta[ii,jj])/(R+H[ii,jj])
         
    v_accel = -F[ii,jj]*u[ii,jj] \
            - g/((R+H[ii,jj])) * (H[ii,jj+1]-H[ii,jj-1])/(2*dtheta) \
            + u[ii,jj]*u[ii,jj]*tan(Theta[ii,jj])/(R+H[ii,jj])
    
    
    #Call the Lax-Wendroff scheme to move forward one timestep
    hnew, unew, vnew = lax_wendroff_sphere(dphi,dtheta,dt,g,u,v,h,H,R,Theta,u_accel,v_accel)
    
    #then add additional metric terms and accelerations
                                   
    # Update the wind and height fields, taking care to enforce  boundary conditions 
    u[ii,jj] = np.copy(unew)
    v[ii,jj] = np.copy(vnew)
    h[ii,jj] = np.copy(hnew)
    
    u[0,jj] = np.copy(unew[-1,:]) #NEED TO UPDATE THESE LINES
    v[0,jj] = np.copy(vnew[-1,:])
    h[0,jj] = np.copy(hnew[-1,:])

    u[-1,jj] = np.copy(unew[0,:])
    v[-1,jj] = np.copy(vnew[0,:])
    h[-1,jj] = np.copy(hnew[0,:])

    v[:,0] = 0
    v[:,-1] = 0
    
    u[:,0] = np.copy(u[:,1])
    u[:,-1] = np.copy(u[:,-2])
   
   
    h[ii,jj] = np.copy(hnew)
    h[:,0] = np.copy(h[:,1])
    h[:,-1] = np.copy(h[:,-2])
    

# save final time
max_u = np.sqrt(np.max(u*u+v*v))
print('Time = ' + str((n)*dt/3600) + ' hours (max ' + str(forecast_length_days*24) + '); max(|u|) = ' +str(max_u))
u_save[:,:,i_save] = u
v_save[:,:,i_save] = v
h_save[:,:,i_save] = h
t_save[i_save] = n*dt

#%%
vorticity =  np.zeros([nx,ny,noutput+1])
vorticity[ii,jj,:] = -(u_save[ii,jj+1,:]*cos(Theta[ii,jj+1]).reshape([nx-2,ny-2,1])-u_save[ii,jj-1,:]*cos(Theta[ii,jj-1]).reshape([nx-2,ny-2,1]) )/(Theta[ii,jj+1]-Theta[ii,jj-1]).reshape([nx-2,ny-2,1])\
                   +  (v_save[ii+1,jj,:]-v_save[ii-1,jj,:])/(Phi[ii+1,jj]-Phi[ii-1,jj]).reshape([nx-2,ny-2,1])

vorticity[ii,jj,:] =  vorticity[ii,jj,:]/(R*cos(Theta[ii,jj]).reshape([nx-2,ny-2,1]))

#%% Saving Data

# create folder
if os.path.isdir("./Data/"):
    print('Data folder already exists')
else: 
    print('Creating Data folder')
    os.makedirs("./Data")
filename = "./Data/FOM.npz"
np.savez(filename, Phi = Phi, Theta = Theta\
                 , h_save = h_save, u_save = u_save, v_save = v_save\
                 , t_save = t_save, vorticity = vorticity)
    

    

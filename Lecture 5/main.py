# assuming all nodes are evenly spaced
# edge exists between nodes
# stretching springs are edge-based quantities
# bending springs are node-based quantities

import numpy as np
import mae263f_functions as mf
import matplotlib.pyplot as plt


# function to create elastic force vector and its Hessian (stretching)
def getFs(q, EA, deltaL):
    # q = DOF vector of size N
    # EA = stretching stiffness of edges
    # deltaL + undefromed reference length (assume to be scalar for this simple example)
    # Outputs:
    # Fs = elastic force vector of size N (Gradient of elastic stretching force)
    # Js = elastic force matrix of size NxN (Hessian of elastic stretching force)

    Fs = np.zeros_like(q)
    Js = np.zeros((len(q), len(q)))
    
    # First stretching spring (USe A LOOP for the general case)
    xkm1 = q[0] # x_{k-1} x-coordinate of the first node
    ykm1 = q[1] # y_{k-1} y-coordinate of the first node
    xk = q[2]   # x_{k} x-coordinate of the second node
    yk = q[3]   # y_{k} y-coordinate of the second node

    gradEnergy = mf.gradEs(xkm1, ykm1, xk, yk, deltaL, EA)
    hessEnergy = mf.hessEs(xkm1, ykm1, xk, yk, deltaL, EA)

    Fs[0:4] += gradEnergy  # Add contribution to the force vector
    Js[0:4, 0:4] += hessEnergy  # Add contribution to the Hessian matrix

    # Second stretching spring (USe A LOOP for the general case)# First stretching spring (USe A LOOP for the general case)
    xkm1 = q[2] # x_{k-1} x-coordinate of the first node
    ykm1 = q[3] # y_{k-1} y-coordinate of the first node
    xk = q[4]   # x_{k} x-coordinate of the second node
    yk = q[5]   # y_{k} y-coordinate of the second node

    gradEnergy = mf.gradEs(xkm1, ykm1, xk, yk, deltaL, EA)
    hessEnergy = mf.hessEs(xkm1, ykm1, xk, yk, deltaL, EA)

    Fs[2:6] += gradEnergy  # Add contribution to the force vector
    Js[2:6, 2:6] += hessEnergy  # Add contribution to the Hessian matrix

    return Fs, Js

# function to create elastic force vector and its Hessian (bending)
def getFb(q, EI, deltaL):
    # q = DOF vector of size N
    # EI = bending stiffness
    # deltaL = undeformed Voronoi length (assume to be scalar for this simple example)
    # Outputs:
    # Fb = elastic force vector of size N (negative Gradient of elastic stretching force)
    # Jb = elastic force matrix of size NxN (negative Hessian of elastic stretching force)

    Fb = np.zeros_like(q)
    Jb = np.zeros((len(q), len(q)))
    
    # First stretching spring (USe A LOOP for the general case)
    xkm1 = q[0] # x_{k-1} x-coordinate of the first node
    ykm1 = q[1] # y_{k-1} y-coordinate of the first node
    xk = q[2]   # x_{k} x-coordinate of the second node
    yk = q[3]   # y_{k} y-coordinate of the second node
    xkp1 = q[4] # x_{k+1} x-coordinate of the third node
    ykp1 = q[5] # y_{k+1} y-coordinate of the third node

    gradEnergy = mf.gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI)
    hessEnergy = mf.hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI)

    Fb[0:6] += gradEnergy  # Add contribution to the force vector
    Jb[0:6, 0:6] += hessEnergy  # Add contribution to the Hessian matrix

    return Fb, Jb

# Objective function or integrator
def objfun(q_old, u_old, dt, tol, maximum_iter,
           m, M, # mass and mass matrix
           EI, EA, # elastic stiffness
           W, C, # external forces
           deltaL):
    
    q_new = q_old.copy() # initial guess

    # Newton-Raphson iteration
    iter_count = 0
    error = tol * 10 # error
    flag = 1 # if flag = 1, it is a good solution

    while error > tol:
        # inertia
        f_inertia = m/dt * ((q_new - q_old) / dt - u_old)
        J_inertia = M/dt**2 #TODO: check if this is correct

        # Elastic forces: stretching and bending
        Fs, Js = getFs(q_new, EA, deltaL)
        Fb, Jb = getFb(q_new, EI, deltaL)
        F_elastic = Fs + Fb
        J_elastic = Js + Jb

        # External forces
        # Viscous Forces
        Fv = - C @ ((q_new - q_old) / dt)
        Jv = - C / dt

        # Equation of Motion
        f = f_inertia + F_elastic + Fv - W
        J = J_inertia + J_elastic + Jv

        # Newton-Raphson update
        q_new = q_new - np.linalg.solve(J, f)
        
        # Get the error
        error = np.linalg.norm(f)

        # Update iteration count
        iter_count += 1
        if iter_count > maximum_iter:
            flag = -1
            print("Maximum number of iterations reached")
            return q_new, flag
        
    return q_new, flag


nv = 3 # number of vertices

# Time Step
dt = 0.01 # seconds

# Rod Length
RodLength = 0.1 # meters

# Discrete length / reference length
deltaL = RodLength / (nv - 1) # meters

# Radii of spheres (given)
R1 = 0.005 # meters
R2 = 0.025 # meters
R3 = 0.005 # meters

# Density of spheres (given)
rho_metal = 7000 # kg/m^3
rho_gl = 1000 # kg/m^3
rho = rho_metal - rho_gl # kg/m^3

# Cross section radius
r0 = 0.0005 # meters

# young's modulus
E = 1e9 # Pascals

# Viscosity 
visc = 1000 # Pa.s

# Maximum number of iterations
maximum_iter = 1000

# Total Time
totalTime = 10.0 # seconds

# Variables related to plotting
saveImage = 0
plotStep = 5 # plot every 'plotStep' time steps

# Utility quantities

ne = nv - 1 # number of edges
EI = np.pi * r0**4 / 4 * E # bending stiffness
EA = np.pi * r0**2 * E # stretching stiffness

# Tolerance
tol = EI / RodLength**2 * 1e-3

# Geometry
nodes = np.zeros((nv, 2)) # initialize node array
for c in range(nv):
    nodes[c, 0] = c * deltaL # x-coordinate
    nodes[c, 1] = 0.0        # y-coordinate

# Mass Vector and Matrix
m = np.zeros(nv * 2) # mass vector
m[0,2] = 4/3 * np.pi * R1**3 * rho # mass of first node
m[2,4] = 4/3 * np.pi * R2**3 * rho # mass of second node
m[4,6] = 4/3 * np.pi * R3**3 * rho # mass of third node
M = np.diag(m) # mass matrix

# Gravity (external force)
W = np.zeros(nv * 2)
g = np.array([0, -9.8]) # m/s^2
W[0:2] = 4.0 / 3.0 * np.pi * R1**3 * rho * g
W[2:4] = 4.0 / 3.0 * np.pi * R2**3 * rho * g
W[4:6] = 4.0 / 3.0 * np.pi * R3**3 * rho * g
# Gradient of W = 0

# Viscous Damping (external force)
C = np.zeros((nv * 2, nv * 2))
C1 = 6 * np.pi * visc * R1 # drag coefficient of first node
C2 = 6 * np.pi * visc * R2 # drag coefficient of second node    
C3 = 6 * np.pi * visc * R3 # drag coefficient of third node
C[0:2, 0:2] = C1 * np.eye(2)
C[2:4, 2:4] = C2 * np.eye(2)
C[4:6, 4:6] = C3 * np.eye(2)


# Initial Conditions
q0 = np.zeros(nv * 2) # initial position
for c in range(nv):
    q0[2 * c] = nodes[c, 0]     # x-coordinate
    q0[2 * c + 1] = nodes[c, 1] # y-coordinate

u0 = np.zeros(nv * 2) # initial velocity

# Number of steps
Nsteps = round(totalTime / dt)

ctime = 0 # current time

# Store the y-coordinate of the middle node, its velocity, and the angle
all_pos = np.zeros(Nsteps)
all_vel = np.zeros(Nsteps)
mid_angle = np.zeros(Nsteps)

# for middle node position, velocity and angle, at time step one equals zero

# Loop over the time steps
for time_step in range(1, Nsteps):

    q_new, err = objfun(q0, u0, dt, tol, maximum_iter,
                       m, M, # mass and mass matrix 
                          EI, EA, # elastic stiffness
                            W, C, # external forces
                                deltaL)
    if err == -1:
        print("Newton-Raphson did not converge")
        break

    u_new = (q_new - q0) / dt # update velocity

    ctime += dt # update current time

    # Save information about the middle node
    all_pos[time_step] = q_new[3]
    all_vel[time_step] = u_new[3]
    vec1 = np.array( q_new[2], q_new[3], 0 ) - np.array( q_new[0], q_new[1], 0 ) # second node - first node
    vec2 = np.array( q_new[4], q_new[5], 0 ) - np.array( q_new[2], q_new[3], 0 ) # third node - second node
    mid_angle[time_step] = np.degrees( np.arctan(np.linalg.norm( np.cross(vec1, vec2), np.dot(vec1, vec2) ))) # angle between first and last edge

    q0 = q_new.copy() # update position
    u0 = u_new.copy() # update velocity

    # Plotting
    if time_step % plotStep == 0:
        x_arr = q_new[::2] # q[0], q[2], q[4], ...
        y_arr = q_new[1::2] # q[1], q[3], q[5], ...
        h1 = plt.figure(1)
        plt.plot(x_arr, y_arr, '-o')
        plt.title('Time = %.2f s' % ctime)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        plt.show()

# Plot the middle node information as a function of time
t_arr = np.linspace(0, totalTime, Nsteps)
h2 = plt.figure(2)
plt.plot(t_arr, all_pos, 'ko-')

plt.xlabel('Time (s)')
plt.ylabel('Middle Node Y Position (m)')

plt.title('Middle Node Y Position vs Time')

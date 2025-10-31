import numpy as np
import mae263f_functions as mf
import getExternalForce as gef

def getForceJacobianImplicit(x_new, x_old, u_old, W,
                             stretch_stiffness_matrix, stretch_index_matrix, l_k_stretch,  
                             bending_stiffness_matrix, bending_index_matrix, l_k_bending,
                             m, dt):
    ndof = x_new.shape[0] # Number of degrees of freedom

    # Inertia
    f_inertia = m / dt * ( (x_new - x_old) / dt - u_old)
    J_inertia = np.diag(m) / dt ** 2

    # Stretching Spring 
    f_stretch = np.zeros(ndof)
    J_stretch = np.zeros((ndof, ndof))
    # Loop over each stretching spring
    for i in range(stretch_stiffness_matrix.shape[0]):
        ind = stretch_index_matrix[i].astype(int)
        xi = x_new[ind[0]]
        yi = x_new[ind[1]]
        xj = x_new[ind[2]]
        yj = x_new[ind[3]]
        stiffness = stretch_stiffness_matrix[i]
        dF = mf.gradEs(xi, yi, xj, yj, l_k_stretch[i], stiffness)
        dJ = mf.hessEs(xi, yi, xj, yj, l_k_stretch[i], stiffness)
        f_stretch[ind] += dF
        J_stretch[np.ix_(ind, ind)] += dJ
    

    # Bending Spring
    f_bending = np.zeros(ndof)
    J_bending = np.zeros((ndof, ndof))
    # Loop over each bending spring
    for i in range(bending_stiffness_matrix.shape[0]):
        ind = bending_index_matrix[i].astype(int)
        xi = x_new[ind[0]]
        yi = x_new[ind[1]]
        xj = x_new[ind[2]]
        yj = x_new[ind[3]]
        xk = x_new[ind[4]]
        yk = x_new[ind[5]]
        stiffness = bending_stiffness_matrix[i]
        dF = mf.gradEb(xi, yi, xj, yj, xk, yk, 0, l_k_bending[i], stiffness) # Assumes the beam is originally straight
        dJ = mf.hessEb(xi, yi, xj, yj, xk, yk, 0, l_k_bending[i], stiffness) # Assumes the beam is originally straight
        f_bending[ind] += dF
        J_bending[np.ix_(ind, ind)] += dJ

    # External Forces
    f_ext = W
    J_ext = np.zeros((ndof, ndof))

    f = f_inertia + f_stretch + f_bending - f_ext
    J = J_inertia + J_stretch + J_bending - J_ext

    return f, J


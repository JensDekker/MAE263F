import numpy as np
import mae263f_functions as mf
import getExternalForce as gef

def getForceJacobianImplicit(x_new, x_old, u_old, stiffness_matrix, index_matrix, m, dt, l_k):
    ndof = x_new.shape[0] # Number of degrees of freedom

    # Inertia
    f_inertia = m / dt * ( (x_new - x_old) / dt - u_old)
    J_inertia = np.diag(m) / dt ** 2

    # Spring 
    f_spring = np.zeros(ndof)
    J_spring = np.zeros((ndof, ndof))
    # Loop over each spring
    for i in range(stiffness_matrix.shape[0]):
        ind = index_matrix[i].astype(int)
        xi = x_new[ind[0]]
        yi = x_new[ind[1]]
        xj = x_new[ind[2]]
        yj = x_new[ind[3]]
        stiffness = stiffness_matrix[i]
        dF = mf.gradEs(xi, yi, xj, yj, l_k[i], stiffness)
        dJ = mf.hessEs(xi, yi, xj, yj, l_k[i], stiffness)
        f_spring[ind] += dF
        J_spring[np.ix_(ind, ind)] += dJ
    
    # External Forces
    f_ext = gef.getExternalForce(m)
    J_ext = np.zeros((ndof, ndof))

    f = f_inertia + f_spring - f_ext
    J = J_inertia + J_spring - J_ext

    return f, J


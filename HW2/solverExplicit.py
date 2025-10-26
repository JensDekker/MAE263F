import numpy as np
import mae263f_functions as mf
import getExternalForce as gef

def solverExplicit(x_old, u_old, stiffness_matrix, index_matrix, m, dt, l_k):
    ndof = x_old.shape[0] # Number of degrees of freedom

    # Spring
    f_spring = np.zeros(ndof)

    # Loop over each spring
    for i in range(stiffness_matrix.shape[0]):
        ind = index_matrix[i].astype(int)
        xi = x_old[ind[0]]
        yi = x_old[ind[1]]
        xj = x_old[ind[2]]
        yj = x_old[ind[3]]
        stiffness = stiffness_matrix[i]
        dF = mf.gradEs(xi, yi, xj, yj, l_k[i], stiffness)
        f_spring[ind] += dF

    # External Forces
    f_ext = gef.getExternalForce(m)

    f = -(f_spring - f_ext)

    x_new = x_old + dt * u_old + 0.5 * np.linalg.solve(np.diag(m), f) * dt**2 
    u_new = (x_new - x_old) / dt

    return x_new, u_new
    
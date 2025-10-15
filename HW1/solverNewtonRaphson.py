import numpy as np
import mae263f_functions as mf
import getForceJacobianImplicit as gfj

def solverNewtonRaphson(t_new, x_old, u_old, free_DOF, stiffness_matrix, index_matrix, m, dt, l_k):
    # t_new is optional for debugging 
    # It should calculate x_new and u_new knowing old positions and velocities
    # free_DOF is a vector containing the indicies of free variables (not boundary)

    # Initial Guess
    x_new = x_old.copy()

    tol = 1e-6 #TODO define a better tolerance
    err = 10 * tol

    # Newton-Raphson Iteration
    while err > tol:
        f, J = gfj.getForceJacobianImplicit(x_new, x_old, u_old, stiffness_matrix, index_matrix, m, dt, l_k)

        # Extract free DOFs
        f_free = f[free_DOF]
        J_free = J[np.ix_(free_DOF, free_DOF)]

        # Solve for the correction (deltaX)
        deltaX_free = np.linalg.solve(J_free, f_free)

        # Full deltaX
        deltaX = np.zeros_like(x_new)
        deltaX[free_DOF] = deltaX_free # Only update the free DOFs

        # Update x_new
        x_new = x_new - deltaX

        # Update error
        err = np.linalg.norm(f_free)

    u_new = (x_new - x_old) / dt

    return x_new, u_new
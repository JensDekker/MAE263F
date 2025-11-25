import numpy as np
import mae263f_functions as mf
import getForceJacobianImplicit as gfj
import format_number as fn

def solverNewtonRaphson(x_old, u_old, a1_old, a2_old, tol, W,
                        stretch_stiffness_matrix, l_k_stretch,  
                        bending_stiffness_matrix, voronoiRefLen, m1, m2, kappaBar,
                        twisting_stiffness_matrix, refTwist, twistBar,
                        m, dt, free_DOF, mu):
    # t_new is optional for debugging 
    # It should calculate x_new and u_new knowing old positions and velocities
    # free_DOF is a vector containing the indicies of free variables (not boundary)

    # Initial Guess - use first-order prediction for better convergence
    x_new = x_old

    err = 10 * tol
    prev_err = err + 1
    iter = 0
    max_iter = 50
    flag = 1

    
    # Newton-Raphson Iteration
    while err > tol:
        # Reference frame
        a1_new, a2_new = mf.computeTimeParallel(a1_old, x_old, x_new) # Time parallel reference frame along the rod
        # Reference twist
        tangent = mf.computeTangent(x_new)
        refTwist_new = mf.getRefTwist(a1_new, tangent, refTwist) # Reference twist vector of size nv
        # Material frame
        theta = x_new[3::4]
        m1, m2 = mf.computeMaterialDirectors(a1_new, a2_new, theta) # Material directors of size nv x 3
        
        f, J = gfj.getForceJacobianImplicit(x_new, x_old, u_old, W,
                             stretch_stiffness_matrix, l_k_stretch,  
                             bending_stiffness_matrix, voronoiRefLen, m1, m2, kappaBar,
                             twisting_stiffness_matrix, refTwist_new, twistBar,
                             m, dt, mu)

        # Extract free DOFs
        f_free = f[free_DOF]
        J_free = J[np.ix_(free_DOF, free_DOF)]

        
        deltaX_free = np.linalg.solve(J_free, f_free)
        
        # Full deltaX
        deltaX = np.zeros_like(x_new)
        deltaX[free_DOF] = deltaX_free # Only update the free DOFs
        
        # Update x_new
        x_new = x_new - deltaX

        # Update error
        err = np.linalg.norm(f_free)
        # Format error with proper alignment for iteration number
        print(fn.format_error_with_iteration(err, iter))

        iter += 1

        # If the error is decreasing, do not allow the iteration count to continue
        if err < prev_err:
            iter -= 1

        prev_err = err

        if iter > max_iter:
            flag = -1
            print("Maximum number of iterations reached")
            break

    u_new = (x_new - x_old) / dt

    return x_new, u_new, a1_new, a2_new, flag
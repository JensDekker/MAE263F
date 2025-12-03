import numpy as np
import mae263f_functions as mf
import getForceJacobianImplicit as gfj
import format_number as fn

# Code taken from Lecture 14 Colab Notebook

def solverNewtonRaphson(qOld, uOld, freeIndex, dt, tol, massVector, massMatrix,
                        ks, refLen, edges,
                        kb, hinges,
                        Fg, visc):
    
    qNew = qOld.copy()
    ndof = len(qOld) # Number of DOFs
  
    iter = 0 # number of iteration
    error = 10 * tol
    # Newton Raphson
    while error > tol:
  
        # Bending force and jacobian
        Fb = np.zeros( ndof )
        Jb = np.zeros( (ndof,ndof) )
        # Loop over every "bending spring" or "hinge"
        for kHinge in range(hinges.shape[0]):
            ind = hinges[kHinge].astype(int)
            x0 = qNew[ind[0:3]]
            x1 = qNew[ind[3:6]]
            x2 = qNew[ind[6:9]]
            x3 = qNew[ind[9:12]]
            dF, dJ = mf.gradEb_hessEb_Shell(x0, x1, x2, x3, 0, kb[kHinge])
            Fb[ind] -= dF
            Jb[np.ix_(ind,ind)] -= dJ
  
        # Stretching force and jacobian
        Fs = np.zeros( ndof )
        Js = np.zeros( (ndof,ndof) )
        for kEdge in range(edges.shape[0]):
            ind = edges[kEdge].astype(int)
            x0 = qNew[ind[0:3]]
            x1 = qNew[ind[3:6]]
            dF, dJ = mf.gradEs_hessEs(x0, x1, refLen[kEdge], ks[kEdge])
            Fs[ind] -= dF
            Js[np.ix_(ind,ind)] -= dJ
    
        # Viscous force
        Fv = - visc * (qNew - qOld) / dt
        Jv = - visc / dt * np.eye(ndof)
    
        Forces = Fb + Fs + Fg + Fv # Sum of forces
        JForces = Jb + Js + Jv # Sum of Jacobians
    
        # Set up my equations of motion and calculating its residual (=0)
        f = massVector / dt * ( (qNew - qOld)/dt - uOld) - Forces # Residual of EOM
        J = massMatrix / dt ** 2 - JForces
    
        # Extract the free part of the f and J arrays
        f_free = f[freeIndex]
        J_free = J[np.ix_(freeIndex, freeIndex)]
        # Correction
        dq_free = np.linalg.solve(J_free, f_free)
    
        # Update my guess for position
        qNew[freeIndex] -= dq_free
    
        # Calculate error
        error = np.sum( np.abs(f_free))
    
        iter += 1
    
        # print('Iter = ', iter, ' error=', error)
  
    uNew = (qNew - qOld) / dt

    return qNew, uNew
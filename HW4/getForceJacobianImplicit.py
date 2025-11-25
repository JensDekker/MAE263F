import numpy as np
import mae263f_functions as mf
import getExternalForce as gef

def getForceJacobianImplicit(x_new, x_old, u_old, W,
                             stretch_stiffness_matrix, l_k_stretch,  
                             bending_stiffness_matrix, voronoiRefLen, m1, m2, kappaBar,
                             twisting_stiffness_matrix, refTwist, twistBar,
                             m, dt, mu):
    
    ndof = x_new.shape[0] # Number of degrees of freedom

    # Inertia
    f_inertia = m / dt * ( (x_new - x_old) / dt - u_old)
    J_inertia = np.diag(m) / dt ** 2

    # Stretching Spring 
    f_stretch, J_stretch = mf.getFs(x_new, stretch_stiffness_matrix, l_k_stretch)

    # Bending Spring
    f_bending, J_bending = mf.getFb(x_new, m1, m2, kappaBar, bending_stiffness_matrix, voronoiRefLen)

    # Twisting Spring
    f_twisting, J_twisting = mf.getFt(x_new, refTwist, twistBar, twisting_stiffness_matrix, voronoiRefLen)

    # External Forces
    f_ext = W
    J_ext = np.zeros((ndof, ndof))

    # Viscous Forces
    f_viscous = - mu * (x_new - x_old) / dt
    J_viscous = - mu / dt * np.eye(ndof)

    f = f_inertia - (f_stretch + f_bending + f_twisting + f_viscous + f_ext)
    J = J_inertia - (J_stretch + J_bending + J_twisting + J_viscous + J_ext)

    return f, J

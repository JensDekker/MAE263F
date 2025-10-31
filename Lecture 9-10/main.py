import numpy as np
import matplotlib.pyplot as plt
import mae263f_functions as mf

def signedAngle(u, v, n):

    # gets the angle between u and v around n
    # u: from vector
    # v: to vector
    # n: "direction" vector for sign 

    # Error check that N is normal to u and v
    if not np.isclose(np.dot(n, u), 0) or not np.isclose(np.dot(n, v), 0):
        raise ValueError("N is not normal to u and v")

    w = np.cross(u, v)
    angle = np.arctan2( np.linalg.norm(w), np.dot(u, v) )

    if np.dot(n, w) < 0:
        angle = -angle
    return angle

def test_signedAngle():
    u = np.array([1, 0, 0])
    v = np.array([0, 1, 0])
    n = np.array([0, 0, 1])
    angle = signedAngle(u, v, n)
    print(angle)

def rotateAxisAngle(u, n, angle):
    # rotates u around n by angle
    # magnitude of u is preserved
    if angle == 0:
        return u
    c = np.cos(angle)
    s = np.sin(angle)
    v = c * u + s * np.cross(n, u) + (1 - c) * np.dot(n, u) * n
    # think of it as a sum of components:
    # 1. c * u: the component in the u direction
    # 2. s * np.cross(n, u): the component in the direction of the cross product of n and u
    # 3. (1 - c) * np.dot(n, u) * n: the component in the direction of n
    return v

def test_rotateAxisAngle():
    u = np.array([1, 0, 0])
    n = np.array([0, 0, -1])
    angle = np.pi / 4
    v = rotateAxisAngle(u, n, angle)
    print(v)

def parallel_transport(u, t1, t2):
    # u: vector (usually reference frame director such as a1 a2) that needs to be transported
    # t1 (from): tangent on the edge from where to be transported, also called t_e for early edge
    # t2 (to): tangent on the edge where to be transported, also called t_f for final edge

    b = np.cross(t1, t2)
    if np.linalg.norm(b) == 0:
        return u
    else:
        b = b / np.linalg.norm(b)
        # Safety checks to ensure that b is normal to t1 and t2
        b = b - np.dot(b, t1) * t1 - np.dot(b, t2) * t2
        b = b / np.linalg.norm(b)   


        n1 = np.cross(t1, b)
        n2 = np.cross(t2, b)
        d = np.dot(u,t1) * t2 + np.dot(u,n1) * n2 + np.dot(u,b) * b
        return d

def test_parallel_transport():
    u = np.array([0, 0, 1])
    t1 = np.array([1, 0, 0])
    t2 = np.array([0, 1, 0])
    d = parallel_transport(u, t1, t2)
    print(d)


def computeReferenceTwist(a1e, a1f, t1, t2, refTwist=None):
    if refTwist is None:
        refTwist = 0
    P_a1e = parallel_transport(a1e, t1, t2)
    P_a1e_t = rotateAxisAngle(P_a1e, t2, refTwist)
    refTwist += signedAngle(P_a1e_t, a1f, t2)
    return refTwist


def getRefTwist(a1, tangent, refTwist=None):
    # Given all the reference frames along the rod, we calculate the reference twist along the rod on every node
    ne = a1.shape[0] # number of edges; shape of a1 is (nv, 3)
    nv = ne + 1 # number of vertices

    if refTwist is None: # no guess is provided
        refTwist = np.zeros(nv) # initialize all zeros
    for c in np.arange(1, ne): # all internal nodes (not first or last)
        u0 = a1[c-1, 0:3]
        u1 = a1[c  , 0:3]
        t1 = tangent[c-1, 0:3]
        t2 = tangent[c  , 0:3]
        refTwist[c]= computeReferenceTwist(u0, u1, t1, t2, refTwist[c])
    return refTwist

# NOTE: Reference frame is a numerical convenience so that a single scalar DOF (theta) per edge is needed to fully describe the material frame


def computekappa(node0, node1, node2, m1e, m2e, m1f, m2f):
    t0 = (node1 - node0) / np.linalg.norm(node1 - node0)
    t1 = (node2 - node1) / np.linalg.norm(node2 - node1)

    kb = 2.0 * np.cross(t0, t1) / (1.0 + np.dot(t0, t1))
    kappa1 = 0.5 * np.dot(kb, m2e + m2f)
    kappa2 = - 0.5 * np.dot(kb, m1e + m1f)

    kappa = np.zeros(2)
    kappa[0] = kappa1
    kappa[1] = kappa2
    return kappa

def getKappa(q, m1, m2):
    nv = (len(q) + 1) / 4
    ne = nv - 1

    kappa = np.zeros( (nv, 2) )

    for c in range(2, nv):
        node0 = q[4*c-8:4*c-5]
        node1 = q[4*c-4:4*c-1]
        node2 = q[4*c:4*c+3]

        m1e = m1[c-2,:].flatten()
        m2e = m2[c-2,:].flatten()
        m1f = m1[c-1,:].flatten()
        m2f = m2[c-1,:].flatten()

        kappa_local = computekappa(node0, node1, node2, m1e, m2e, m1f, m2f)
        kappa[c-1,0] = kappa_local[0]
        kappa[c-1,1] = kappa_local[1]
    return kappa


def computeTangent(q):
    # q is the DOF vector of size 4*nv - 1 = 3 * nv + ne
    nv = (len(q) + 1) / 4
    ne = nv - 1

    tangent = np.zeros( (ne, 3) ) # every edge has a tangent vector

    for c in range(ne):
        node0 = q[4*c:4*c+3]
        node1 = q[4*c+4:4*c+7]
        edge = node1 - node0
        tangent[c,:] = edge / np.linalg.norm(edge)
    return tangent


def computeMaterialDirectors(a1, a2, theta):

    # a1 = matrix of size ne x 3 - First reference director
    # a2 = matrix of size ne x 3 - Second reference director
    # theta = vecotre of size ne (extracted from the DOF vector; every fourth element of the DOF vector)

    ne = len(theta)
    m1 = np.zeros_like(a1)
    m2 = np.zeros_like(a2)

    for c in range(ne):
        cs = np.cos(theta[c])
        sn = np.sin(theta[c])
        m1[c,:] = cs * a1[c,:] + sn * a2[c,:]
        m2[c,:] = -sn * a1[c,:] + cs * a2[c,:]  
    return m1, m2


def computeSpaceParallel(u1_first, q):
    # u1_first = first reference frame vector (arbitrary but orthonormal adapted) on the first edge
    # q is the DOF vector of size 4*nv - 1
    # used for the initial configuration at t = 0
    nv = (len(q) + 1) / 4
    ne = nv - 1

    tangent = computeTangent(q)
    u1 = np.zeros( (ne, 3) )
    u2 = np.zeros( (ne, 3) )

    # First edge
    u1_first = u1_first / np.linalg.norm(u1_first)
    u1[0,:] = u1_first
    t0 = tangent[0,:]
    u2[0,:] = np.cross(t0, u1_first)
    u2[0,:] = u2[0,:] / np.linalg.norm(u2[0,:])

    # Internal edges
    for c in np.arange(1, ne):
        t0 = tangent[c-1,:] # tangent "from"
        t1 = tangent[c,:] # tangent "to"
        
        u1[c,:] = parallel_transport(u1[c-1,:]  , t0, t1)
        u1[c,:] = u1[c,:] / np.linalg.norm(u1[c,:])

        u2[c,:] = np.cross(t1, u1[c,:])
        u2[c,:] = u2[c,:] / np.linalg.norm(u2[c,:])
    
    return u1, u2

# NOTE: a1(t = 0) = u1  and a2(t = 0) = u2


def computeTimeParallel(a1_old, q0, q):
    # a1_old: First time parallel frame director in "old" configuration
    # q0: the old shape of the rod / DOF vector
    # q: the new shape of the rod / DOF vector
    # a1 on this new shape is unknown

    nv = (len(q) + 1) / 4
    ne = nv - 1

    tangent0 = computeTangent(q0) # from old configuration
    tangent = computeTangent(q) # from new configuration

    a1 = np.zeros( (ne, 3) )
    a2 = np.zeros( (ne, 3) )

    for c in np.arange(ne): # Loop over every edge
        t0 = tangent0[c,:] # tangent "from"
        t1 = tangent[c,:] # tangent "to"
        a1[c,:] = parallel_transport(a1_old[c,:], t0, t1)
        a1[c,:] = a1[c,:] / np.linalg.norm(a1[c,:])
        a2[c,:] = np.cross(t1, a1[c,:])
        a2[c,:] = a2[c,:] / np.linalg.norm(a2[c,:])
    return a1, a2


# Twist Computation

# Descriptoino of a rod (3 nodes only)

xkm1 = np.array([0, 0, 0])
xk = np.array([1, 0, 0])
xkp1 = np.array([1, 1, 0])

# Material frames
m1e = np.array([0, 1, 0])
m2e = np.array([0, 0, 1]) # most likely never used
m1f = np.array([-0.707, 0, 0.707])
m2f = np.array([0.707, 0, 0.707]) # most likely never used

# -----------------------------------------------------------------------------------
# Reference frame (orthonormal adapted but otherwise arbitrary)
a1e = np.array([0, 0.707, 0.707])
a1e = a1e / np.linalg.norm(a1e)
a1f = np.array([-1,0,0])
a1f = a1f / np.linalg.norm(a1f)

# -----------------------------------------------------------------------------------

e1 = xk - xkm1
e2 = xkp1 - xk

t1 = e1 / np.linalg.norm(e1)
t2 = e2 / np.linalg.norm(e2)
print('t1: ', t1)
print('t2: ', t2)

theta_e = signedAngle(a1e, m1e, t1) # DOF
theta_f = signedAngle(a1f, m1f, t2) # DOF

print('Theta_e: ', theta_e, 'Theta_f: ', theta_f)

# NOTE: If a1e and a1f were "space parallel," then my twist = theta_f - theta_e
# However, a1e and a1f are not twist free. So, I need to add the twist of the reference frame to the twist

P_a1e = parallel_transport(a1e, t1, t2)
refTwist = signedAngle(P_a1e, a1f, t2)
print('refTwist: ', refTwist)

# Discrete integrated Twist (or just"Twist")
descrete_twist_reframe = theta_f - theta_e + refTwist
print('descrete_twist_reframe: ', descrete_twist_reframe)


'''
WORK FROM LECTURE 9

# Compute twist
P_m1e = parallel_transport(m1e, t1, t2)
discrete_twist = signedAngle(P_m1e, m1f, t2)
print('discrete_twist: ', discrete_twist, np.pi / 4)

# Compute twist using m2
P_m2e = parallel_transport(m2e, t1, t2)
discrete_twist_m2 = signedAngle(P_m2e, m2f, t2)
print('discrete_twist_m2: ', discrete_twist_m2, np.pi / 4)


# General Knowledge
l_k = 0.5 * (np.linalg.norm(e1) + np.linalg.norm(e2)) # Voronoi length of the node
twist_continuous = discrete_twist / l_k 

undeformed_twist = 0 # usually given
GJ = 50 # Nm^2
E_k = 0.5 * GJ / l_k * (discrete_twist - undeformed_twist) ** 2
print('E_k: ', E_k)

WORK FROM LECTURE 10
'''
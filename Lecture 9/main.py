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

e1 = xk - xkm1
e2 = xkp1 - xk

t1 = e1 / np.linalg.norm(e1)
t2 = e2 / np.linalg.norm(e2)
print('t1: ', t1)
print('t2: ', t2)

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
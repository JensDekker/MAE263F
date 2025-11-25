import numpy as np 

def signedAngle(u = None,v = None,n = None):
    w = np.cross(u,v)
    angle = np.arctan2( np.linalg.norm(w), np.dot(u,v) )
    if (np.dot(n,w) < 0):
        angle = - angle
    return angle
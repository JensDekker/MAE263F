import numpy as np
def targetNodePosition(t_new, length):

    target_x = length / 2 * np.cos( np.pi / 2 * (t_new / 1000))
    target_y = - length / 2 * np.sin( np.pi / 2 * (t_new / 1000))

    return target_x, target_y
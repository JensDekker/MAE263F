import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import mae263f_functions as mf

def plotShell_hinge(positions, bending_index_matrix, ctime, output_dir=None, show_plots=True):

    fig = plt.figure(1)
    clear_output()
    plt.clf()  # Clear the figure
    ax = fig.add_subplot(111, projection='3d')
  
    for i in range(bending_index_matrix.shape[0]):
        ind = bending_index_matrix[i].astype(int)
        xi = positions[ind[0]]
        yi = positions[ind[1]]
        zi = positions[ind[2]]
        xj = positions[ind[3]]
        yj = positions[ind[4]]
        zj = positions[ind[5]]

        ax.plot3D([xi, xj], [yi, yj], [zi, zj], 'ko-')
  
    # Set the title with current time
    ax.set_title(f't={ctime:.2f}')
    
    # Set axes labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
  
    # Set equal scaling and a 3D view
    mf.set_axes_equal(ax)
    plt.draw()  # Force a redraw of the figure
  
    
    if show_plots:
        plt.show()
    if output_dir is not None:
        plt.savefig(f'{output_dir}/plot_t_{ctime:.2f}.png')
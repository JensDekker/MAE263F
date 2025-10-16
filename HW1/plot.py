import matplotlib.pyplot as plt
import os

def plot(x, index_matrix, t, save_plots=True, output_dir='plots'):
    plt.figure()
    plt.title(f'Time: {t:.2f} second')
    for i in range(index_matrix.shape[0]): # All springs
        ind = index_matrix[i].astype(int)
        xi = x[ind[0]]
        yi = x[ind[1]]
        xj = x[ind[2]]
        yj = x[ind[3]]
        plt.plot([xi, xj], [yi, yj], 'bo-')
    plt.axis('equal')
    plt.xlabel('x [meter]')
    plt.ylabel('y [meter]')
    
    if save_plots:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save plot with timestamp in filename
        filename = f'{output_dir}/plot_t_{t:.2f}s.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'Plot saved to: {filename}')
    
    # plt.show()
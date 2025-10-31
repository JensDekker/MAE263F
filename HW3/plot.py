import matplotlib.pyplot as plt
import os

def plot(x, index_matrix, t, save_plots=True, output_dir='plots', expected_deflection=None, expected_max_deflection_location=None, applied_force=None):
    plt.figure()
    if applied_force is not None:
        plt.title(f'Time: {t:.2f} s, Force: {applied_force} N')
    else:
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
    
    if expected_deflection is not None:
        plt.axhline(y=expected_deflection, color='r', linestyle='--', label='Expected Deflection')
        plt.legend()

    if expected_max_deflection_location is not None:
        plt.axvline(x=expected_max_deflection_location, color='g', linestyle='--', label='Expected Max Deflection Location')
        plt.legend()

    if save_plots:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save plot with timestamp in filename
        if applied_force is not None:
            filename = f'{output_dir}/plot_t_{t:.2f}s_F{int(applied_force)}N.png'
        else:
            filename = f'{output_dir}/plot_t_{t:.2f}s.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'Plot saved to: {filename}')
    
    # plt.show()
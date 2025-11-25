"""
Script to plot time vs. last element of position array (z-coordinate of final node)
from the filtered output data file.
"""

import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
from typing import Optional, Tuple, List


def extract_last_element_from_positions(pos_str: str) -> float:
    """
    Extract the last element from the positions array string.
    Uses regex to find the last number before the closing bracket.
    """
    if not pos_str:
        return np.nan
    
    # Find the last number before ]" or ]",1
    # Pattern: find the last number before ]"
    pattern = r'([-+]?\d+\.?\d*[eE]?[-+]?\d*)\]\",'
    match = re.search(pattern, pos_str)
    
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    
    # Fallback: try to find last number before ]
    pattern2 = r'([-+]?\d+\.?\d*[eE]?[-+]?\d*)\]\s*"'
    match2 = re.search(pattern2, pos_str)
    if match2:
        try:
            return float(match2.group(1))
        except ValueError:
            return np.nan
    
    return np.nan


def read_parameters(parameters_file: str = 'HW4/springNetworkData/parameters.txt') -> dict:
    """Read parameters from the parameters file."""
    params = {}
    if not os.path.exists(parameters_file):
        return params
    
    with open(parameters_file, 'r') as f:
        for line in f:
            parts = [part.strip() for part in line.split(':')]
            if len(parts) == 2:
                try:
                    params[parts[0]] = float(parts[1])
                except ValueError:
                    continue
    return params


def calculate_steady_time(parameters_file: str = 'HW4/springNetworkData/parameters.txt') -> Optional[float]:
    """Calculate steady_time from parameters (5 * time_scale). Returns None if parameters not available."""
    params = read_parameters(parameters_file)
    
    if 'L' not in params or 'E' not in params or 'rho' not in params:
        return None
    
    L = params["L"]
    E = params["E"]
    rho = params["rho"]
    time_scale = L / np.sqrt(E / rho)
    steady_time = 5 * time_scale
    return steady_time


def read_csv_data(csv_file: str) -> Tuple[list, list, list]:
    """Read time, z_difference, and load data from CSV file."""
    times = []
    z_differences = []
    loads = []
    
    if not os.path.exists(csv_file):
        return times, z_differences, loads
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 3:
                try:
                    times.append(float(row[0]))
                    z_differences.append(float(row[1]))
                    loads.append(float(row[2]))
                except ValueError:
                    continue
    
    return times, z_differences, loads


def calculate_steady_state_from_csv(times: list, z_differences: list, 
                                     parameters_file: str = 'HW4/springNetworkData/parameters.txt',
                                     final_period_fraction: float = 0.1) -> Optional[float]:
    """
    Calculate steady state value by averaging min and max in the final period.
    
    Parameters:
    -----------
    times : list
        List of time values
    z_differences : list
        List of z-coordinate differences
    parameters_file : str
        Path to parameters file (to calculate steady_time)
    final_period_fraction : float
        Fraction of total time to use as final period if steady_time not available (default: 0.1)
    
    Returns:
    --------
    steady_state_value : float or None
        Steady state value (average of min and max in final period)
    """
    if not times or not z_differences or len(times) != len(z_differences):
        return None
    
    # Try to get steady_time from parameters
    steady_time = calculate_steady_time(parameters_file)
    
    max_time = max(times)
    
    if steady_time is not None:
        # Use steady_time period
        time_threshold = max_time - steady_time
    else:
        # Use fraction of total time range
        time_range = max_time - min(times)
        time_threshold = max_time - (time_range * final_period_fraction)
    
    # Find data points in the final period
    final_period_indices = [i for i in range(len(times)) if times[i] >= time_threshold]
    
    if not final_period_indices:
        # If no data in final period, use all data
        final_period_indices = list(range(len(times)))
    
    # Get z_differences in final period
    final_period_z_diffs = [z_differences[i] for i in final_period_indices]
    
    if not final_period_z_diffs:
        return None
    
    # Calculate average of min and max
    min_val = min(final_period_z_diffs)
    max_val = max(final_period_z_diffs)
    steady_state_value = (min_val + max_val) / 2.0
    
    return steady_state_value


def save_plot_data_to_csv(times: list, z_differences: list, loads: list, 
                          output_csv_file: str):
    """
    Save the plotting data to a CSV file for easier reference.
    
    Parameters:
    -----------
    times : list
        List of time values
    z_differences : list
        List of z-coordinate differences from initial
    loads : list
        List of load values
    output_csv_file : str
        Path to save the CSV file
    """
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Time', 'Z_Coordinate_Difference', 'Load'])
        # Write data
        for time, z_diff, load in zip(times, z_differences, loads):
            writer.writerow([time, z_diff, load])
    print(f"Plot data saved to: {output_csv_file}")


def parse_positions_array(pos_str: str) -> Optional[np.ndarray]:
    """
    Parse the positions array string into a numpy array.
    
    Parameters:
    -----------
    pos_str : str
        String representation of positions array like "[ x0 y0 z0 theta0 x1 y1 z1 theta1 ... ]"
        Can span multiple lines
    
    Returns:
    --------
    positions : np.ndarray or None
        Array of positions (DOF vector), or None if parsing fails
    """
    if not pos_str:
        return None
    
    try:
        # Remove brackets and extract all numbers using regex
        # This handles multi-line arrays with various whitespace
        values = re.findall(r'[-+]?\d+\.?\d*[eE]?[-+]?\d*', pos_str)
        if not values:
            return None
        positions = np.array([float(v) for v in values])
        return positions
    except (ValueError, AttributeError, TypeError):
        return None


def extract_full_data_from_file(input_file: str) -> Tuple[List[float], List[float], List[np.ndarray]]:
    """
    Extract time, load, and full position arrays from the output file.
    
    Returns:
    --------
    times : List[float]
        List of time values
    loads : List[float]
        List of load values
    positions_list : List[np.ndarray]
        List of position arrays (DOF vectors)
    """
    times = []
    loads = []
    positions_list = []
    
    if not os.path.exists(input_file):
        print(f"Warning: File not found: {input_file}")
        return times, loads, positions_list
    
    # Read file and extract full data
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match each row: Load,Time,Time Step,Positions,Velocities,Flags
    # Positions array spans multiple lines until we hit ]","[
    # Use a pattern that matches Load,Time,Time Step, then captures everything until ]","[
    # The positions array is between "[ and ]","[
    pattern = r'([\d\.eE\+\-]+),([\d\.eE\+\-]+),([\d\.eE\+\-]+),\"\[(.*?)\]\",\"\[' 
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        try:
            load_val = float(match.group(1))
            time_val = float(match.group(2))
            pos_str_content = match.group(4)  # Content between [ and ]
            
            # Reconstruct the positions array string
            pos_str = '[' + pos_str_content + ']'
            
            # Parse positions array
            positions = parse_positions_array(pos_str)
            
            if positions is not None and len(positions) > 0:
                times.append(time_val)
                loads.append(load_val)
                positions_list.append(positions)
        except (ValueError, IndexError, AttributeError):
            continue
    
    return times, loads, positions_list


def set_axes_equal(ax):
    """
    Set equal aspect ratio for a 3D plot in Matplotlib.
    This function adjusts the limits of the plot to make sure
    that the scale is equal along all three axes.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def plot_helix_snapshot(q: np.ndarray, time: float, output_file: str):
    """
    Plot a 3D snapshot of the helix centerline.
    
    Parameters:
    -----------
    q : np.ndarray
        Position vector (DOF vector) with format [x0, y0, z0, theta0, x1, y1, z1, theta1, ...]
    time : float
        Current simulation time
    output_file : str
        Path to save the snapshot
    """
    # Extract x, y, z coordinates (every 4th element starting at indices 0, 1, 2)
    x = q[0::4]
    y = q[1::4]
    z = q[2::4]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the rod as black circles connected by lines
    ax.plot3D(x, y, z, 'ko-', markersize=4, linewidth=1.5)
    
    # Plot the first node with a red triangle
    ax.plot3D([x[0]], [y[0]], [z[0]], 'r^', markersize=10)
    
    # Set axes labels with units
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_zlabel('z [m]', fontsize=12)
    
    # Set equal scaling
    set_axes_equal(ax)
    
    # Add time label directly on the figure
    ax.text2D(0.02, 0.98, f't = {time:.2f} s', transform=ax.transAxes,
              fontsize=14, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Snapshot saved: {output_file}")


def create_helix_snapshots_from_data(times: List[float], loads: List[float], 
                                     positions_list: List[np.ndarray],
                                     input_file: str, output_dir: str, 
                                     num_snapshots: int = 5):
    """
    Create 3D snapshots from already-extracted position data.
    
    Parameters:
    -----------
    times : List[float]
        List of time values
    loads : List[float]
        List of load values
    positions_list : List[np.ndarray]
        List of position arrays (DOF vectors)
    input_file : str
        Path to the output data file (for naming output files)
    output_dir : str
        Directory to save snapshot images
    num_snapshots : int
        Number of snapshots to create (default: 5)
    """
    if not times or not positions_list:
        print(f"Warning: No position data provided")
        return
    
    # Determine snapshot times (evenly distributed)
    max_time = max(times)
    snapshot_times = np.linspace(0, max_time, num_snapshots)
    
    # Extract base name from input file for output naming
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating {num_snapshots} snapshots from {os.path.basename(input_file)}...")
    print(f"  Time range: 0 to {max_time:.6e} s")
    
    # Find closest data points to snapshot times and create plots
    for i, target_time in enumerate(snapshot_times):
        # Find the index of the closest time
        time_diffs = [abs(t - target_time) for t in times]
        closest_idx = np.argmin(time_diffs)
        actual_time = times[closest_idx]
        
        # Get positions at this time
        q = positions_list[closest_idx]
        
        # Create output filename
        output_file = os.path.join(output_dir, f"{base_name}_t{actual_time:.2f}s.png")
        
        # Create snapshot
        plot_helix_snapshot(q, actual_time, output_file)


def create_helix_snapshots(input_file: str, output_dir: str, 
                            num_snapshots: int = 5,
                            parameters_file: str = 'HW4/springNetworkData/parameters.txt'):
    """
    Create 3D snapshots of the helix centerline at different simulation times.
    Reads the full data from the input file.
    
    Parameters:
    -----------
    input_file : str
        Path to the output data file
    output_dir : str
        Directory to save snapshot images
    num_snapshots : int
        Number of snapshots to create (default: 5)
    parameters_file : str
        Path to parameters file (for calculating max time)
    """
    # Extract full data
    times, loads, positions_list = extract_full_data_from_file(input_file)
    
    if not times or not positions_list:
        print(f"Warning: Could not extract data from {input_file}")
        return
    
    # Use the shared function
    create_helix_snapshots_from_data(times, loads, positions_list, input_file, output_dir, num_snapshots)


def plot_time_vs_z_coordinate(input_file: str, output_file: str = None, 
                              save_csv: bool = True,
                              parameters_file: str = 'HW4/springNetworkData/parameters.txt',
                              show_steady_state: bool = True,
                              use_csv_if_available: bool = True,
                              create_snapshots: bool = True,
                              num_snapshots: int = 5,
                              snapshot_output_dir: str = None):
    """
    Plot time vs. z-coordinate of final node from the completed output data.
    
    Parameters:
    -----------
    input_file : str
        Path to the completed output data file
    output_file : str, optional
        Path to save the plot (if None, displays interactively)
    save_csv : bool, optional
        Whether to save the plotting data to a CSV file (default: True)
    parameters_file : str, optional
        Path to parameters file for steady state calculation
    show_steady_state : bool, optional
        Whether to show steady state line on plot (default: True)
    use_csv_if_available : bool, optional
        If True, use CSV file for plotting data if it exists (default: True)
    create_snapshots : bool, optional
        Whether to create 3D helix snapshots (default: True)
    num_snapshots : int, optional
        Number of snapshots to create (default: 5)
    snapshot_output_dir : str, optional
        Directory to save snapshots (default: same directory as output_file or HW4/plots/snapshots)
    """
    times = []
    z_coords = []
    loads = []
    z_differences = []
    starting_z = 0.0
    
    # Check if CSV exists and should be used
    csv_file = None
    if use_csv_if_available and output_file:
        csv_file = output_file.rsplit('.', 1)[0] + '_data.csv'
        if os.path.exists(csv_file):
            print(f"Reading data from CSV: {csv_file}")
            csv_times, csv_z_diffs, csv_loads = read_csv_data(csv_file)
            if csv_times and csv_z_diffs:
                times = csv_times
                z_differences = csv_z_diffs
                loads = csv_loads
                # Calculate starting z from first value (z_differences[0] should be 0)
                # We need to reconstruct z_coords to get starting_z
                # Since z_differences = z_coords - starting_z, and first z_diff is 0,
                # we can't recover starting_z from CSV alone, but we don't need it for plotting
                starting_z = 0.0  # Will be handled when we calculate differences
                print(f"Loaded {len(times)} data points from CSV")
    
    # Store full position data for snapshots (only if we read from text file)
    full_times = []
    full_loads = []
    full_positions_list = []
    
    # If CSV not used or not available, read from text file
    if not times:
        print(f"Reading data from: {input_file}")
        
        if not os.path.exists(input_file):
            print(f"Error: File not found: {input_file}")
            sys.exit(1)
        
        # If snapshots are needed, extract full position data
        if create_snapshots:
            print("Extracting full position data for snapshots...")
            full_times, full_loads, full_positions_list = extract_full_data_from_file(input_file)
            # Use the full data for plotting too
            if full_times:
                times = full_times
                loads = full_loads
                # Extract z-coords from full positions (last element of each position array)
                z_coords = [pos[-1] if len(pos) > 0 else 0.0 for pos in full_positions_list]
            else:
                print("Warning: Could not extract full position data, falling back to simple extraction")
        
        # If we don't have full data yet (snapshots disabled or extraction failed), use simple extraction
        if not times:
            # Read file and extract data using regex (handles multi-line CSV)
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract times and last elements using regex patterns
            # Pattern 1: Extract Load,Time,Time Step from the beginning of each row
            time_pattern = r'([\d\.eE\+\-]+),([\d\.eE\+\-]+),([\d\.eE\+\-]+),\"\['
            time_matches = re.findall(time_pattern, content)
            
            # Pattern 2: Extract last element from positions arrays
            # Match number before ]","[ to ensure we only get positions, not velocities
            pos_pattern = r'([-+]?\d+\.?\d*[eE]?[-+]?\d*)\]\",\"\['
            pos_matches = re.findall(pos_pattern, content)
            
            print(f"Found {len(time_matches)} time entries")
            print(f"Found {len(pos_matches)} position entries")
            
            # Match them up
            min_len = min(len(time_matches), len(pos_matches))
            for i in range(min_len):
                try:
                    load_val = float(time_matches[i][0])
                    time_val = float(time_matches[i][1])
                    z_coord = float(pos_matches[i])
                    
                    times.append(time_val)
                    z_coords.append(z_coord)
                    loads.append(load_val)
                except (ValueError, IndexError):
                    continue
        
        if not times:
            print("Error: No valid data found in file")
            sys.exit(1)
        
        # Calculate difference from starting z-coordinate
        if len(z_coords) > 0:
            starting_z = z_coords[0]
            z_differences = [z - starting_z for z in z_coords]
        else:
            z_differences = []
            starting_z = 0.0
    
    print(f"Loaded {len(times)} data points")
    print(f"Time range: {min(times):.6e} to {max(times):.6e}")
    print(f"Starting z-coordinate: {starting_z:.6e}")
    print(f"Z-coordinate difference range: {min(z_differences):.6e} to {max(z_differences):.6e}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Check if there are multiple loads (different load values)
    unique_loads = sorted(set(loads))
    
    if len(unique_loads) > 1:
        # Plot each load with a different color
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_loads)))
        for load_val, color in zip(unique_loads, colors):
            mask = np.array(loads) == load_val
            plt.plot(np.array(times)[mask], np.array(z_differences)[mask], 
                    'o-', label=f'Load: {load_val:.6e}', color=color, markersize=3)
        plt.legend()
    else:
        # Single load, just plot the data
        plt.plot(times, z_differences, 'b-', linewidth=1.5, markersize=3)
        if unique_loads:
            plt.title(f'Time vs. Final Node Z-Coordinate Difference (Load: {unique_loads[0]:.6e})')
    
    plt.xlabel('Time [seconds]', fontsize=12)
    plt.ylabel('Z-Coordinate Difference [meters] (from initial)', fontsize=12)
    plt.title('Time vs. Final Node Z-Coordinate Difference', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add steady state line if requested
    if show_steady_state:
        # Use CSV file if available, otherwise use current data
        if csv_file and os.path.exists(csv_file):
            # CSV was already read, use it for steady state
            csv_times, csv_z_diffs, csv_loads = read_csv_data(csv_file)
            steady_state_data_times = csv_times
            steady_state_data_z_diffs = csv_z_diffs
        elif output_file:
            # Try to read CSV file
            csv_file = output_file.rsplit('.', 1)[0] + '_data.csv'
            csv_times, csv_z_diffs, csv_loads = read_csv_data(csv_file)
            if csv_times and csv_z_diffs:
                steady_state_data_times = csv_times
                steady_state_data_z_diffs = csv_z_diffs
            else:
                # Use current data
                steady_state_data_times = times
                steady_state_data_z_diffs = z_differences
        else:
            # Use current data
            steady_state_data_times = times
            steady_state_data_z_diffs = z_differences
        
        # Calculate steady state value
        steady_state_value = calculate_steady_state_from_csv(
            steady_state_data_times, steady_state_data_z_diffs, parameters_file
        )
        
        if steady_state_value is not None:
            # Add horizontal line for steady state
            xlim = plt.xlim()
            plt.axhline(y=steady_state_value, color='r', linestyle='--', 
                       linewidth=2, label=f'Steady State: {steady_state_value:.6e}')
            plt.xlim(xlim)  # Restore x limits
            plt.legend()
            print(f"Steady state value: {steady_state_value:.6e}")
    
    plt.tight_layout()
    
    # Save plotting data to CSV if requested
    if save_csv:
        if output_file:
            # Generate CSV filename from plot filename
            csv_file_to_save = output_file.rsplit('.', 1)[0] + '_data.csv'
        else:
            # Generate CSV filename from input filename
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            csv_file_to_save = os.path.join(os.path.dirname(input_file), 
                                           f'{base_name}_plot_data.csv')
        # Only save if we read from text file (not from CSV)
        if not (use_csv_if_available and csv_file and os.path.exists(csv_file)):
            save_plot_data_to_csv(times, z_differences, loads, csv_file_to_save)
    
    # Create snapshots if requested
    if create_snapshots:
        # Set up snapshot output directory
        if snapshot_output_dir is None:
            if output_file:
                snapshot_output_dir = os.path.join(os.path.dirname(output_file), 'snapshots')
            else:
                snapshot_output_dir = os.path.join(os.path.dirname(input_file), 'snapshots')
        
        print(f"\n{'='*60}")
        print("Creating 3D helix snapshots...")
        print(f"{'='*60}")
        
        # If we already extracted full data, use it; otherwise read from file
        if full_times and full_positions_list:
            print("Using previously extracted position data (no need to re-read file)")
            create_helix_snapshots_from_data(full_times, full_loads, full_positions_list, 
                                            input_file, snapshot_output_dir, num_snapshots)
        else:
            # Need to read from file
            create_helix_snapshots(input_file, snapshot_output_dir, num_snapshots, parameters_file)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def main():
    """Main function to handle command line arguments."""
    # Default input file - modify this to run the script directly without command line arguments
    DEFAULT_INPUT_FILE = 'HW4/plots/output_data_0.040m_20251124_151656_completed.txt'  # Set to None to require command line argument
    DEFAULT_OUTPUT_FILE = 'HW4/plots/output_data_0.040m_20251124_151656.png'  # Set to None to display interactively
    DEFAULT_SAVE_CSV = True  # Set to False to disable CSV saving
    
    if len(sys.argv) < 2:
        if DEFAULT_INPUT_FILE and os.path.exists(DEFAULT_INPUT_FILE):
            # Use default file if specified and exists
            input_file = DEFAULT_INPUT_FILE
            output_file = DEFAULT_OUTPUT_FILE
            save_csv = DEFAULT_SAVE_CSV
            print(f"Using default input file: {input_file}")
            if output_file:
                print(f"Output will be saved to: {output_file}")
        else:
            print("Usage: python output_plotter_pt1.py <input_file> [output_file] [--no-csv]")
            print("\nExample:")
            print("  python output_plotter_pt1.py HW4/plots/output_data_20251124_031619_completed.txt")
            print("  python output_plotter_pt1.py HW4/plots/output_data_20251124_031619_completed.txt plot.png")
            print("  python output_plotter_pt1.py HW4/plots/output_data_20251124_031619_completed.txt plot.png --no-csv")
            print("\nNote: By default, plotting data is saved to a CSV file for easier reference.")
            print("\nAlternatively, set DEFAULT_INPUT_FILE in the script to run without arguments.")
            sys.exit(1)
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
        save_csv = '--no-csv' not in sys.argv
    
    plot_time_vs_z_coordinate(input_file, output_file, save_csv=save_csv)


if __name__ == "__main__":
    main()

# Uncomment and modify the lines below to run directly without command line arguments:
input_file = 'HW4/plots/output_data_20251124_031619_completed.txt'
output_file = 'HW4/plots/time_vs_z_coordinate_difference.png'  # Set to None to display interactively
save_csv = True
plot_time_vs_z_coordinate(input_file, output_file, save_csv=save_csv)


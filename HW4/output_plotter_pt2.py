"""
Script to plot force vs. steady state z-coordinate difference.
Determines steady state values by averaging min/max in the steady_time period.
"""

import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Tuple, Optional


def read_parameters(parameters_file: str = 'HW4/springNetworkData/parameters.txt') -> Dict[str, float]:
    """Read parameters from the parameters file."""
    params = {}
    if not os.path.exists(parameters_file):
        print(f"Warning: Parameters file not found: {parameters_file}")
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


def calculate_steady_time(parameters_file: str = 'HW4/springNetworkData/parameters.txt') -> float:
    """Calculate steady_time from parameters (5 * time_scale)."""
    params = read_parameters(parameters_file)
    
    if 'L' not in params or 'E' not in params or 'rho' not in params:
        print("Error: Missing required parameters (L, E, rho)")
        sys.exit(1)
    
    L = params["L"]
    E = params["E"]
    rho = params["rho"]
    time_scale = L / np.sqrt(E / rho)
    steady_time = 5 * time_scale
    return steady_time


def extract_data_from_file(input_file: str) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract time, z-coordinate, and load data from the completed output file.
    
    Returns:
    --------
    times : List[float]
        List of time values
    z_coords : List[float]
        List of z-coordinates (last element of positions array)
    loads : List[float]
        List of load values
    """
    times = []
    z_coords = []
    loads = []
    
    print(f"Reading data from: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
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
    
    return times, z_coords, loads


def calculate_steady_state_values(times: List[float], z_coords: List[float], 
                                  loads: List[float], steady_time: float) -> Dict[float, float]:
    """
    Calculate steady state value for each load.
    
    For each load, finds the data in the last steady_time period and calculates
    the average of min and max values in that period.
    
    Parameters:
    -----------
    times : List[float]
        List of time values
    z_coords : List[float]
        List of z-coordinates
    loads : List[float]
        List of load values
    steady_time : float
        Time period required for steady state
    
    Returns:
    --------
    steady_state_values : Dict[float, float]
        Dictionary mapping load to steady state value
    """
    steady_state_values = {}
    unique_loads = sorted(set(loads))
    
    # Calculate starting z-coordinate (use first value)
    starting_z = z_coords[0] if z_coords else 0.0
    z_differences = [z - starting_z for z in z_coords]
    
    for load_val in unique_loads:
        # Find all indices for this load
        load_indices = [i for i, l in enumerate(loads) if abs(l - load_val) < 1e-12]
        
        if not load_indices:
            continue
        
        # Get times and z_differences for this load
        load_times = [times[i] for i in load_indices]
        load_z_diffs = [z_differences[i] for i in load_indices]
        
        if not load_times:
            continue
        
        # Find the maximum time for this load
        max_time = max(load_times)
        
        # Find data points in the last steady_time period
        time_threshold = max_time - steady_time
        steady_state_indices = [i for i in load_indices 
                               if times[i] >= time_threshold]
        
        if not steady_state_indices:
            # If no data in steady_time period, use all data
            steady_state_indices = load_indices
        
        # Get z_differences in steady state period
        steady_state_z_diffs = [z_differences[i] for i in steady_state_indices]
        
        if not steady_state_z_diffs:
            continue
        
        # Calculate average of min and max
        min_val = min(steady_state_z_diffs)
        max_val = max(steady_state_z_diffs)
        steady_state_value = (min_val + max_val) / 2.0
        
        steady_state_values[load_val] = steady_state_value
        
        print(f"Load: {load_val:.6e}, Steady state value: {steady_state_value:.6e} "
              f"(min: {min_val:.6e}, max: {max_val:.6e}, "
              f"period: {max_time - time_threshold:.6e} to {max_time:.6e})")
    
    return steady_state_values


def fit_line_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    """
    Fit a line through the origin (y = mx).
    
    Returns the slope m.
    """
    # For y = mx, we minimize sum((y - mx)^2)
    # Taking derivative and setting to zero: sum(2m*x^2 - 2*x*y) = 0
    # m = sum(x*y) / sum(x^2)
    if len(x) == 0 or np.sum(x**2) == 0:
        return 0.0
    
    slope = np.sum(x * y) / np.sum(x**2)
    return slope


def save_plot_data_to_csv(forces: List[float], steady_state_values: List[float],
                          slope: float, output_csv_file: str):
    """
    Save the plotting data to a CSV file.
    
    Parameters:
    -----------
    forces : List[float]
        List of force values
    steady_state_values : List[float]
        List of steady state values
    slope : float
        Slope of the fitted line
    output_csv_file : str
        Path to save the CSV file
    """
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Force', 'Steady_State_Value', 'Fitted_Value'])
        # Write data
        for force, ss_val in zip(forces, steady_state_values):
            fitted_val = slope * force
            writer.writerow([force, ss_val, fitted_val])
        # Write slope at the end
        writer.writerow([])
        writer.writerow(['Slope', slope])
    print(f"Plot data saved to: {output_csv_file}")


def plot_force_vs_steady_state(input_file: str, output_file: str = None,
                                parameters_file: str = 'HW4/springNetworkData/parameters.txt',
                                save_csv: bool = True):
    """
    Plot force vs. steady state z-coordinate difference.
    
    Parameters:
    -----------
    input_file : str
        Path to the completed output data file
    output_file : str, optional
        Path to save the plot (if None, displays interactively)
    parameters_file : str, optional
        Path to parameters file (default: HW4/springNetworkData/parameters.txt)
    save_csv : bool, optional
        Whether to save the plotting data to a CSV file (default: True)
    """
    # Calculate steady_time from parameters
    steady_time = calculate_steady_time(parameters_file)
    print(f"Steady time period: {steady_time:.6e}")
    
    # Extract data from file
    times, z_coords, loads = extract_data_from_file(input_file)
    
    if not times:
        print("Error: No valid data found in file")
        sys.exit(1)
    
    # Calculate steady state values for each load
    steady_state_values_dict = calculate_steady_state_values(
        times, z_coords, loads, steady_time
    )
    
    if not steady_state_values_dict:
        print("Error: No steady state values calculated")
        sys.exit(1)
    
    # Prepare data for plotting
    forces = sorted(steady_state_values_dict.keys())
    steady_state_values = [steady_state_values_dict[f] for f in forces]
    
    # Fit line through origin
    forces_array = np.array(forces)
    ss_values_array = np.array(steady_state_values)
    slope = fit_line_through_origin(forces_array, ss_values_array)
    
    print(f"\nFitted line slope: {slope:.6e}")
    print(f"Line equation: y = {slope:.6e} * x")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.plot(forces, steady_state_values, 'bo', markersize=8, label='Steady State Values')
    
    # Plot fitted line through origin
    force_range = np.linspace(0, max(forces), 100)
    fitted_values = slope * force_range
    plt.plot(force_range, fitted_values, 'r-', linewidth=2, 
             label=f'Best Fit Line (slope = {slope:.6e})')
    
    plt.xlabel('Force [N]', fontsize=12)
    plt.ylabel('Steady State Z-Coordinate Difference [meters]', fontsize=12)
    plt.title('Force vs. Steady State Z-Coordinate Difference', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plotting data to CSV if requested
    if save_csv:
        if output_file:
            # Generate CSV filename from plot filename
            csv_file = output_file.rsplit('.', 1)[0] + '_data.csv'
        else:
            # Generate CSV filename from input filename
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            csv_file = os.path.join(os.path.dirname(input_file), 
                                   f'{base_name}_force_vs_steady_state_data.csv')
        save_plot_data_to_csv(forces, steady_state_values, slope, csv_file)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def main():
    """Main function to handle command line arguments."""
    # Default values - modify these to run the script directly without command line arguments
    DEFAULT_INPUT_FILE = None  # Set to a file path to use as default, or None to require command line argument
    DEFAULT_OUTPUT_FILE = None  # Set to a file path to save plot, or None to display interactively
    DEFAULT_PARAMETERS_FILE = 'HW4/springNetworkData/parameters.txt'
    DEFAULT_SAVE_CSV = True  # Set to False to disable CSV saving
    
    if len(sys.argv) < 2:
        if DEFAULT_INPUT_FILE and os.path.exists(DEFAULT_INPUT_FILE):
            # Use default file if specified and exists
            input_file = DEFAULT_INPUT_FILE
            output_file = DEFAULT_OUTPUT_FILE
            parameters_file = DEFAULT_PARAMETERS_FILE
            save_csv = DEFAULT_SAVE_CSV
            print(f"Using default input file: {input_file}")
            if output_file:
                print(f"Output will be saved to: {output_file}")
        else:
            print("Usage: python output_plotter_pt2.py <input_file> [output_file] [--parameters PARAMS_FILE] [--no-csv]")
            print("\nExample:")
            print("  python output_plotter_pt2.py HW4/plots/output_data_20251123_013345_completed.txt")
            print("  python output_plotter_pt2.py HW4/plots/output_data_20251123_013345_completed.txt plot.png")
            print("  python output_plotter_pt2.py HW4/plots/output_data_20251123_013345_completed.txt plot.png --no-csv")
            print("\nNote: By default, plotting data is saved to a CSV file for easier reference.")
            print("\nAlternatively, set DEFAULT_INPUT_FILE in the script to run without arguments.")
            sys.exit(1)
    else:
        input_file = sys.argv[1]
        output_file = None
        parameters_file = DEFAULT_PARAMETERS_FILE
        save_csv = DEFAULT_SAVE_CSV
        
        # Parse arguments
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == '--parameters' and i + 1 < len(sys.argv):
                parameters_file = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--no-csv':
                save_csv = False
                i += 1
            elif not sys.argv[i].startswith('--'):
                output_file = sys.argv[i]
                i += 1
            else:
                i += 1
    
    plot_force_vs_steady_state(input_file, output_file, parameters_file, save_csv)


if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        # Run via CLI
        main()
    else:
        # Run directly from file - modify the lines below to set your file paths
        input_file = 'HW4/plots/output_data_20251123_013345_completed.txt'
        output_file = 'HW4/plots/force_vs_steady_state.png'  # Set to None to display interactively
        parameters_file = 'HW4/springNetworkData/parameters.txt'
        save_csv = True
        plot_force_vs_steady_state(input_file, output_file, parameters_file, save_csv)


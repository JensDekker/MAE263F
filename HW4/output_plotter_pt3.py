"""
Script to plot diameter sweep vs. textbook trend (Part 3).
Plots numerical stiffness k vs. textbook prediction G*d^4/(8*N*D^3).
"""

import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Tuple, Optional
from glob import glob


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


def extract_diameter_from_filename(filename: str) -> Optional[float]:
    """
    Extract diameter D from filename.
    Expected format: output_data_*_0.XXXm.txt where XXX is the diameter in meters.
    """
    # Pattern to match: _0.XXXm. or _0.XXXm.txt
    pattern = r'_([0-9]+\.[0-9]+)m\.txt'
    match = re.search(pattern, filename)
    
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    
    return None


def extract_data_from_file(input_file: str) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract time, z-coordinate, and load data from the output file.
    
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
    
    if not os.path.exists(input_file):
        print(f"Warning: File not found: {input_file}")
        return times, z_coords, loads
    
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


def calculate_stiffness_from_data(times: List[float], z_coords: List[float], 
                                  loads: List[float], 
                                  steady_time: float) -> Optional[float]:
    """
    Calculate stiffness k from force and displacement data.
    
    Stiffness is calculated as k = F / Δz, where F is the force (load)
    and Δz is the steady-state displacement.
    
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
    stiffness : float or None
        Calculated stiffness k = F / Δz
    """
    if not times or not z_coords or not loads:
        return None
    
    # Calculate starting z-coordinate (use first value)
    starting_z = z_coords[0] if z_coords else 0.0
    z_differences = [z - starting_z for z in z_coords]
    
    # Get unique loads (should typically be one for Part 3)
    unique_loads = sorted(set(loads))
    
    if not unique_loads:
        return None
    
    # For Part 3, we expect a single load value per file
    # Use the first (and likely only) load value
    load_val = unique_loads[0]
    
    # Find all indices for this load
    load_indices = [i for i, l in enumerate(loads) if abs(l - load_val) < 1e-12]
    
    if not load_indices:
        return None
    
    # Get times and z_differences for this load
    load_times = [times[i] for i in load_indices]
    load_z_diffs = [z_differences[i] for i in load_indices]
    
    if not load_times:
        return None
    
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
        return None
    
    # Calculate average of min and max for steady state displacement
    min_val = min(steady_state_z_diffs)
    max_val = max(steady_state_z_diffs)
    steady_state_displacement = (min_val + max_val) / 2.0
    
    # Calculate stiffness: k = F / Δz
    if abs(steady_state_displacement) < 1e-12:
        return None
    
    stiffness = abs(load_val) / abs(steady_state_displacement)
    
    return stiffness


def calculate_textbook_stiffness(G: float, d: float, N: float, D: float) -> float:
    """
    Calculate textbook stiffness prediction: k_text = G * d^4 / (8 * N * D^3)
    
    Parameters:
    -----------
    G : float
        Shear modulus
    d : float
        Wire diameter
    N : float
        Number of turns
    D : float
        Helix diameter
    
    Returns:
    --------
    k_text : float
        Textbook stiffness prediction
    """
    k_text = (G * d**4) / (8 * N * D**3)
    return k_text


def process_diameter_files(file_patterns: List[str], 
                           parameters_file: str = 'HW4/springNetworkData/parameters.txt',
                           steady_time: Optional[float] = None) -> Tuple[List[float], List[float], List[float]]:
    """
    Process multiple data files for different diameters and calculate stiffness values.
    
    Parameters:
    -----------
    file_patterns : List[str]
        List of file paths or glob patterns
    parameters_file : str
        Path to parameters file
    steady_time : float, optional
        Steady time period (calculated from parameters if not provided)
    
    Returns:
    --------
    diameters : List[float]
        List of helix diameters D
    stiffnesses : List[float]
        List of numerical stiffnesses k
    textbook_stiffnesses : List[float]
        List of textbook stiffness predictions
    """
    # Read parameters
    params = read_parameters(parameters_file)
    
    if 'G' not in params:
        # Calculate G from E and ν if not directly available
        if 'E' in params and 'v_poisson' in params:
            E = params['E']
            nu = params['v_poisson']
            G = E / (2 * (1 + nu))
        else:
            print("Error: Need G or (E and v_poisson) in parameters file")
            sys.exit(1)
    else:
        G = params['G']
    
    if 'wire_dia' not in params:
        print("Error: Need wire_dia (d) in parameters file")
        sys.exit(1)
    d = params['wire_dia']
    
    if 'N' not in params:
        print("Error: Need N (number of turns) in parameters file")
        sys.exit(1)
    N = params['N']
    
    # Calculate steady_time if not provided
    if steady_time is None:
        steady_time = calculate_steady_time(parameters_file)
    
    print(f"Using parameters: G={G:.6e}, d={d:.6e}, N={N:.0f}")
    print(f"Steady time period: {steady_time:.6e}")
    
    # Expand glob patterns and collect files
    all_files = []
    for pattern in file_patterns:
        if '*' in pattern or '?' in pattern:
            matched_files = glob(pattern)
            all_files.extend(matched_files)
        else:
            if os.path.exists(pattern):
                all_files.append(pattern)
    
    if not all_files:
        print("Error: No data files found")
        sys.exit(1)
    
    # Extract diameters and sort files by diameter
    file_diameter_pairs = []
    for filepath in all_files:
        D = extract_diameter_from_filename(filepath)
        if D is not None:
            file_diameter_pairs.append((filepath, D))
        else:
            print(f"Warning: Could not extract diameter from {filepath}, skipping")
    
    # Sort by diameter
    file_diameter_pairs.sort(key=lambda x: x[1])
    
    diameters = []
    stiffnesses = []
    textbook_stiffnesses = []
    
    print(f"\nProcessing {len(file_diameter_pairs)} files...")
    
    for filepath, D in file_diameter_pairs:
        print(f"\nProcessing D={D:.6e} m from {os.path.basename(filepath)}")
        
        # Extract data
        times, z_coords, loads = extract_data_from_file(filepath)
        
        if not times:
            print(f"  Warning: No data found, skipping")
            continue
        
        # Calculate numerical stiffness
        k = calculate_stiffness_from_data(times, z_coords, loads, steady_time)
        
        if k is None:
            print(f"  Warning: Could not calculate stiffness, skipping")
            continue
        
        # Calculate textbook stiffness
        k_text = calculate_textbook_stiffness(G, d, N, D)
        
        diameters.append(D)
        stiffnesses.append(k)
        textbook_stiffnesses.append(k_text)
        
        print(f"  k (numerical) = {k:.6e} N/m")
        print(f"  k_text (textbook) = {k_text:.6e} N/m")
        print(f"  Ratio k/k_text = {k/k_text:.6f}")
    
    return diameters, stiffnesses, textbook_stiffnesses


def save_plot_data_to_csv(diameters: List[float], stiffnesses: List[float],
                          textbook_stiffnesses: List[float], output_csv_file: str):
    """
    Save the plotting data to a CSV file.
    
    Parameters:
    -----------
    diameters : List[float]
        List of helix diameters
    stiffnesses : List[float]
        List of numerical stiffnesses
    textbook_stiffnesses : List[float]
        List of textbook stiffness predictions
    output_csv_file : str
        Path to save the CSV file
    """
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Diameter_D', 'Numerical_Stiffness_k', 'Textbook_Stiffness_k_text', 'Ratio_k_k_text'])
        # Write data
        for D, k, k_text in zip(diameters, stiffnesses, textbook_stiffnesses):
            writer.writerow([D, k, k_text, k/k_text])
    print(f"\nPlot data saved to: {output_csv_file}")


def plot_diameter_sweep(file_patterns: List[str], 
                        output_file: str = None,
                        parameters_file: str = 'HW4/springNetworkData/parameters.txt',
                        save_csv: bool = True):
    """
    Plot diameter sweep vs. textbook trend.
    
    Parameters:
    -----------
    file_patterns : List[str]
        List of file paths or glob patterns for data files
    output_file : str, optional
        Path to save the plot (if None, displays interactively)
    parameters_file : str, optional
        Path to parameters file (default: HW4/springNetworkData/parameters.txt)
    save_csv : bool, optional
        Whether to save the plotting data to a CSV file (default: True)
    """
    # Process all files
    diameters, stiffnesses, textbook_stiffnesses = process_diameter_files(
        file_patterns, parameters_file
    )
    
    if not diameters:
        print("Error: No valid data processed")
        sys.exit(1)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot data points: k (y-axis) vs k_text (x-axis)
    plt.plot(textbook_stiffnesses, stiffnesses, 'bo', markersize=10, 
             label='DER Simulation Results', zorder=3)
    
    # Plot reference line of slope 1 through origin (textbook relation)
    max_k_text = max(textbook_stiffnesses)
    min_k_text = min(textbook_stiffnesses)
    # Extend range slightly for better visualization
    # Start from a small positive value to avoid log(0) for log scale
    k_text_range = np.linspace(max(min_k_text * 0.1, 1e-6), max_k_text * 1.1, 100)
    reference_line = k_text_range  # y = x (slope 1 through origin)
    plt.plot(k_text_range, reference_line, 'r--', linewidth=2, 
             label='Textbook Relation (k = G d⁴/(8 N D³))', zorder=2)
    
    plt.xlabel(r'$G d^4 / (8 N D^3)$ [N/m]', fontsize=14)
    plt.ylabel(r'$k$ (Numerical Stiffness) [N/m]', fontsize=14)
    plt.title('Diameter Sweep: Numerical Stiffness vs. Textbook Prediction', fontsize=16)
    plt.yscale('log')  # Set y-axis to log scale
    plt.grid(True, alpha=0.3, which='both')  # Show both major and minor grid lines for log scale
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Add text annotation with parameter values
    params = read_parameters(parameters_file)
    if 'G' in params and 'wire_dia' in params and 'N' in params:
        param_text = f'G = {params["G"]:.2e} Pa\n'
        param_text += f'd = {params["wire_dia"]:.4f} m\n'
        param_text += f'N = {params["N"]:.0f}'
        plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save plotting data to CSV if requested
    if save_csv:
        if output_file:
            # Generate CSV filename from plot filename
            csv_file = output_file.rsplit('.', 1)[0] + '_data.csv'
        else:
            # Generate CSV filename in plots directory
            csv_file = 'HW4/plots/diameter_sweep_data.csv'
        save_plot_data_to_csv(diameters, stiffnesses, textbook_stiffnesses, csv_file)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()


def main():
    """Main function to handle command line arguments."""
    # Default values - modify these to run the script directly without command line arguments
    DEFAULT_FILE_PATTERNS = [
        'HW4/plots/output_data_*_0.010m.txt',
        'HW4/plots/output_data_*_0.013m.txt',
        'HW4/plots/output_data_*_0.017m.txt',
        'HW4/plots/output_data_*_0.020m.txt',
        'HW4/plots/output_data_*_0.025m.txt',
        'HW4/plots/output_data_*_0.030m.txt',
        'HW4/plots/output_data_*_0.035m.txt',
        'HW4/plots/output_data_*_0.040m.txt',
        'HW4/plots/output_data_*_0.045m.txt',
        'HW4/plots/output_data_*_0.050m.txt',
    ]
    DEFAULT_OUTPUT_FILE = 'HW4/plots/diameter_sweep_vs_textbook.png'
    DEFAULT_PARAMETERS_FILE = 'HW4/springNetworkData/parameters.txt'
    DEFAULT_SAVE_CSV = True
    
    if len(sys.argv) < 2:
        # Use default file patterns
        file_patterns = DEFAULT_FILE_PATTERNS
        output_file = DEFAULT_OUTPUT_FILE
        parameters_file = DEFAULT_PARAMETERS_FILE
        save_csv = DEFAULT_SAVE_CSV
        print("Using default file patterns")
        if output_file:
            print(f"Output will be saved to: {output_file}")
    else:
        # Parse command line arguments
        file_patterns = []
        output_file = None
        parameters_file = DEFAULT_PARAMETERS_FILE
        save_csv = DEFAULT_SAVE_CSV
        
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == '--output' and i + 1 < len(sys.argv):
                output_file = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--parameters' and i + 1 < len(sys.argv):
                parameters_file = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--no-csv':
                save_csv = False
                i += 1
            elif not sys.argv[i].startswith('--'):
                file_patterns.append(sys.argv[i])
                i += 1
            else:
                i += 1
        
        if not file_patterns:
            print("Usage: python output_plotter_pt3.py [file_patterns...] [--output OUTPUT_FILE] [--parameters PARAMS_FILE] [--no-csv]")
            print("\nExample:")
            print("  python output_plotter_pt3.py HW4/plots/output_data_*_0.010m.txt HW4/plots/output_data_*_0.020m.txt")
            print("  python output_plotter_pt3.py --output plot.png")
            print("\nNote: By default, uses all diameter files in HW4/plots/")
            print("\nAlternatively, set DEFAULT_FILE_PATTERNS in the script to run without arguments.")
            sys.exit(1)
    
    plot_diameter_sweep(file_patterns, output_file, parameters_file, save_csv)


if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        # Run via CLI
        main()
    else:
        # Run directly from file - modify the lines below to set your file paths
        file_patterns = [
            'HW4/plots/output_data_20251124_222359_0.010m.txt',
            'HW4/plots/output_data_20251124_222933_0.013m.txt',
            'HW4/plots/output_data_20251124_223154_0.017m.txt',
            'HW4/plots/output_data_20251124_224237_0.020m.txt',
            'HW4/plots/output_data_20251124_224601_0.025m.txt',
            'HW4/plots/output_data_20251124_225450_0.030m.txt',
            'HW4/plots/output_data_20251124_231009_0.035m.txt',
            'HW4/plots/output_data_20251124_231346_0.040m.txt',
            'HW4/plots/output_data_20251124_231644_0.045m.txt',
            'HW4/plots/output_data_20251124_231859_0.050m.txt',
        ]
        output_file = 'HW4/plots/diameter_sweep_vs_textbook.png'
        parameters_file = 'HW4/springNetworkData/parameters.txt'
        save_csv = True
        plot_diameter_sweep(file_patterns, output_file, parameters_file, save_csv)


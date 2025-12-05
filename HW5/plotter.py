import numpy as np
import matplotlib.pyplot as plt
import csv
import re
import sys

# Increase CSV field size limit to handle large position arrays
# Use a large but safe value that works across different systems
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # On some systems, sys.maxsize is too large; use a large but safe value
    csv.field_size_limit(2147483647)  # 2^31 - 1, maximum 32-bit signed integer

def parse_positions_array(pos_str: str) -> np.ndarray:
    """
    Parse the positions array string into a numpy array.
    Uses regex to extract all numbers, similar to output_plotter_pt1.py.
    
    Parameters:
    -----------
    pos_str : str
        String representation of positions array like "[ x0 y0 z0 x1 y1 z1 ... ]"
        Can span multiple lines
    
    Returns:
    --------
    positions : np.ndarray
        Array of positions (DOF vector)
    """
    if not pos_str:
        return np.array([])
    
    # Use regex to find all numbers (handles multi-line arrays with various whitespace)
    # This pattern matches: optional sign, digits, optional decimal, optional scientific notation
    values = re.findall(r'[-+]?\d+\.?\d*[eE]?[-+]?\d*', pos_str)
    if not values:
        return np.array([])
    
    try:
        positions = np.array([float(v) for v in values])
        return positions
    except (ValueError, AttributeError, TypeError):
        return np.array([])


time_lst = []
last_four_values_lst = []  # Store the last 4 values from positions array

with open('HW5/plots/output_data_0s-5s.txt', 'r', encoding='utf-8') as f:
    csv_reader = csv.DictReader(f)
    
    for row in csv_reader:
        # Use dictionary access for more robust parsing
        # This ensures we get the correct column even if CSV parsing has issues
        time_str = row.get('Time')
        positions_str = row.get('Positions')
        
        # Skip rows with missing data
        if not time_str or not positions_str:
            continue
        
        try:
            # Extract time
            time = float(time_str)
            
            # Extract ONLY positions string
            # NOTE: row['Velocities'] contains Velocities - we explicitly ignore it
            # Using dictionary access prevents position/velocity mixing
            
            # Parse the positions array using regex (more robust than string splitting)
            positions_array = parse_positions_array(positions_str)
            
            if len(positions_array) < 4:
                # Skip if we don't have at least 4 values
                continue
            
            # Extract the last 4 values from the positions array
            last_four_values = positions_array[-4:]
            
            time_lst.append(time)
            last_four_values_lst.append(last_four_values)
        except (ValueError, IndexError) as e:
            # Skip rows that can't be parsed
            print(f"Warning: Skipping row at time {time_str} due to parse error: {e}")
            continue


# isolate the last and 4th last values for each time step
last_and_fourth_last_values_lst = []
for i in range(len(last_four_values_lst)):
    last_and_fourth_last_values_lst.append([last_four_values_lst[i][-1], last_four_values_lst[i][-4]])

avg_end_height_lst = []
for i in range(len(last_and_fourth_last_values_lst)):
    avg_end_height_lst.append(np.mean(last_and_fourth_last_values_lst[i]))

parameters_filepath = 'HW5/springNetworkData/parameters.txt'

# Read the parameters file
with open(parameters_filepath, 'r') as f:
    for line in f:
        parts = [part.strip() for part in line.split(':')]
        if parts[0] == 'nL_edge':
            nL_edge = int(parts[1])
        elif parts[0] == 'nW_edge':
            nW_edge = int(parts[1])
        elif parts[0] == 'length':
            length = float(parts[1])
        elif parts[0] == 'width':
            width = float(parts[1])
        elif parts[0] == 'thickness':
            thickness = float(parts[1])
        elif parts[0] == 'E':
            E = float(parts[1])
        elif parts[0] == 'I':
            I = float(parts[1])
        elif parts[0] == 'total_mass':
            total_mass = float(parts[1])
        elif parts[0] == 'N':
            N = int(parts[1])
        elif parts[0] == 'q':
            q = float(parts[1])

print("Parameters successfully loaded.")

expected_deflection = (q * length**4) / (8 * E * I)

# plot the first 5 seconds 
plt.plot(time_lst[:25000], avg_end_height_lst[:25000])
plt.xlabel('Time (s)')
plt.ylabel('Average End Height (m)')
plt.title('Average End Height vs Time')

# plot the expected deflection as a horizontal line
plt.axhline(y=expected_deflection, color='r', linestyle='--', label='Expected Deflection')
plt.legend()

# Save the plot
plt.savefig('HW5/plots/average_end_height_vs_time.png')


print(f"Expected deflection: {expected_deflection} m")

print(f"Actual deflection: {avg_end_height_lst[-1]} m")
print(f"Error: {abs(avg_end_height_lst[-1] - expected_deflection) / expected_deflection * 100}%")
# print the difference between the actual deflection and the expected deflection
print(f"Difference: {abs(avg_end_height_lst[-1] - expected_deflection)} m")

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

with open('HW5/plots/output_data.txt', 'r', encoding='utf-8') as f:
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


# plot the first 5 seconds 
plt.plot(time_lst[:], avg_end_height_lst[:])
plt.xlabel('Time (s)')
plt.ylabel('Average End Height (m)')
plt.title('Average End Height vs Time')
plt.show()


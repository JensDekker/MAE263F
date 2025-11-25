import numpy as np
import re

# Read the output file
filepath = 'HW4/plots/output_data_20251122_180052.txt'

last_elements = []
times = []

# Read the file line by line and parse manually
with open(filepath, 'r') as f:
    lines = f.readlines()
    
    # Skip header
    i = 1
    while i < len(lines):
        if i % 10000 == 0:
            print(f"Processing around line {i}...")
        
        # Find the start of a new row (starts with a number or Load value)
        line = lines[i].strip()
        if not line or line.startswith('Load'):
            i += 1
            continue
        
        # Parse the row - format: Load,Time,"[positions array...]",...
        # The positions array spans multiple lines
        try:
            # Extract Load and Time from first line
            parts = line.split(',', 2)
            if len(parts) < 3:
                i += 1
                continue
            
            load_val = parts[0]
            time_val = parts[1]
            positions_start = parts[2]
            
            # Find where the positions array ends (closing bracket and quote)
            positions_str = positions_start
            j = i + 1
            while j < len(lines) and ']"' not in positions_str:
                positions_str += ' ' + lines[j].strip()
                j += 1
            
            # Extract the array part
            match = re.search(r'\[(.*?)\]', positions_str)
            if match:
                array_content = match.group(1)
                # Parse the array values
                values = re.findall(r'[-+]?\d*\.?\d+[eE]?[-+]?\d*', array_content)
                if values:
                    positions = np.array([float(v) for v in values])
                    last_element = positions[-1]
                    last_elements.append(last_element)
                    times.append(float(time_val))
            
            # Move to next row (skip to after the positions array)
            i = j + 1
            # Skip empty lines
            while i < len(lines) and not lines[i].strip():
                i += 1
                
        except Exception as e:
            print(f"Error parsing around line {i}: {e}")
            i += 1
            continue

# Convert to numpy array for analysis
last_elements = np.array(last_elements)
times = np.array(times)

# Print statistics
print(f"\n{'='*60}")
print(f"Total time steps: {len(last_elements)}")
print(f"Time range: {times[0]:.6e} to {times[-1]:.6e}")
print(f"\nLast element statistics:")
print(f"  Min value: {np.min(last_elements):.10e}")
print(f"  Max value: {np.max(last_elements):.10e}")
print(f"  Mean value: {np.mean(last_elements):.10e}")
print(f"  Std deviation: {np.std(last_elements):.10e}")
print(f"  Range (max - min): {np.max(last_elements) - np.min(last_elements):.10e}")

# Check if all values are the same
if np.allclose(last_elements, last_elements[0]):
    print(f"\n⚠️  WARNING: All last elements are essentially the same!")
    print(f"   All values are approximately: {last_elements[0]:.10e}")
else:
    print(f"\n✓ Last elements DO change across time steps")
    print(f"  First value: {last_elements[0]:.10e}")
    print(f"  Last value: {last_elements[-1]:.10e}")
    print(f"  Difference: {abs(last_elements[-1] - last_elements[0]):.10e}")

# Show first 10 and last 10 values
print(f"\n{'='*60}")
print("First 10 last elements:")
for i in range(min(10, len(last_elements))):
    print(f"  Step {i}: {last_elements[i]:.10e} (t={times[i]:.6e})")

print(f"\nLast 10 last elements:")
for i in range(max(0, len(last_elements)-10), len(last_elements)):
    print(f"  Step {i}: {last_elements[i]:.10e} (t={times[i]:.6e})")


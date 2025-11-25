import re

filepath = 'HW4/plots/output_data_20251122_183546.txt'

# Extract last element from each positions array
last_elements = []
times = []

with open(filepath, 'r') as f:
    content = f.read()
    
    # Find all positions arrays - they end with ]" or ]",1
    # Pattern: find the last number before ]"
    pattern = r'([-+]?\d+\.?\d*[eE]?[-+]?\d*)\]\",'
    matches = re.findall(pattern, content)
    
    print(f"Found {len(matches)} positions arrays")
    
    # Also extract times
    time_pattern = r'([\d\.eE\+\-]+),([\d\.eE\+\-]+),\"\['
    time_matches = re.findall(time_pattern, content)
    
    print(f"Found {len(time_matches)} time entries")
    
    # Convert to floats
    last_elements = [float(m) for m in matches]
    times = [float(t[1]) for t in time_matches[:len(last_elements)]]
    
    if len(times) > len(last_elements):
        times = times[:len(last_elements)]
    elif len(last_elements) > len(times):
        last_elements = last_elements[:len(times)]

# Print statistics
print(f"\n{'='*60}")
print(f"Total time steps analyzed: {len(last_elements)}")
if len(times) > 0:
    print(f"Time range: {times[0]:.6e} to {times[-1]:.6e}")
print(f"\nLast element statistics:")
print(f"  Min value: {min(last_elements):.10e}")
print(f"  Max value: {max(last_elements):.10e}")
print(f"  Mean value: {sum(last_elements)/len(last_elements):.10e}")
print(f"  Range (max - min): {max(last_elements) - min(last_elements):.10e}")

# Check if all values are the same
if all(abs(x - last_elements[0]) < 1e-10 for x in last_elements):
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
    time_str = f" (t={times[i]:.6e})" if i < len(times) else ""
    print(f"  Step {i}: {last_elements[i]:.10e}{time_str}")

print(f"\nLast 10 last elements:")
for i in range(max(0, len(last_elements)-10), len(last_elements)):
    time_str = f" (t={times[i]:.6e})" if i < len(times) else ""
    print(f"  Step {i}: {last_elements[i]:.10e}{time_str}")


import imageio
import os
import glob
import re

# Get all plot images from the plots directory
plot_dir = "HW5/plots"
output_path = "HW5/plots/HW5_1.mp4"

# Find all plot_t_*.png files
frame_files = glob.glob(f"{plot_dir}/plot1_t_*.png")

# Sort by extracting the numeric time value from filename (e.g., "plot1_t_10.50.png" -> 10.50)
# This ensures proper numeric ordering instead of lexicographic ordering
def extract_time(filename):
    # Use basename to avoid issues with path separators
    basename = os.path.basename(filename)
    match = re.search(r'plot\d*_t_([\d.]+)\.png', basename)
    if match:
        return float(match.group(1))
    else:
        print(f"Warning: Could not extract time from {basename}")
        return 0.0

# Verify sorting is working - show first few times before and after
if frame_files:
    print(f"Found {len(frame_files)} frames to combine")
    print("\nFirst 5 files before sorting:")
    for f in frame_files[:5]:
        print(f"  {extract_time(f):6.2f} - {os.path.basename(f)}")
    
    frame_files = sorted(frame_files, key=extract_time)
    
    print("\nFirst 5 files after sorting:")
    for f in frame_files[:5]:
        print(f"  {extract_time(f):6.2f} - {os.path.basename(f)}")
    
    # Verify no ordering issues
    times = [extract_time(f) for f in frame_files]
    for i in range(len(times) - 1):
        if times[i] > times[i+1]:
            print(f"\nERROR: Sorting failed! {os.path.basename(frame_files[i])} ({times[i]:.2f}) > {os.path.basename(frame_files[i+1])} ({times[i+1]:.2f})")
            break
    else:
        print("\n✓ All frames are correctly sorted by time")
        print()
else:
    print(f"No plot images found in {plot_dir}")
    exit(1)

# Create MP4 video using get_writer
# fps parameter controls frames per second (playback speed)
print(f"Creating video with {len(frame_files)} frames at {40} fps...")
with imageio.get_writer(output_path, fps=40) as writer:
    for i, frame_file in enumerate(frame_files):
        frame = imageio.imread(frame_file)
        writer.append_data(frame)
        # Print progress every 100 frames to avoid excessive output
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Progress: {i+1}/{len(frame_files)} frames added...")

print(f"Video saved to: {output_path}")

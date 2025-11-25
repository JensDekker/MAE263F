import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read the CSV file (relative to script location)
csv_file = os.path.join(script_dir, 'plots', 'time_vs_z_coordinate_difference_data.csv')
df = pd.read_csv(csv_file)

# Filter data for the first second (Time <= 1.0)
df_first_second = df[df['Time'] <= 0.05]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df_first_second['Time'], df_first_second['Z_Coordinate_Difference'], 
         linewidth=1.5, color='blue')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Z Coordinate Difference', fontsize=12)
plt.title('Z Coordinate Difference vs Time (First Second)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot (relative to script location)
output_file = os.path.join(script_dir, 'plots', 'time_vs_z_coordinate_difference_first_second.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")

# Display the plot
plt.show()


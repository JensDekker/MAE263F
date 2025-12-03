import format_number as fn
import numpy as np

def steady_state_checker(x_new, x_new_idcs, h_container, allowable_error, steady_time, dt):
    
    data_length = steady_time / dt
    # Force the data length to be an integer
    data_length = int(data_length)
    
    # Calculate the average based on the last two nodes z-coordinates
    avg_height = np.mean(x_new[x_new_idcs])
    
    h_container.append(avg_height)
    # Check if the container has enough data to check for steady state
    if len(h_container) < data_length:
        # Debug: show progress filling the container
        if len(h_container) % max(1, data_length //5) == 0:  # Print every 5% progress
            print(f"Filling steady state container: {len(h_container)}/{data_length} points", flush=True)
        return False
    else:
        h_container.pop(0)

    # Check difference between the maximum and minimum height in the container
    max_height = max(h_container)
    min_height = min(h_container)
    height_difference = max_height - min_height

    steady_state_flag = False
    if height_difference < allowable_error:
        steady_state_flag = True
    
    height_difference_ratio = (height_difference / allowable_error) - 1
    print("Height difference ratio: " + fn.format_number(height_difference_ratio, decimals=3))

    return steady_state_flag

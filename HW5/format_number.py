def format_number(value, decimals=2, sci_threshold_large=1e6, sci_threshold_small=1e-3):
    """Format with fixed decimals, auto scientific for large/small numbers
    
    Parameters:
    - value: The number to format
    - decimals: Number of decimal places (default: 2)
    - sci_threshold_large: Use scientific notation if abs(value) >= this (default: 1e6)
    - sci_threshold_small: Use scientific notation if 0 < abs(value) < this (default: 1e-3)
    
    Examples:
        format_number(123.456, 3) -> "123.456" (normal range)
        format_number(1234567.89, 3) -> "1.235e+06" (>= 1e6)
        format_number(0.000123, 3) -> "1.230e-04" (< 1e-3)
    """
    # Handle zero explicitly
    if value == 0:
        return f"{0:.{decimals}f}"
    
    # Use scientific notation for very large or very small numbers
    if abs(value) >= sci_threshold_large or abs(value) < sci_threshold_small:
        return f"{value:.{decimals}e}"
    else:
        return f"{value:.{decimals}f}"


def format_error_with_iteration(error, iteration, decimals=3, error_width=12, 
                                padding=8, sci_threshold_small=5e-4):
    """Format error value with proper alignment for iteration number
    
    Parameters:
    - error: The error value to format
    - iteration: The iteration number
    - decimals: Number of decimal places for error (default: 3)
    - error_width: Total width for the formatted error string (default: 12)
    - padding: Number of spaces between error and "Iteration:" (default: 8)
    - sci_threshold_small: Threshold for scientific notation (default: 5e-4)
    
    Returns:
    - Formatted string: "Error: [formatted_error][padding]Iteration: [iteration]"
        where formatted_error is right-aligned to error_width
    
    Example:
        format_error_with_iteration(0.002, 19) -> "Error:        0.002        Iteration: 19"
        format_error_with_iteration(3.714e-04, 19) -> "Error:    3.714e-04        Iteration: 19"
    """
    formatted_error = format_number(error, decimals=decimals, 
                                    sci_threshold_small=sci_threshold_small)
    # Right-align the formatted error to the specified width
    padded_error = f"{formatted_error:>{error_width}}"
    # Add padding spaces between error and "Iteration:"
    return f"Error: {padded_error}{' ' * padding}Iteration: {iteration}"


# Test the function
if __name__ == "__main__":
    test_values = [123.456, 1234567.89, 0.000123, 0.5, 1000, 0.001, 0.0001]
    print("Testing format_number with decimals=3:")
    for val in test_values:
        print(f"  {val:15} -> {format_number(val, decimals=3)}")
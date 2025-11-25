"""
Progress bars for showing value approaching a target (e.g., convergence)
"""

# ============================================================================
# OPTION 1: tqdm with custom postfix (shows current value vs target)
# ============================================================================
from tqdm import tqdm
import numpy as np
import time

def example1_tqdm_value_target():
    """Show current value approaching target using tqdm"""
    print("Example 1: tqdm with value-to-target display")
    
    target = 1e-6  # Target tolerance
    current_value = 1.0  # Starting value
    
    # Create progress bar
    pbar = tqdm(desc="Converging", unit="iter")
    
    iteration = 0
    while current_value > target and iteration < 100:
        # Simulate convergence (exponential decay)
        current_value *= 0.9
        iteration += 1
        
        # Update progress bar with current value and target
        pbar.set_postfix({
            'error': f"{current_value:.2e}",
            'target': f"{target:.2e}",
            'ratio': f"{current_value/target:.2f}x"
        })
        pbar.update(1)
        time.sleep(0.05)
    
    pbar.close()
    print(f"Converged! Final value: {current_value:.2e}, Target: {target:.2e}\n")

# ============================================================================
# OPTION 2: rich Progress with custom columns (Best for value-to-target)
# ============================================================================
try:
    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
    from rich.console import Console
    from rich.table import Column
    
    def example2_rich_value_target():
        """Rich progress bar showing value approaching target"""
        print("Example 2: Rich progress bar with value-to-target")
        
        console = Console()
        target = 1e-6
        current_value = 1.0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("Error: {task.fields[error]}"),
            TextColumn("Target: {task.fields[target]}"),
            TextColumn("Ratio: {task.fields[ratio]}"),
            console=console
        ) as progress:
            task = progress.add_task(
                "Converging...",
                total=100,
                error="1.00e+00",
                target=f"{target:.2e}",
                ratio="1.00e+06"
            )
            
            iteration = 0
            while current_value > target and iteration < 100:
                current_value *= 0.9
                iteration += 1
                
                # Calculate progress as percentage of distance covered
                # (log scale for better visualization)
                if current_value > 0:
                    log_current = np.log10(current_value)
                    log_target = np.log10(target)
                    log_start = np.log10(1.0)
                    progress_pct = (log_start - log_current) / (log_start - log_target) * 100
                    progress.update(task, completed=min(progress_pct, 100))
                    
                    progress.update(task, fields={
                        'error': f"{current_value:.2e}",
                        'target': f"{target:.2e}",
                        'ratio': f"{current_value/target:.2f}x"
                    })
                
                time.sleep(0.05)
        
        print(f"Converged! Final value: {current_value:.2e}\n")

except ImportError:
    print("Rich not installed. Install with: pip install rich\n")

# ============================================================================
# OPTION 3: Custom progress bar for convergence (Newton-Raphson style)
# ============================================================================

def example3_custom_convergence_bar():
    """Custom progress bar showing error approaching tolerance"""
    print("Example 3: Custom convergence progress bar")
    
    tolerance = 1e-6
    error = 1.0
    bar_length = 50
    
    iteration = 0
    while error > tolerance and iteration < 100:
        error *= 0.9  # Simulate convergence
        iteration += 1
        
        # Calculate progress on log scale (better for orders of magnitude)
        if error > 0 and tolerance > 0:
            log_error = np.log10(error)
            log_tol = np.log10(tolerance)
            log_start = np.log10(1.0)
            
            # Progress from start to target (0 to 1)
            if log_start != log_tol:
                progress = (log_start - log_error) / (log_start - log_tol)
                progress = max(0, min(1, progress))  # Clamp to [0, 1]
            else:
                progress = 1.0
            
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f'\r[{bar}] Error: {error:.2e} → Target: {tolerance:.2e} '
                  f'({progress*100:.1f}%) | Iter: {iteration}', 
                  end='', flush=True)
        
        time.sleep(0.05)
    
    print(f"\nConverged in {iteration} iterations!\n")

# ============================================================================
# RECOMMENDED: For Newton-Raphson convergence (your use case)
# ============================================================================

def recommended_for_newton_raphson():
    """
    Recommended approach for showing Newton-Raphson convergence
    Shows error approaching tolerance
    """
    from tqdm import tqdm
    
    tolerance = 1e-6
    error = 1.0
    max_iter = 50
    
    # Create progress bar
    pbar = tqdm(total=max_iter, desc="Newton-Raphson", unit="iter")
    
    for iteration in range(max_iter):
        # Simulate Newton-Raphson iteration
        error *= 0.85  # Your actual error update
        
        # Update progress bar
        pbar.set_postfix({
            'error': f"{error:.2e}",
            'tol': f"{tolerance:.2e}",
            'ratio': f"{error/tolerance:.2f}x"
        })
        pbar.update(1)
        
        if error < tolerance:
            pbar.set_postfix({'status': 'CONVERGED!'})
            break
        
        time.sleep(0.01)
    
    pbar.close()
    
    if error >= tolerance:
        print("Did not converge")
    else:
        print(f"Converged in {iteration+1} iterations")

# ============================================================================
# OPTION 4: tqdm with custom format showing distance to target
# ============================================================================

def example4_tqdm_distance_to_target():
    """Show distance remaining to target"""
    print("Example 4: tqdm showing distance to target")
    
    target = 100.0
    current = 0.0
    step = 2.5
    
    pbar = tqdm(total=target, desc="Approaching target", unit="value")
    
    while current < target:
        current += step
        remaining = target - current
        
        pbar.set_postfix({
            'current': f"{current:.2f}",
            'target': f"{target:.2f}",
            'remaining': f"{remaining:.2f}"
        })
        pbar.update(step)
        time.sleep(0.05)
    
    pbar.close()
    print(f"Reached target! Value: {current:.2f}\n")

if __name__ == "__main__":
    print("="*60)
    print("PROGRESS BARS FOR VALUE-TO-TARGET")
    print("="*60 + "\n")
    
    # Uncomment to test:
    # example1_tqdm_value_target()
    # example2_rich_value_target()
    # example3_custom_convergence_bar()
    # example4_tqdm_distance_to_target()
    # recommended_for_newton_raphson()
    
    print("\nFor Newton-Raphson convergence, use recommended_for_newton_raphson()")
    print("It shows error approaching tolerance with ratio and status.")



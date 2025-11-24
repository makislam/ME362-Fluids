import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
from SpinAoAFrisbeeModel import simulate_3d_throw
import time
import sys
import io
from contextlib import redirect_stdout
import os

# ==========================================
#   OPTIMIZATION CONFIGURATION
# ==========================================

# Fixed parameters
FIXED_VELOCITY = 25.0  # m/s (keep constant)
FIXED_RELEASE_HEIGHT = 1.0  # m (keep constant)

# Air conditions
AIR_TEMP_C = 25.0
AIR_PRESSURE_PA = 101325

# Parameter bounds for optimization
# Format: (min, max)
BOUNDS = {
    'spin_rpm': (400, 1000),           # Spin rate (RPM)
    'launch_angle': (10, 45),            # Launch angle (degrees)
    'release_angle': (-20, 20),         # Release angle (degrees)
    'tilt_angle': (-45, 45),             # Tilt angle (degrees)
    'aoa': (-4, 20)                      # Angle of attack (degrees)
}

# ==========================================
#   OPTIMIZATION FUNCTIONS
# ==========================================

def objective_function(params):
    """
    Objective function to minimize (negative distance for maximization)
    
    Parameters:
        params: [spin_rpm, launch_angle, release_angle, tilt_angle, aoa]
    
    Returns:
        Negative distance (for minimization)
    """
    spin_rpm, launch_angle, release_angle, tilt_angle, aoa = params
    
    try:
        # Run simulation with suppressed output
        with redirect_stdout(io.StringIO()):
            sol = simulate_3d_throw(
                FIXED_VELOCITY, 
                spin_rpm, 
                FIXED_RELEASE_HEIGHT,
                launch_angle, 
                release_angle, 
                tilt_angle, 
                aoa,
                AIR_TEMP_C, 
                AIR_PRESSURE_PA
            )
        
        # Extract distance
        distance = sol.y[0, -1]
        
        # Return negative distance (we want to maximize, but optimizer minimizes)
        return -distance
        
    except Exception as e:
        # If simulation fails, return a large penalty
        return 1e6

def run_optimization(method='differential_evolution', initial_guess=None):
    """
    Run optimization to find best throw parameters
    
    Parameters:
        method: 'differential_evolution' (global) or 'nelder-mead' (local)
        initial_guess: Starting point for local optimization
    
    Returns:
        Optimization result
    """
    
    bounds_list = [
        BOUNDS['spin_rpm'],
        BOUNDS['launch_angle'],
        BOUNDS['release_angle'],
        BOUNDS['tilt_angle'],
        BOUNDS['aoa']
    ]
    
    print("=" * 80)
    print(f"OPTIMIZING THROW DISTANCE (V0 = {FIXED_VELOCITY} m/s, Height = {FIXED_RELEASE_HEIGHT} m)")
    print("=" * 80)
    print(f"\nOptimization method: {method}")
    print(f"\nParameter bounds:")
    print(f"  Spin:          {BOUNDS['spin_rpm'][0]:.0f} - {BOUNDS['spin_rpm'][1]:.0f} RPM")
    print(f"  Launch Angle:  {BOUNDS['launch_angle'][0]:.1f} - {BOUNDS['launch_angle'][1]:.1f} deg")
    print(f"  Release Angle: {BOUNDS['release_angle'][0]:.1f} - {BOUNDS['release_angle'][1]:.1f} deg")
    print(f"  Tilt Angle:    {BOUNDS['tilt_angle'][0]:.1f} - {BOUNDS['tilt_angle'][1]:.1f} deg")
    print(f"  Angle of Attack:{BOUNDS['aoa'][0]:.1f} - {BOUNDS['aoa'][1]:.1f} deg")
    print()
    
    if method == 'differential_evolution':
        # Global optimization - optimized for speed
        print(">>> OPTIMIZATION STARTING - Please wait...")
        
        # Calculate optimal number of workers (4-6 is best for most systems)
        total_cores = os.cpu_count() or 4
        num_workers = min(6, max(1, total_cores - 2))  # Use max 6 cores, leave 2 free
        print(f">>> Using parallel processing with {num_workers} CPU cores\n")
        
        # Progress tracking
        start_time = time.time()
        best_distance = [0]
        iteration_count = [0]
        max_iterations = 50  # Reduced from 100
        popsize = 10  # Reduced from 15
        total_evals = max_iterations * popsize * len(bounds_list)
        
        def callback(xk, convergence):
            iteration_count[0] += 1
            # Don't call objective_function here - it's already been called by the optimizer
            # Just track the best result from convergence
            
            # Progress bar with ASCII characters
            progress = iteration_count[0] / max_iterations
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '#' * filled + '-' * (bar_length - filled)
            elapsed = time.time() - start_time
            
            # Estimate time remaining
            if progress > 0:
                eta = (elapsed / progress) * (1 - progress)
                sys.stdout.write(f'\r[{bar}] {progress*100:.1f}% | Iter {iteration_count[0]}/{max_iterations} | '
                               f'Time: {elapsed:.1f}s | ETA: {eta:.1f}s    ')
            else:
                sys.stdout.write(f'\r[{bar}] {progress*100:.1f}% | Iter {iteration_count[0]}/{max_iterations} | '
                               f'Time: {elapsed:.1f}s    ')
            sys.stdout.flush()
            return False
        
        result = differential_evolution(
            objective_function,
            bounds_list,
            strategy='best1bin',
            maxiter=max_iterations,
            popsize=popsize,
            tol=0.01,
            atol=0.1,  # Absolute tolerance for faster convergence
            seed=42,
            disp=False,
            workers=num_workers,
            callback=callback,
            polish=True,
            updating='deferred'
        )
        print()  # New line after progress bar
    else:
        # Local optimization - faster but may find local optimum
        if initial_guess is None:
            # Use middle of bounds as initial guess
            initial_guess = [(b[0] + b[1]) / 2 for b in bounds_list]
        
        print("Running local optimization...\n")
        start_time = time.time()
        
        result = minimize(
            objective_function,
            initial_guess,
            method='Nelder-Mead',
            options={'disp': False, 'maxiter': 300}  # Reduced from 500
        )
        
        elapsed = time.time() - start_time
        print(f"Optimization completed in {elapsed:.1f}s")
    
    return result

def display_results(result, elapsed_time=None):
    """Display optimization results and run final simulation"""
    
    optimal_params = result.x
    spin_rpm, launch_angle, release_angle, tilt_angle, aoa = optimal_params
    optimal_distance = -result.fun  # Convert back to positive
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    if elapsed_time:
        print(f"\nTotal optimization time: {elapsed_time:.1f}s")
    print(f"\nOptimal Parameters (V0 = {FIXED_VELOCITY} m/s, Height = {FIXED_RELEASE_HEIGHT} m):")
    print(f"  Spin:           {spin_rpm:.1f} RPM")
    print(f"  Launch Angle:   {launch_angle:.2f}°")
    print(f"  Release Angle:  {release_angle:.2f}°")
    print(f"  Tilt Angle:     {tilt_angle:.2f}°")
    print(f"  Angle of Attack:{aoa:.2f}°")
    print(f"\nOptimized Distance: {optimal_distance:.2f} m")
    
    # Run final simulation to get full trajectory
    print("\nRunning final simulation with optimal parameters...")
    sol = simulate_3d_throw(
        FIXED_VELOCITY, 
        spin_rpm, 
        FIXED_RELEASE_HEIGHT,
        launch_angle, 
        release_angle, 
        tilt_angle, 
        aoa,
        AIR_TEMP_C, 
        AIR_PRESSURE_PA
    )
    
    x, y, z = sol.y[0], sol.y[1], sol.y[2]
    flight_time = sol.t[-1]
    max_height = np.max(z)
    drift = y[-1]
    
    print(f"\nFinal Results:")
    print(f"  Distance:       {x[-1]:.2f} m")
    print(f"  Drift:          {drift:.2f} m")
    print(f"  Flight Time:    {flight_time:.2f} s")
    print(f"  Max Height:     {max_height:.2f} m")
    
    # Plot optimal trajectory
    plot_optimal_trajectory(x, y, z, optimal_params, optimal_distance)
    
    return optimal_params, x, y, z

def plot_optimal_trajectory(x, y, z, params, distance):
    """Plot the optimal trajectory"""
    
    spin_rpm, launch_angle, release_angle, tilt_angle, aoa = params
    
    fig = plt.figure(figsize=(16, 5))
    
    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(x, y, z, linewidth=2, color='red', label='Optimal Trajectory')
    ax1.scatter(x[0], y[0], z[0], color='green', s=100, marker='o', label='Start')
    ax1.scatter(x[-1], y[-1], z[-1], color='red', s=100, marker='x', label='End')
    ax1.set_xlabel('Distance X (m)')
    ax1.set_ylabel('Drift Y (m)')
    ax1.set_zlabel('Height Z (m)')
    ax1.set_title('Optimal 3D Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top view
    ax2 = fig.add_subplot(132)
    ax2.plot(x, y, linewidth=2, color='red')
    ax2.scatter(x[0], y[0], color='green', s=100, marker='o', label='Start')
    ax2.scatter(x[-1], y[-1], color='red', s=100, marker='x', label='End')
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('Distance X (m)')
    ax2.set_ylabel('Drift Y (m)')
    ax2.set_title('Top View - Drift Pattern')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Side view
    ax3 = fig.add_subplot(133)
    ax3.plot(x, z, linewidth=2, color='red')
    ax3.scatter(x[0], z[0], color='green', s=100, marker='o', label='Start')
    ax3.scatter(x[-1], z[-1], color='red', s=100, marker='x', label='End')
    ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax3.set_xlabel('Distance X (m)')
    ax3.set_ylabel('Height Z (m)')
    ax3.set_title('Side View - Flight Path')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add parameter info
    param_text = (f"Optimal Parameters (V0={FIXED_VELOCITY} m/s, Height={FIXED_RELEASE_HEIGHT} m):\n"
                  f"Spin: {spin_rpm:.1f} RPM\n"
                  f"Launch: {launch_angle:.1f} deg, Release: {release_angle:.1f} deg\n"
                  f"Tilt: {tilt_angle:.1f} deg, AoA: {aoa:.1f} deg\n"
                  f"Distance: {distance:.2f} m")
    fig.text(0.5, 0.02, param_text, ha='center', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()

def compare_with_baseline(optimal_params, baseline_params=None):
    """Compare optimal throw with a baseline throw"""
    
    if baseline_params is None:
        # Default baseline (typical throw)
        baseline_params = [600, 8.0, 0.0, -12.0, 4.0]
    
    print("\n" + "=" * 80)
    print("COMPARISON: OPTIMAL vs BASELINE")
    print("=" * 80)
    
    # Run baseline
    print("\nBaseline throw:")
    sol_base = simulate_3d_throw(
        FIXED_VELOCITY, 
        baseline_params[0], FIXED_RELEASE_HEIGHT,
        baseline_params[1], baseline_params[2], 
        baseline_params[3], baseline_params[4],
        AIR_TEMP_C, AIR_PRESSURE_PA
    )
    baseline_distance = sol_base.y[0, -1]
    baseline_drift = sol_base.y[1, -1]
    print(f"  Distance: {baseline_distance:.2f} m")
    print(f"  Drift:    {baseline_drift:.2f} m")
    
    # Run optimal
    print("\nOptimal throw:")
    sol_opt = simulate_3d_throw(
        FIXED_VELOCITY, 
        optimal_params[0], FIXED_RELEASE_HEIGHT,
        optimal_params[1], optimal_params[2], 
        optimal_params[3], optimal_params[4],
        AIR_TEMP_C, AIR_PRESSURE_PA
    )
    optimal_distance = sol_opt.y[0, -1]
    optimal_drift = sol_opt.y[1, -1]
    print(f"  Distance: {optimal_distance:.2f} m")
    print(f"  Drift:    {optimal_drift:.2f} m")
    
    # Improvement
    improvement = optimal_distance - baseline_distance
    improvement_pct = (improvement / baseline_distance) * 100
    print(f"\nImprovement: +{improvement:.2f} m ({improvement_pct:.1f}%)")

# ==========================================
#   MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize frisbee throw distance')
    parser.add_argument('--method', '-m', type=str, default='differential_evolution',
                        choices=['differential_evolution', 'nelder-mead'],
                        help='Optimization method (default: differential_evolution)')
    parser.add_argument('--velocity', '-v', type=float, default=14.0,
                        help='Fixed initial velocity in m/s (default: 14.0)')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Compare with baseline throw')
    
    args = parser.parse_args()
    
    # Update fixed velocity if provided
    FIXED_VELOCITY = args.velocity
    
    # Run optimization with timing
    start_time = time.time()
    result = run_optimization(method=args.method)
    total_time = time.time() - start_time
    
    # Display and plot results
    optimal_params, x, y, z = display_results(result, total_time)
    
    # Optional comparison
    if args.compare:
        compare_with_baseline(optimal_params)
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

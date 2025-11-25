# type: ignore
import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
# Import the simulation model
try:
    from frisbee_3d_model_final import simulate_3d_throw  # type: ignore
except ImportError:
    from SpinAoAFrisbeeModel import simulate_3d_throw  # type: ignore
import time
import sys
import io
from contextlib import redirect_stdout
import os
import argparse

# ==========================================
#   OPTIMIZATION CONFIGURATION
# ==========================================

AIR_TEMP_C = 25.0
AIR_PRESSURE_PA = 101325

# Parameter bounds for optimization
BOUNDS = {
    'spin_rpm': (400, 1100),           
    'launch_angle': (5, 25),           # 5-25 deg is realistic for distance
    'release_angle': (-20, 20),        # Aim Left/Right to compensate for turn
    'tilt_angle': (-45, 45),           # Full range to find the true Hyzer angle
    'aoa': (-4, 15)                    
}

# ==========================================
#   OPTIMIZATION FUNCTIONS
# ==========================================

def objective_function(params, velocity, height):
    """
    Maximize Forward Distance (X) while penalizing Side Drift (Y).
    """
    spin_rpm, launch_angle, release_angle, tilt_angle, aoa = params
    
    try:
        # Suppress output
        with redirect_stdout(io.StringIO()):
            sol = simulate_3d_throw(
                velocity, spin_rpm, height,
                launch_angle, release_angle, tilt_angle, aoa,
                AIR_TEMP_C, AIR_PRESSURE_PA
            )
        
        x_final = sol.y[0, -1]
        y_final = sol.y[1, -1]
        
        # OBJECTIVE: Maximize X, but keep Y close to 0.
        # Penalty factor: losing 1m of distance is worth fixing 2m of drift.
        drift_penalty = 0.5 * abs(y_final)
        
        # If it goes backwards (negative X), massive penalty
        if x_final < 0: return 1e6
        
        # We minimize the negative score
        score = -(x_final - drift_penalty)
        return score
        
    except Exception:
        return 1e6

def run_optimization(velocity, height, method='differential_evolution', popsize=10, maxiter=30, workers=-1, initial_guess=None):
    
    bounds_list = [
        BOUNDS['spin_rpm'], BOUNDS['launch_angle'],
        BOUNDS['release_angle'], BOUNDS['tilt_angle'], BOUNDS['aoa']
    ]
    
    print("=" * 80)
    print(f"OPTIMIZING FOR MAX X-DISTANCE (V0 = {velocity} m/s)")
    print(f"Constraint: Penalizing side drift to ensure straight flight.")
    print("=" * 80)
    
    optimizer_args = (velocity, height)
    
    if method == 'differential_evolution':
        if workers == -1:
            total_cores = os.cpu_count() or 4
            workers = min(6, max(1, total_cores - 1)) 
            
        print(f">>> Global Search (PopSize={popsize}, MaxIter={maxiter}, Workers={workers})...")
        start_time = time.time()
        iteration_count = [0]
        
        def callback(xk, convergence):
            iteration_count[0] += 1
            progress = iteration_count[0] / maxiter
            if progress > 1.0: progress = 1.0
            bar_length = 30
            filled = int(bar_length * progress)
            bar = '#' * filled + '-' * (bar_length - filled)
            elapsed = time.time() - start_time
            
            msg = f"\r[{bar}] {progress*100:.0f}% | Iter {iteration_count[0]}/{maxiter} | {elapsed:.1f}s"
            sys.stdout.write(msg)
            sys.stdout.flush()
            return False
        
        result = differential_evolution(
            objective_function,
            bounds_list,
            args=optimizer_args,
            strategy='best1bin',
            maxiter=maxiter,
            popsize=popsize,
            tol=0.05,
            workers=workers,
            updating='deferred',
            callback=callback,
            polish=True
        )
        print()
    else:
        if initial_guess is None:
            initial_guess = [800, 12.0, 5.0, 15.0, 4.0] # Guessing Positive Tilt for Hyzer
        
        print(">>> Local Search (Nelder-Mead)...")
        start_time = time.time()
        result = minimize(
            objective_function,
            initial_guess,
            args=optimizer_args,
            method='Nelder-Mead',
            options={'disp': False, 'maxiter': 300}
        )
    
    return result

def display_results(result, velocity, height, elapsed_time=None):
    optimal_params = result.x
    spin_rpm, launch, release, tilt, aoa = optimal_params
    
    # Run final sim to get real stats
    sol = simulate_3d_throw(velocity, spin_rpm, height, launch, release, tilt, aoa, AIR_TEMP_C, AIR_PRESSURE_PA)
    x_final = sol.y[0, -1]
    y_final = sol.y[1, -1]
    max_z = max(sol.y[2])
    
    print("\n" + "=" * 80)
    print(f"RESULTS (Time: {elapsed_time:.1f}s)")
    print("=" * 80)
    print(f"Optimal Parameters:")
    print(f"  Spin:    {spin_rpm:.0f} RPM")
    print(f"  Launch:  {launch:.2f}°")
    print(f"  Release: {release:.2f}° (Aim)")
    print(f"  Tilt:    {tilt:.2f}° (Bank)")
    print(f"  AoA:     {aoa:.2f}°")
    print("-" * 40)
    print(f"Forward Distance (X): {x_final:.2f} m")
    print(f"Side Drift (Y):       {y_final:.2f} m")
    print(f"Max Height (Z):       {max_z:.2f} m")
    
    plot_trajectory(sol.y[0], sol.y[1], sol.y[2], optimal_params, x_final)

def plot_trajectory(x, y, z, params, distance):
    fig = plt.figure(figsize=(14, 6))
    
    # 3D View
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(x, y, z, linewidth=2, color='red')
    ax1.scatter(x[0], y[0], z[0], color='green', label='Start')
    ax1.scatter(x[-1], y[-1], z[-1], color='red', label='End')
    ax1.plot([0, max(x)], [0, 0], [0, 0], 'k--', alpha=0.3)
    ax1.set_title('3D Flight Path')
    ax1.set_xlabel('X (Forward)')
    ax1.set_ylabel('Y (Side)')
    ax1.set_zlabel('Z (Height)')
    
    # Top View (Checking S-Curve)
    ax2 = fig.add_subplot(132)
    ax2.plot(x, y, linewidth=2, color='blue')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5) # Centerline
    ax2.scatter(x[-1], y[-1], color='red', marker='x')
    ax2.set_title('Top View (Drift vs Distance)')
    ax2.set_xlabel('Distance X (m)')
    ax2.set_ylabel('Drift Y (m)')
    ax2.grid(True, alpha=0.3)
    
    # Side View (Checking Height)
    ax3 = fig.add_subplot(133)
    ax3.plot(x, z, linewidth=2, color='green')
    ax3.set_title('Side View (Height Profile)')
    ax3.set_xlabel('Distance X (m)')
    ax3.set_ylabel('Height Z (m)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--velocity', '-v', type=float, default=25.0, help='Throw velocity (m/s)')
    parser.add_argument('--height', '-H', type=float, default=1.2, help='Release height (m)')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--maxiter', type=int, default=30)
    parser.add_argument('--popsize', type=int, default=10)
    
    args = parser.parse_args()
    
    method = 'nelder-mead' if args.fast else 'differential_evolution'
    
    start = time.time()
    res = run_optimization(args.velocity, args.height, method=method, maxiter=args.maxiter, popsize=args.popsize)
    display_results(res, args.velocity, args.height, time.time() - start)
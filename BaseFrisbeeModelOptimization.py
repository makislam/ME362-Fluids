# type: ignore
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from frisbee_3d_model_final import simulate_3d_throw

# ==========================================
#   OPTIMIZATION CONFIGURATION
# ==========================================
# We increase V0 to 25 m/s because 14 m/s is too slow for S-Curves
FIXED_VELOCITY = 25.0   # m/s (Competition Power)
FIXED_SPIN = 800.0      # RPM (High spin for stability)
RELEASE_HEIGHT = 1.2    # m

# Parameter Bounds for the Optimizer
# (Min, Max)
BOUNDS = [
    (5.0, 25.0),    # Launch Angle (deg) - Don't check extreme highs/lows
    (-30.0, 0.0),   # Tilt Angle (deg)   - Hyzer range (Negative)
    (2.0, 12.0)     # Angle of Attack    - Positive nose up
]

# ==========================================

def objective_function(params):
    """
    The optimizer tries to MINIMIZE this value.
    We return (-Distance) to MAXIMIZE Distance.
    """
    launch_angle, tilt_angle, aoa = params
    
    # Run Simulation
    sol = simulate_3d_throw(
        v0=FIXED_VELOCITY, 
        spin_rpm=FIXED_SPIN, 
        release_height=RELEASE_HEIGHT,
        launch_angle=launch_angle, 
        release_angle=0.0, 
        tilt_angle=tilt_angle, 
        aoa=aoa, 
        temp_c=25.0, 
        pressure_pa=101325
    )
    
    # Extract Landing Position
    x_final = sol.y[0][-1]
    y_final = sol.y[1][-1]
    
    # Calculate Straight-Line Range from thrower (Hypotenuse)
    total_range = np.sqrt(x_final**2 + y_final**2)
    
    # Penalty for massive drift? (Optional)
    # If you want it to land near the center line, penalize Y.
    # For pure distance, we just use total_range.
    
    return -total_range # Negative because we want to Maximize

def run_optimization():
    print(f"--- OPTIMIZING 3D TRAJECTORY (V0={FIXED_VELOCITY} m/s) ---")
    print("Searching for optimal Launch, Tilt, and AoA...")
    
    # Run Genetic Algorithm (Differential Evolution)
    # This is better than minimize() for finding global peaks in complex curves
    result = differential_evolution(
        objective_function, 
        BOUNDS, 
        strategy='best1bin', 
        maxiter=20,
        popsize=15,
        tol=0.01,
        disp=True # Show progress
    )
    
    opt_launch, opt_tilt, opt_aoa = result.x
    max_dist = -result.fun
    
    print("\n--- OPTIMIZATION COMPLETE ---")
    print(f"Maximum Distance: {max_dist:.2f} meters")
    print(f"Optimal Launch:   {opt_launch:.2f} deg")
    print(f"Optimal Tilt:     {opt_tilt:.2f} deg (Hyzer)")
    print(f"Optimal AoA:      {opt_aoa:.2f} deg")
    
    plot_optimal_flight(opt_launch, opt_tilt, opt_aoa)

def plot_optimal_flight(launch, tilt, aoa):
    sol = simulate_3d_throw(FIXED_VELOCITY, FIXED_SPIN, RELEASE_HEIGHT, launch, 0.0, tilt, aoa, 25.0, 101325)
    x, y, z = sol.y[0], sol.y[1], sol.y[2]
    
    # 2D Overhead Plot (Top Down) to show the S-Curve
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y, linewidth=2)
    plt.plot([0, max(x)], [0, 0], 'k--', alpha=0.3) # Centerline
    plt.scatter(0, 0, color='green', label='Start')
    plt.scatter(x[-1], y[-1], color='red', label='End')
    plt.title(f"Top View (Drift)\nHyzer: {tilt:.1f}°")
    plt.xlabel("Distance X (m)")
    plt.ylabel("Drift Y (m)")
    plt.grid(True)
    plt.axis('equal')
    
    # Side View (Height)
    plt.subplot(1, 2, 2)
    plt.plot(x, z, linewidth=2)
    plt.title(f"Side View (Height)\nLaunch: {launch:.1f}°, AoA: {aoa:.1f}°")
    plt.xlabel("Distance X (m)")
    plt.ylabel("Height Z (m)")
    plt.grid(True)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_optimization()
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# ==========================================
#   OPTIMIZATION SETTINGS
# ==========================================
# We will find the best angle for this specific throw speed
INPUT_VELOCITY = 14.0    # m/s (Typical Ultimate throw)
RELEASE_HEIGHT = 1.5     # meters

# Search range for angles (don't waste time checking -90 degrees)
ANGLE_BOUNDS = (-10, 45) # degrees
# ==========================================

def get_distance_for_angle(angle_deg, v0, y0):
    """
    Helper function that runs the simulation for a specific angle 
    and returns ONLY the distance. 
    """
    # --- Physics Constants ---
    m = 0.175      # Mass (kg)
    g = 9.81       # Gravity (m/s^2)
    rho = 1.184   # Air density (kg/m^3)
    D = 0.26       # Diameter (m)
    area = np.pi * (D/2)**2    # Planform Area (m^2)
    CD0 = 0.08     # Drag Coeff (No spin/No alpha)
    CL0 = 0.15    # Lift Coeff (Camber only)

    # --- Initial Conditions ---
    angle_rad = np.radians(angle_deg)
    vx0 = v0 * np.cos(angle_rad)
    vy0 = v0 * np.sin(angle_rad)
    
    # --- Solver ---
    # Event: Hit Ground
    def hit_ground(t, state, *args): return state[2]
    hit_ground.terminal = True
    hit_ground.direction = -1
    
    # Equations of Motion (Nested to avoid passing tons of args)
    def equations(t, state):
        x, vx, y, vy = state
        v = np.sqrt(vx**2 + vy**2)
        if v == 0: return [vx, 0, vy, -g]
        
        k = (0.5 * rho * area * v) / m
        ax = -k * (CD0 * vx + CL0 * vy)
        ay = -g + k * (CL0 * vx - CD0 * vy)
        return [vx, ax, vy, ay]

    # Solve
    sol = solve_ivp(
        equations, (0, 15), [0, vx0, y0, vy0], 
        events=hit_ground, rtol=1e-5, atol=1e-8
    )
    
    if sol.t.size == 0: return 0.0
    return sol.y[0, -1] # Return Final X

def run_optimization():
    print(f"--- OPTIMIZING THROW FOR V0 = {INPUT_VELOCITY} m/s ---")
    
    # 1. Define Objective Function (Negative distance because we want to Maximize)
    def objective(theta):
        dist = get_distance_for_angle(theta, INPUT_VELOCITY, RELEASE_HEIGHT)
        return -dist # Minimize negative distance = Maximize positive distance

    # 2. Run Optimization (Golden Section Search)
    result = minimize_scalar(
        objective, 
        bounds=ANGLE_BOUNDS, 
        method='bounded'
    )
    
    optimal_angle = result.x
    max_distance = -result.fun
    
    print(f"Optimization Complete!")
    print(f"Optimal Launch Angle: {optimal_angle:.3f} degrees")
    print(f"Maximum Distance:     {max_distance:.3f} meters")
    
    # 3. Visualization (Sweep & Plot)
    visualize_results(optimal_angle, max_distance)

def visualize_results(opt_angle, max_dist):
    # Generate data for the "Hill" plot
    angles = np.linspace(ANGLE_BOUNDS[0], ANGLE_BOUNDS[1], 50)
    distances = [get_distance_for_angle(a, INPUT_VELOCITY, RELEASE_HEIGHT) for a in angles]
    
    plt.figure(figsize=(10, 6))
    
    # Plot the curve
    plt.plot(angles, distances, label='Distance vs Angle', color='blue', linewidth=2)
    
    # Mark the peak
    plt.plot(opt_angle, max_dist, 'ro', markersize=10, label='Optimal Point')
    plt.vlines(opt_angle, 0, max_dist, color='red', linestyle='--', alpha=0.5)
    
    plt.title(f"Optimization of Launch Angle (v0={INPUT_VELOCITY} m/s)")
    plt.xlabel("Launch Angle (degrees)")
    plt.ylabel("Total Distance (m)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Annotate the optimal point text
    plt.annotate(
        f"Peak: {opt_angle:.1f}°\nDist: {max_dist:.1f}m",
        xy=(opt_angle, max_dist), 
        xytext=(opt_angle+5, max_dist),
        arrowprops=dict(facecolor='black', shrink=0.05)
    )
    
    plt.show()

if __name__ == "__main__":
    run_optimization()
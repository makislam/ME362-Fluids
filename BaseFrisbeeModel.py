import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==========================================
#   USER CONFIGURATION (EDIT THESE VALUES)
# ==========================================
INITIAL_VELOCITY = 14.0   # m/s (Speed of the throw)
LAUNCH_ANGLE     = 10.0    # degrees (Angle relative to ground)
RELEASE_HEIGHT   = 1.5     # meters (Height of hand at release)

# ==========================================

def simulate_throw(v0, angle_deg, y0=1.5, plot_trajectory=False):
    """
    Simulates a frisbee throw (No spin/No Angle of Attack model).
    
    Arguments:
        v0:              Release velocity (m/s)
        angle_deg:       Release angle (degrees)
        y0:              Release height (m)
        plot_trajectory: If True, creates a plot with flight time info.
        
    Returns:
        final_distance:  Total x-distance traveled (m)
        flight_time:     Total time in the air (s)
    """
    
    # --- 1. Define Constants --- (At SATP conditions)
    m = 0.175      # Mass (kg)
    g = 9.81       # Gravity (m/s^2)
    rho = 1.184   # Air density (kg/m^3)
    D = 0.26       # Diameter (m)
    area = np.pi * (D/2)**2    # Planform Area (m^2)
    CD0 = 0.08     # Drag Coeff (No spin/No alpha)
    CL0 = 0.15    # Lift Coeff (Camber only)
    
    # --- 2. Set Initial Conditions ---
    angle_rad = np.radians(angle_deg)
    vx0 = v0 * np.cos(angle_rad)
    vy0 = v0 * np.sin(angle_rad)
    
    # State Vector: [x, vx, y, vy]
    initial_state = [0, vx0, y0, vy0]
    
    # --- 3. Define Stopping Condition (Ground) ---
    def hit_ground(t, state, *args):
        return state[2] 
    
    hit_ground.terminal = True  
    hit_ground.direction = -1   
    
    # --- 4. Run the Solver ---
    constants = (m, g, rho, area, CD0, CL0)
    
    sol = solve_ivp(
        equations_of_motion, 
        (0, 10), # Max time (will stop early)
        initial_state, 
        events=hit_ground, 
        args=constants,
        rtol=1e-6, atol=1e-9
    )
    
    # --- 5. Extract Results ---
    final_distance = sol.y[0, -1]
    flight_time = sol.t[-1]
    
    # --- 6. Plotting (Optional) ---
    if plot_trajectory:
        plt.figure(figsize=(10, 6))
        plt.plot(sol.y[0], sol.y[2], label='Trajectory', linewidth=2)
        
        # Add annotations
        plt.title(f"Frisbee Flight Path\nDistance: {final_distance:.2f}m | Flight Time: {flight_time:.2f}s")
        plt.xlabel("Distance (m)")
        plt.ylabel("Height (m)")
        plt.axhline(0, color='black', linewidth=1) # Ground line
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal') # Important to see the real arc shape
        plt.ylim(bottom=-0.5) # Give a little space below ground
        
        # Add initial conditions text box
        initial_conditions_text = (
            f"Initial Conditions:\n"
            f"Velocity: {v0:.1f} m/s\n"
            f"Launch Angle: {angle_deg:.1f}°\n"
            f"Release Height: {y0:.2f} m"
        )
        plt.text(0.02, 0.98, initial_conditions_text, 
                transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.legend()
        plt.show()
        
        print(f"--- Simulation Results ---")
        print(f"Flight Time: {flight_time:.4f} seconds")
        print(f"Total Range: {final_distance:.4f} meters")

    return final_distance, flight_time

def equations_of_motion(t, state, m, g, rho, area, CD0, CL0):
    """ Physics Engine """
    x, vx, y, vy = state
    v = np.sqrt(vx**2 + vy**2)
    
    if v == 0: return [vx, 0, vy, -g]
    
    k = (0.5 * rho * area * v) / m
    ax = -k * (CD0 * vx + CL0 * vy)
    ay = -g + k * (CL0 * vx - CD0 * vy)
    
    return [vx, ax, vy, ay]

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # This runs automatically using the variables defined at the very top
    simulate_throw(
        INITIAL_VELOCITY, 
        LAUNCH_ANGLE, 
        RELEASE_HEIGHT, 
        plot_trajectory=True
    )
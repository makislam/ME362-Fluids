import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from thermo import Mixture
import argparse

# ==========================================
#   USER CONFIGURATION
# ==========================================
AIR_TEMP_C = 25.0
AIR_PRESSURE_PA = 101325

# Default Throw Parameters
DEFAULT_V0 = 14.0           
DEFAULT_SPIN_RPM = 600.0    
DEFAULT_RELEASE_HEIGHT = 1.0

# Default Angles
DEFAULT_LAUNCH_ANGLE = 8.0    
DEFAULT_RELEASE_ANGLE = 0.0   
DEFAULT_TILT_ANGLE = -12.0    
DEFAULT_ANGLE_OF_ATTACK = 4.0 

# ==========================================

def calculate_air_density(temp_c, pressure_pa):
    temp_k = temp_c + 273.15
    air = Mixture('air', T=temp_k, P=pressure_pa)
    return air.rho

def simulate_3d_throw(v0, spin_rpm, release_height, launch_angle, release_angle, tilt_angle, aoa, temp_c, pressure_pa):
    m = 0.175           
    g = 9.81            
    rho = calculate_air_density(temp_c, pressure_pa)
    diameter = 0.269 
    area = 0.057     
    
    Izz = 0.00235       
    Ixx = 0.00122       
    Iyy = 0.00122       

    # --- DYNAMIC COEFFICIENT SELECTION ---
    # We switch models based on Velocity (Reynolds Number proxy).
    # Threshold is set to 16 m/s (~35 mph).
    
    if v0 < 16.0:
        # HUMMEL (2003) - High Drag / Low Speed
        # Better for short tosses where wobble/early separation increases drag.
        CL0, CL_a = 0.188, 2.37
        CD0, CD_a = 0.15, 1.24
        model_name = "Hummel (Short Range / High Drag)"
    else:
        # POTTS & CROWTHER (2002) - Low Drag / High Speed
        # Better for long stable flights (Wind Tunnel Data).
        CL0, CL_a = 0.20, 2.96
        CD0, CD_a = 0.08, 2.60
        model_name = "Potts (Long Range / Low Drag)"

    alpha_0 = -4 * np.pi/180
    
    print(f"--- Simulation Config ---")
    print(f"Velocity: {v0:.1f} m/s -> Selected Model: {model_name}")
    
    # Moments (Shared)
    CM0, CM_a = -0.01, -0.2 
    CR_r = -0.014      
    CN_r = -0.000034  

    # Initial Conditions
    gamma = np.radians(launch_angle)
    zeta = np.radians(release_angle)
    vx = v0 * np.cos(gamma) * np.cos(zeta)
    vy = v0 * np.cos(gamma) * np.sin(zeta)
    vz = v0 * np.sin(gamma)
    
    psi = zeta
    alpha_rad = np.radians(aoa)
    theta = gamma + alpha_rad 
    phi = np.radians(tilt_angle)
    
    spin_rad_s = spin_rpm * (2 * np.pi / 60.0)
    
    initial_state = [0, 0, release_height, vx, vy, vz, phi, theta, psi, 0, 0, spin_rad_s] 

    def equations(t, state):
        x, y, z, vx, vy, vz, phi, theta, psi, wx, wy, wz = state
        
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        ss, cs = np.sin(psi), np.cos(psi)
        tt = np.tan(theta)
        
        R_IB = np.array([
            [ct*cs, ct*ss, st],    
            [sp*st*cs - cp*ss, sp*st*ss + cp*cs, -sp*ct], 
            [-cp*st*cs - sp*ss, -cp*st*ss + sp*cs, cp*ct] 
        ])
        
        v_earth = np.array([vx, vy, vz])
        v_body = R_IB @ v_earth
        ub, vb, wb = v_body
        
        vel = np.linalg.norm(v_body)
        if vel == 0: vel = 0.001
        
        # Angle of Attack
        alpha = -np.arctan2(wb, ub)
        
        cl = CL0 + CL_a * alpha
        cd = CD0 + CD_a * (alpha - alpha_0)**2
        cm = CM0 + CM_a * alpha
        
        q_dyn = 0.5 * rho * area * vel**2
        FL = cl * q_dyn
        FD = cd * q_dyn
        
        sa, ca = np.sin(alpha), np.cos(alpha)
        
        # Forces (Body Frame)
        Fx_aero = -FD * ca + FL * sa
        Fz_aero = -FD * sa + FL * ca  
        Fy_aero = 0 
        
        F_body = np.array([Fx_aero, Fy_aero, Fz_aero])
        
        Mx = (CR_r * wx * diameter / (2*vel)) * q_dyn * diameter
        My = cm * q_dyn * diameter 
        Mz = CN_r * wz * q_dyn * diameter
        M_body = np.array([Mx, My, Mz])
        
        # Newton-Euler
        F_aero_earth = R_IB.T @ F_body
        F_gravity = np.array([0, 0, -m*g])
        accel_earth = (F_aero_earth + F_gravity) / m
        
        I_mat = np.diag([Ixx, Iyy, Izz])
        omega_vec = np.array([wx, wy, wz])
        term2 = np.cross(omega_vec, I_mat @ omega_vec)
        dw_dt = np.linalg.inv(I_mat) @ (M_body - term2)
        
        if np.abs(ct) < 0.001: ct = 0.001
        
        dphi   = wx + (wy * sp) * tt  
        dtheta = wy * cp              
        dpsi   = (wy * sp) / ct       
        
        return [vx, vy, vz, 
                accel_earth[0], accel_earth[1], accel_earth[2], 
                dphi, dtheta, dpsi, 
                dw_dt[0], dw_dt[1], dw_dt[2]]

    def hit_ground(t, state): return state[2]
    hit_ground.terminal = True
    hit_ground.direction = -1
    
    sol = solve_ivp(equations, (0, 15), initial_state, events=hit_ground, rtol=1e-5)
    return sol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Frisbee Flight Simulator')
    parser.add_argument('--velocity', '-v', type=float, default=DEFAULT_V0,
                        help=f'Initial velocity (m/s), default: {DEFAULT_V0}')
    parser.add_argument('--spin', '-s', type=float, default=DEFAULT_SPIN_RPM,
                        help=f'Spin rate (RPM), default: {DEFAULT_SPIN_RPM}')
    parser.add_argument('--height', '-z', type=float, default=DEFAULT_RELEASE_HEIGHT,
                        help=f'Release height (m), default: {DEFAULT_RELEASE_HEIGHT}')
    parser.add_argument('--launch-angle', '-l', type=float, default=DEFAULT_LAUNCH_ANGLE,
                        help=f'Launch angle - path up/down (degrees), default: {DEFAULT_LAUNCH_ANGLE}')
    parser.add_argument('--release-angle', '-r', type=float, default=DEFAULT_RELEASE_ANGLE,
                        help=f'Release angle - path left/right (degrees), default: {DEFAULT_RELEASE_ANGLE}')
    parser.add_argument('--tilt-angle', '-t', type=float, default=DEFAULT_TILT_ANGLE,
                        help=f'Tilt angle - hyzer/roll (degrees), default: {DEFAULT_TILT_ANGLE}')
    parser.add_argument('--aoa', '-a', type=float, default=DEFAULT_ANGLE_OF_ATTACK,
                        help=f'Angle of attack (degrees), default: {DEFAULT_ANGLE_OF_ATTACK}')
    
    args = parser.parse_args()
    
    sol = simulate_3d_throw(args.velocity, args.spin, args.height, 
                           args.launch_angle, args.release_angle, 
                           args.tilt_angle, args.aoa, 
                           AIR_TEMP_C, AIR_PRESSURE_PA)
    
    x, y, z = sol.y[0], sol.y[1], sol.y[2]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, linewidth=2, label='Flight Path')
    ax.scatter(x[0], y[0], z[0], color='green', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', label='End')
    
    ax.plot([0, max(x)], [0, 0], [0, 0], 'k--', alpha=0.5)
    ax.set_xlabel('Distance X (m)'); ax.set_ylabel('Drift Y (m)'); ax.set_zlabel('Height Z (m)')
    
    title_text = (f"3D Frisbee Trajectory (Adaptive Model)\n"
                  f"Dist: {x[-1]:.2f}m | Drift: {y[-1]:.2f}m\n"
                  f"V₀={args.velocity:.1f}m/s, Spin={args.spin:.0f}RPM, Height={args.height:.1f}m, "
                  f"Launch={args.launch_angle:.1f}°, Release={args.release_angle:.1f}°, Tilt={args.tilt_angle:.1f}°, AoA={args.aoa:.1f}°")
    ax.set_title(title_text, fontsize=9)
    ax.set_box_aspect([np.ptp(x), np.ptp(x)*0.5, np.ptp(x)*0.2])
    ax.legend()
    
    print(f"Dist: {x[-1]:.2f}m, Drift: {y[-1]:.2f}m")
    plt.show()
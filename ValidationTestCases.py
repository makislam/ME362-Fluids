import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from thermo import Mixture

# Import the simulation function from SpinAoAFrisbeeModel
from SpinAoAFrisbeeModel import simulate_3d_throw, calculate_air_density

# ==========================================
#   TEST CASE DEFINITIONS
# ==========================================

test_cases = [
    {
        "name": "Case 1: Short Flight (Hummel fssh3)",
        "description": "Short, controlled throw for lift/drag balance validation",
        "params": {
            "v0": 13.5,
            "spin_rpm": 477.0,
            "release_height": 1.0,
            "launch_angle": 7.2,
            "release_angle": 0.0,
            "tilt_angle": -10.0,
            "aoa": 9.0
        },
        "expected": {
            "distance": "18-20 m",
            "drift": "S-curve (right then left)",
            "notes": "High AoA creates significant lift"
        }
    },
    {
        "name": "Case 2: Long Flight (Hummel f2302)",
        "description": "Gold standard for distance throw validation",
        "params": {
            "v0": 14.0,
            "spin_rpm": 516.0,
            "release_height": 1.0,
            "launch_angle": 9.8,
            "release_angle": 0.0,
            "tilt_angle": -12.0,
            "aoa": 9.0
        },
        "expected": {
            "distance": "35-40 m",
            "flight_time": "~2.8 s",
            "notes": "Standard distance benchmark"
        }
    },
    {
        "name": "Case 3: Straight/Stable Throw (Ultimate Pull)",
        "description": "Competition speed stability benchmark",
        "params": {
            "v0": 25.0,
            "spin_rpm": 700.0,
            "release_height": 1.0,
            "launch_angle": 10.0,
            "release_angle": 0.0,
            "tilt_angle": -15.0,
            "aoa": 5.0
        },
        "expected": {
            "distance": "60+ m",
            "drift": "< 5m deviation",
            "notes": "Minimal drift expected"
        }
    }
]

# Air conditions
AIR_TEMP_C = 25.0
AIR_PRESSURE_PA = 101325

# ==========================================
#   RUN VALIDATION TESTS
# ==========================================

def run_validation_tests():
    """Run all validation test cases and display results"""
    
    results = []
    
    print("=" * 80)
    print("FRISBEE FLIGHT MODEL VALIDATION TESTS")
    print("=" * 80)
    print()
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"{case['name']}")
        print(f"{case['description']}")
        print(f"{'=' * 80}")
        
        # Extract parameters
        params = case['params']
        print(f"\nInitial Conditions:")
        print(f"  Velocity:       {params['v0']:.1f} m/s")
        print(f"  Spin:           {params['spin_rpm']:.0f} RPM")
        print(f"  Release Height: {params['release_height']:.1f} m")
        print(f"  Launch Angle:   {params['launch_angle']:.1f}°")
        print(f"  Release Angle:  {params['release_angle']:.1f}°")
        print(f"  Tilt Angle:     {params['tilt_angle']:.1f}°")
        print(f"  Angle of Attack:{params['aoa']:.1f}°")
        
        print(f"\nExpected Results:")
        for key, value in case['expected'].items():
            print(f"  {key.capitalize():15s}: {value}")
        
        # Run simulation
        sol = simulate_3d_throw(
            params['v0'], 
            params['spin_rpm'], 
            params['release_height'],
            params['launch_angle'], 
            params['release_angle'], 
            params['tilt_angle'], 
            params['aoa'],
            AIR_TEMP_C, 
            AIR_PRESSURE_PA
        )
        
        # Extract results
        x, y, z = sol.y[0], sol.y[1], sol.y[2]
        distance = x[-1]
        drift = y[-1]
        flight_time = sol.t[-1]
        max_height = np.max(z)
        
        print(f"\nSimulated Results:")
        print(f"  Distance:       {distance:.2f} m")
        print(f"  Drift:          {drift:.2f} m")
        print(f"  Flight Time:    {flight_time:.2f} s")
        print(f"  Max Height:     {max_height:.2f} m")
        
        # Store results
        results.append({
            "case": case['name'],
            "params": params,
            "distance": distance,
            "drift": drift,
            "flight_time": flight_time,
            "max_height": max_height,
            "trajectory": (x, y, z)
        })
    
    print(f"\n{'=' * 80}")
    print("SUMMARY OF ALL TEST CASES")
    print(f"{'=' * 80}\n")
    
    # Summary table
    print(f"{'Case':<40} {'Distance':<12} {'Drift':<12} {'Time':<10}")
    print(f"{'-'*80}")
    for result in results:
        case_name = result['case'].split(':')[0]  # Shorten name
        print(f"{case_name:<40} {result['distance']:>8.2f} m   {result['drift']:>8.2f} m   {result['flight_time']:>6.2f} s")
    
    return results

def plot_all_trajectories(results):
    """Create comparison plots for all test cases"""
    
    # 3D Plot - All trajectories
    fig = plt.figure(figsize=(16, 6))
    
    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    colors = ['blue', 'red', 'green']
    
    for i, result in enumerate(results):
        x, y, z = result['trajectory']
        case_label = result['case'].split(':')[0]  # Shorten label
        ax1.plot(x, y, z, linewidth=2, label=case_label, color=colors[i])
        ax1.scatter(x[0], y[0], z[0], color=colors[i], s=50, marker='o')
        ax1.scatter(x[-1], y[-1], z[-1], color=colors[i], s=50, marker='x')
    
    ax1.set_xlabel('Distance X (m)')
    ax1.set_ylabel('Drift Y (m)')
    ax1.set_zlabel('Height Z (m)')
    ax1.set_title('3D Trajectories - All Cases')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Top view (X-Y)
    ax2 = fig.add_subplot(132)
    for i, result in enumerate(results):
        x, y, z = result['trajectory']
        case_label = result['case'].split(':')[0]
        ax2.plot(x, y, linewidth=2, label=case_label, color=colors[i])
        ax2.scatter(x[0], y[0], color=colors[i], s=50, marker='o')
        ax2.scatter(x[-1], y[-1], color=colors[i], s=50, marker='x')
    
    ax2.set_xlabel('Distance X (m)')
    ax2.set_ylabel('Drift Y (m)')
    ax2.set_title('Top View - Drift Pattern')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
    
    # Side view (X-Z)
    ax3 = fig.add_subplot(133)
    for i, result in enumerate(results):
        x, y, z = result['trajectory']
        case_label = result['case'].split(':')[0]
        ax3.plot(x, z, linewidth=2, label=case_label, color=colors[i])
        ax3.scatter(x[0], z[0], color=colors[i], s=50, marker='o')
        ax3.scatter(x[-1], z[-1], color=colors[i], s=50, marker='x')
    
    ax3.set_xlabel('Distance X (m)')
    ax3.set_ylabel('Height Z (m)')
    ax3.set_title('Side View - Flight Path')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
    
    plt.tight_layout()
    plt.show()

# ==========================================
#   MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Run all validation tests
    results = run_validation_tests()
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_all_trajectories(results)
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

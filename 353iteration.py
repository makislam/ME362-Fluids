import math
from thermo import Mixture

def get_air_properties_thermo(T):
    """
    Calculates air properties using the 'thermo' library.
    Since 'Air' is a mixture, we define it by component mole fractions:
    Nitrogen (79%) and Oxygen (21%).
    """
    # Initialize Air as a Mixture at Temperature T and Pressure P (1 atm)
    # zs denotes mole fractions
    air = Mixture(['nitrogen', 'oxygen'], zs=[0.79, 0.21], T=T, P=101325)
    
    rho = air.rho
    mu = air.mu
    k = air.k
    Pr = air.Pr
    
    return rho, mu, k, Pr

def calculate_t_infinity():
    # --- CONSTANTS ---
    T_tc = 471.38       # Measured Temp [K]
    T_wall = 293.72     # Wall Temp [K]
    V = 21.76           # Velocity [m/s]
    D = 0.0012          # Diameter [m]
    epsilon = 0.9       # Emissivity
    sigma = 5.67e-8     # Stefan-Boltzmann Constant
    
    # Initial Guess: T_infinity is assumed to be the measured thermocouple temp initially
    T_inf = T_tc 
    
    print(f"Using Library: thermo (Mixture: N2/O2)")
    print(f"{'Iter':<5} | {'T_inf(K)':<10} | {'rho':<6} | {'mu (x10^-5)':<12} | {'k':<8} | {'Pr':<6} | {'h':<8} | {'Change':<8}")
    print("-" * 95)
    
    for i in range(1, 20): 
        # 1. Get Properties at the current T_inf estimate
        # For the first iteration, T_inf is T_tc (471.38 K)
        rho, mu, k, Pr = get_air_properties_thermo(T_inf)
        
        # 2. Calculate Reynolds
        Re = (rho * V * D) / mu
        
        # 3. Calculate Nusselt (Whitaker)
        Nu = 2 + (0.4 * Re**0.5 + 0.06 * Re**(2/3)) * (Pr**0.4)
        
        # 4. Calculate h
        h = (Nu * k) / D
        
        # 5. Energy Balance
        q_rad = epsilon * sigma * (T_tc**4 - T_wall**4)
        T_inf_new = T_tc + (q_rad / h)
        
        change = abs(T_inf_new - T_inf)
        
        print(f"{i:<5} | {T_inf_new:<10.4f} | {rho:<6.4f} | {mu*1e5:<12.4f} | {k:<8.5f} | {Pr:<6.4f} | {h:<8.3f} | {change:<8.5f}")
        
        if change < 0.00001:
            print("-" * 95)
            print(f"CONVERGED at T_infinity = {T_inf_new:.5f} K")
            print(f"Total Error (Delta T) = {T_inf_new - T_tc:.5f} K")
            break
            
        # Update T_inf for the next iteration
        T_inf = T_inf_new

if __name__ == "__main__":
    calculate_t_infinity()
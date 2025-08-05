#!/usr/bin/env python3
"""
Power Sweep Thermal Analysis for InGaN/GaN HEMT
Shows temperature vs power for different operating conditions
"""

def main():
    print("="*70)
    print("InGaN/GaN HEMT - POWER SWEEP THERMAL ANALYSIS")
    print("="*70)
    
    # Device parameters
    print("\nDevice: 0.25μm × 100μm InGaN/GaN HEMT on SiC")
    print("Thermal Resistance: 50 K/W (typical for SiC substrate)")
    print("Maximum Junction Temperature: 150°C")
    
    # Constants
    Rth = 50  # K/W
    T_ambient = 27  # °C
    T_max = 150  # °C
    
    print("\n" + "="*70)
    print("POWER vs TEMPERATURE ANALYSIS")
    print("="*70)
    
    # Header
    print(f"\n{'Power':>8} {'Vds':>8} {'Ids':>8} {'Tj':>8} {'ΔT':>8} {'Status':>20}")
    print(f"{'(W)':>8} {'(V)':>8} {'(mA)':>8} {'(°C)':>8} {'(K)':>8}")
    print("-"*60)
    
    # Power sweep
    power_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    vds = 20  # V
    
    for power in power_levels:
        ids = power / vds * 1000  # mA
        delta_T = power * Rth
        Tj = T_ambient + delta_T
        
        if Tj > 150:
            status = "❌ CRITICAL"
        elif Tj > 125:
            status = "⚠️  WARNING"
        else:
            status = "✅ GOOD"
        
        print(f"{power:>8.1f} {vds:>8} {ids:>8.0f} {Tj:>8.1f} {delta_T:>8.1f} {status:>20}")
    
    # Maximum safe power
    P_max = (T_max - T_ambient) / Rth
    print(f"\nMaximum safe power: {P_max:.2f} W @ 150°C")
    
    # Visual representation
    print("\n" + "="*70)
    print("TEMPERATURE PROFILE VISUALIZATION")
    print("="*70)
    print("\nJunction Temperature (°C)")
    print("  180 ┤")
    print("  170 ┤                              ███ 3.0W")
    print("  160 ┤                         ████████")
    print("  150 ┤─────────────────────████████████─── MAX LIMIT")
    print("  140 ┤                ████████████")
    print("  130 ┤           ████████████ 2.0W")
    print("  120 ┤      ████████████")
    print("  110 ┤ ████████████")
    print("  100 ┤█████████ 1.0W")
    print("   90 ┤████")
    print("   80 ┤")
    print("   70 ┤")
    print("   60 ┤")
    print("   50 ┤")
    print("   40 ┤")
    print("   27 └────┬────┬────┬────┬────┬────┬────")
    print("       0.0  0.5  1.0  1.5  2.0  2.5  3.0  Power (W)")
    
    # Efficiency considerations
    print("\n" + "="*70)
    print("THERMAL DERATING AND EFFICIENCY")
    print("="*70)
    
    print("\nOutput Power Derating with Temperature:")
    temps = [27, 50, 75, 100, 125, 150]
    print(f"\n{'Tcase':>8} {'Pmax':>8} {'Derating':>10}")
    print(f"{'(°C)':>8} {'(W)':>8} {'(%)':>10}")
    print("-"*30)
    
    for T in temps:
        P_derated = (T_max - T) / Rth
        derating = (1 - P_derated/P_max) * 100
        print(f"{T:>8} {P_derated:>8.2f} {derating:>10.1f}")
    
    # Recommendations
    print("\n" + "="*70)
    print("THERMAL MANAGEMENT RECOMMENDATIONS")
    print("="*70)
    print("\n1. CONTINUOUS WAVE (CW) OPERATION:")
    print("   • P < 2.0W: Passive cooling sufficient")
    print("   • P = 2.0-2.5W: Add heat spreader")
    print("   • P > 2.5W: Active cooling required")
    
    print("\n2. PULSED OPERATION:")
    print("   • 10% duty cycle: Can handle up to 10W peak")
    print("   • 50% duty cycle: Limited to ~4W peak")
    print("   • Consider thermal time constant (τ = 5μs)")
    
    print("\n3. RELIABILITY GUIDELINES:")
    print("   • Keep Tj < 125°C for >20 year lifetime")
    print("   • Every 10°C reduction doubles lifetime")
    print("   • Monitor case temperature in operation")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
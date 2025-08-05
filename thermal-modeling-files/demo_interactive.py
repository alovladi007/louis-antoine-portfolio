#!/usr/bin/env python3
"""
Interactive Thermal Analysis Demo for InGaN/GaN HEMT
Shows step-by-step thermal calculations with visualizations
"""

import math
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f" {text} ")
    print("="*70)

def print_section(text):
    """Print section header"""
    print(f"\n{text}")
    print("-"*len(text))

def animate_loading(text, duration=1):
    """Simple loading animation"""
    print(f"\n{text}", end="")
    for _ in range(3):
        time.sleep(duration/3)
        print(".", end="", flush=True)
    print(" Done!")

def create_temperature_plot():
    """Create ASCII art temperature distribution"""
    print("\n3D Temperature Distribution:")
    print("="*50)
    print("         TOP VIEW (at 2DEG)")
    print("    Source    Gate    Drain")
    print("    |---------|▓▓▓|----------|")
    print("    |         |███|          |  ← Hot spot")
    print("    |         |▓▓▓|          |")
    print("    |---------|---|----------|")
    print()
    print("Temperature Scale:")
    print("  ░░░ : 27°C (Ambient)")
    print("  ▒▒▒ : 50-80°C")
    print("  ▓▓▓ : 80-120°C")
    print("  ███ : >120°C (Hot spot)")

def calculate_thermal_resistance(substrate, gate_area):
    """Calculate realistic thermal resistance"""
    # Simplified but realistic thermal resistance model
    # Based on published data for GaN HEMTs
    
    # Base spreading resistance in GaN layers
    R_spreading_gan = 15  # K/W for typical gate area
    
    # Substrate contribution
    substrate_R = {
        "SiC": 35,      # Best thermal conductor
        "Si": 65,       # Moderate
        "Sapphire": 185, # Poor thermal conductor
        "GaN": 45       # Good but expensive
    }
    
    R_substrate = substrate_R.get(substrate, 50)
    
    # Interface resistance
    R_interface = 10  # K/W
    
    # Total thermal resistance
    R_total = R_spreading_gan + R_substrate + R_interface
    
    return R_total, R_spreading_gan, R_substrate, R_interface

def main():
    """Run interactive thermal analysis demo"""
    
    print_header("InGaN/GaN HEMT THERMAL ANALYSIS DEMO")
    
    # Step 1: Device Configuration
    print_section("STEP 1: Device Configuration")
    print("Device: InGaN/GaN HEMT")
    print("Gate dimensions: 0.25 μm × 100 μm")
    print("Technology: Enhancement-mode (E-mode)")
    
    input("\nPress Enter to continue...")
    
    # Step 2: Select Substrate
    print_section("STEP 2: Substrate Selection")
    print("Available substrates:")
    substrates = {
        "1": ("SiC", 370, "Best thermal performance"),
        "2": ("Si", 148, "Cost-effective, moderate performance"),
        "3": ("Sapphire", 35, "Low cost, poor thermal performance"),
        "4": ("GaN", 230, "Excellent but expensive")
    }
    
    for key, (name, k, desc) in substrates.items():
        print(f"  {key}. {name} (k = {k} W/m·K) - {desc}")
    
    choice = input("\nSelect substrate (1-4) [default: 1]: ") or "1"
    substrate, k_value, _ = substrates.get(choice, substrates["1"])
    
    print(f"\nSelected: {substrate} substrate")
    animate_loading("Configuring device structure")
    
    # Step 3: Operating Conditions
    print_section("STEP 3: Operating Conditions")
    print("Enter operating conditions:")
    
    try:
        vds = float(input("  Drain voltage Vds (V) [default: 20]: ") or "20")
        ids = float(input("  Drain current Ids (mA) [default: 100]: ") or "100") / 1000
    except:
        vds, ids = 20, 0.1
    
    power = vds * ids
    print(f"\nPower dissipation: {power:.2f} W")
    
    # Step 4: Thermal Analysis
    print_section("STEP 4: Thermal Analysis")
    animate_loading("Calculating thermal resistance")
    
    gate_area = 0.25e-6 * 100e-6  # m²
    R_total, R_gan, R_sub, R_int = calculate_thermal_resistance(substrate, gate_area)
    
    print("\nThermal Resistance Breakdown:")
    print(f"  GaN spreading resistance:  {R_gan:6.1f} K/W ({100*R_gan/R_total:4.1f}%)")
    print(f"  Substrate resistance:      {R_sub:6.1f} K/W ({100*R_sub/R_total:4.1f}%)")
    print(f"  Interface resistance:      {R_int:6.1f} K/W ({100*R_int/R_total:4.1f}%)")
    print(f"  {'─'*35}")
    print(f"  TOTAL:                     {R_total:6.1f} K/W")
    
    # Step 5: Temperature Calculation
    print_section("STEP 5: Temperature Calculation")
    animate_loading("Computing temperature distribution")
    
    T_ambient = 27  # °C
    T_rise = power * R_total
    T_junction = T_ambient + T_rise
    
    print(f"\nTemperature Results:")
    print(f"  Ambient temperature:    {T_ambient:6.1f} °C")
    print(f"  Temperature rise:       {T_rise:6.1f} K")
    print(f"  Junction temperature:   {T_junction:6.1f} °C")
    
    # Temperature at different points
    T_interface = T_junction - power * R_gan
    T_substrate_bottom = T_ambient
    
    print(f"\nTemperature Profile:")
    print(f"  2DEG (hot spot):        {T_junction:6.1f} °C")
    print(f"  GaN/substrate interface:{T_interface:6.1f} °C")
    print(f"  Substrate bottom:       {T_substrate_bottom:6.1f} °C")
    
    # Step 6: Visualization
    print_section("STEP 6: Temperature Distribution")
    create_temperature_plot()
    
    # Step 7: Thermal Assessment
    print_section("STEP 7: Thermal Assessment")
    
    if T_junction > 150:
        status = "⚠️  CRITICAL - Exceeds 150°C limit!"
        recommendation = "Reduce power or improve cooling"
    elif T_junction > 125:
        status = "⚠️  WARNING - Approaching thermal limit"
        recommendation = "Consider thermal management"
    elif T_junction > 100:
        status = "✓ ACCEPTABLE - Within safe limits"
        recommendation = "Monitor during long-term operation"
    else:
        status = "✓ EXCELLENT - Low temperature"
        recommendation = "Good thermal margin"
    
    print(f"\nStatus: {status}")
    print(f"Recommendation: {recommendation}")
    
    # Step 8: Comparison
    print_section("STEP 8: Substrate Comparison")
    print("\nThermal performance for all substrates ({:.1f}W):".format(power))
    print(f"{'Substrate':12} {'Rth (K/W)':>10} {'Tj (°C)':>10} {'Status':>15}")
    print("-"*50)
    
    for sub in ["SiC", "Si", "Sapphire", "GaN"]:
        R, _, _, _ = calculate_thermal_resistance(sub, gate_area)
        T = T_ambient + power * R
        status = "OK" if T < 150 else "TOO HOT"
        print(f"{sub:12} {R:10.1f} {T:10.1f} {status:>15}")
    
    # Step 9: Transient Analysis
    print_section("STEP 9: Transient Analysis (Pulsed Operation)")
    
    # Thermal time constant (simplified)
    C_th = 1e-7  # J/K (typical for small HEMT)
    tau = R_total * C_th
    
    print(f"\nThermal time constant: {tau*1e6:.1f} μs")
    
    duty_cycle = 0.1  # 10%
    T_avg = T_ambient + power * duty_cycle * R_total
    
    print(f"\nPulsed operation (10% duty cycle):")
    print(f"  Peak temperature:    {T_junction:6.1f} °C")
    print(f"  Average temperature: {T_avg:6.1f} °C")
    print(f"  Temperature reduction: {T_junction - T_avg:6.1f} K")
    
    # Summary
    print_header("ANALYSIS COMPLETE")
    
    print("\nSummary:")
    print(f"  Device: InGaN/GaN HEMT on {substrate}")
    print(f"  Power: {power:.2f} W ({vds}V × {ids*1000:.0f}mA)")
    print(f"  Thermal Resistance: {R_total:.1f} K/W")
    print(f"  Junction Temperature: {T_junction:.1f} °C")
    print(f"  Thermal Status: {status}")
    
    # Save results option
    save = input("\nSave results to file? (y/n) [default: y]: ") or "y"
    if save.lower() == 'y':
        filename = "thermal_analysis_results.txt"
        with open(filename, 'w') as f:
            f.write(f"InGaN/GaN HEMT Thermal Analysis Results\n")
            f.write(f"="*40 + "\n")
            f.write(f"Substrate: {substrate}\n")
            f.write(f"Operating Point: {vds}V, {ids*1000:.0f}mA\n")
            f.write(f"Power Dissipation: {power:.2f} W\n")
            f.write(f"Thermal Resistance: {R_total:.1f} K/W\n")
            f.write(f"Junction Temperature: {T_junction:.1f} °C\n")
            f.write(f"Status: {status}\n")
        print(f"\nResults saved to {filename}")
    
    print("\nThank you for using the thermal analysis demo!")
    print("For full 3D simulations, install dependencies and run:")
    print("  python simulations/python/thermal_simulation.py")

if __name__ == "__main__":
    main()
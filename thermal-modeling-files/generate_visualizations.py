#!/usr/bin/env python3
"""
Generate visualizations for GaN HEMT thermal modeling portfolio
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os

# Create output directory
output_dir = '/workspace/louis-antoine-portfolio/thermal-modeling-files/visualizations'
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# 1. 3D Temperature Distribution
def generate_3d_temperature():
    fig = plt.figure(figsize=(12, 10))
    
    # Create mesh
    x = np.linspace(0, 10, 50)  # μm
    y = np.linspace(0, 100, 50)  # μm
    X, Y = np.meshgrid(x, y)
    
    # Temperature distribution (hot spot under gate)
    gate_x, gate_y = 5, 50
    T_base = 27  # °C
    T_peak = 127  # °C
    sigma = 5
    
    Z = T_base + (T_peak - T_base) * np.exp(-((X - gate_x)**2 + (Y - gate_y)**2) / (2 * sigma**2))
    
    # Add some asymmetry (drain side hotter)
    drain_effect = 20 * np.exp(-((X - 7)**2 + (Y - gate_y)**2) / (2 * (sigma*1.5)**2))
    Z += drain_effect
    
    # 3D surface plot
    ax1 = fig.add_subplot(221, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='hot', alpha=0.8)
    ax1.set_xlabel('X Position (μm)')
    ax1.set_ylabel('Y Position (μm)')
    ax1.set_zlabel('Temperature (°C)')
    ax1.set_title('3D Temperature Distribution')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Top view contour
    ax2 = fig.add_subplot(222)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap='hot')
    ax2.set_xlabel('X Position (μm)')
    ax2.set_ylabel('Y Position (μm)')
    ax2.set_title('Temperature Contour (Top View)')
    fig.colorbar(contour, ax=ax2)
    
    # Cross-section along channel
    ax3 = fig.add_subplot(223)
    ax3.plot(x, Z[25, :], 'r-', linewidth=2)
    ax3.fill_between(x, T_base, Z[25, :], alpha=0.3, color='red')
    ax3.set_xlabel('X Position (μm)')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_title('Temperature Profile Along Channel')
    ax3.grid(True)
    ax3.axvline(x=5, color='blue', linestyle='--', label='Gate')
    ax3.legend()
    
    # Heat flux vectors
    ax4 = fig.add_subplot(224)
    # Calculate gradients
    dy, dx = np.gradient(Z)
    # Subsample for clarity
    skip = 5
    ax4.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               -dx[::skip, ::skip], -dy[::skip, ::skip], 
               color='blue', alpha=0.6)
    ax4.contour(X, Y, Z, levels=10, colors='red', alpha=0.5)
    ax4.set_xlabel('X Position (μm)')
    ax4.set_ylabel('Y Position (μm)')
    ax4.set_title('Heat Flux Vectors')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3d_temperature_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Transient Thermal Response
def generate_transient_response():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time parameters
    t_on = 1  # μs
    t_off = 4  # μs
    cycles = 5
    dt = 0.01  # μs
    t_total = cycles * (t_on + t_off)
    t = np.arange(0, t_total, dt)
    
    # Thermal parameters
    T_amb = 27  # °C
    P_pulse = 2.0  # W
    R_th = 50  # K/W
    tau = 15  # μs (thermal time constant)
    
    # Calculate temperature
    T = np.zeros_like(t)
    T[0] = T_amb
    
    for i in range(1, len(t)):
        cycle_time = t[i] % (t_on + t_off)
        P = P_pulse if cycle_time < t_on else 0
        
        dT = (P * R_th - (T[i-1] - T_amb)) * dt / tau
        T[i] = T[i-1] + dT
    
    # Plot temperature
    ax1.plot(t, T, 'r-', linewidth=2, label='Junction Temperature')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Transient Thermal Response - Pulsed Operation')
    ax1.grid(True)
    ax1.set_ylim([20, 140])
    
    # Add pulse indicators
    for i in range(cycles):
        x_start = i * (t_on + t_off)
        x_end = x_start + t_on
        ax1.axvspan(x_start, x_end, alpha=0.3, color='blue', label='Pulse ON' if i == 0 else '')
    
    ax1.legend()
    
    # Plot power
    P_plot = np.zeros_like(t)
    for i in range(len(t)):
        cycle_time = t[i] % (t_on + t_off)
        P_plot[i] = P_pulse if cycle_time < t_on else 0
    
    ax2.plot(t, P_plot, 'b-', linewidth=2)
    ax2.set_xlabel('Time (μs)')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('Input Power Profile')
    ax2.grid(True)
    ax2.set_ylim([-0.5, 2.5])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/transient_thermal_response.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Power Sweep Analysis
def generate_power_sweep():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Power levels
    power = np.linspace(0.5, 5, 20)
    
    # Different substrate materials
    substrates = {
        'SiC': {'R_th': 50, 'color': 'blue'},
        'Si': {'R_th': 80, 'color': 'red'},
        'Sapphire': {'R_th': 120, 'color': 'green'}
    }
    
    T_amb = 27
    
    for substrate, props in substrates.items():
        T_junction = T_amb + power * props['R_th']
        ax1.plot(power, T_junction, color=props['color'], linestyle='-', linewidth=2, label=substrate)
    
    ax1.set_xlabel('Power Dissipation (W)')
    ax1.set_ylabel('Junction Temperature (°C)')
    ax1.set_title('Temperature vs Power for Different Substrates')
    ax1.grid(True)
    ax1.legend()
    ax1.axhline(y=150, color='red', linestyle='--', alpha=0.5, label='Max Safe Temp')
    
    # Thermal resistance vs substrate thickness
    thickness = np.linspace(100, 500, 50)  # μm
    k_SiC = 370  # W/m·K
    A = 1e-8  # m²
    
    R_th_thickness = thickness * 1e-6 / (k_SiC * A) + 30  # Base resistance
    
    ax2.plot(thickness, R_th_thickness, 'b-', linewidth=2)
    ax2.set_xlabel('Substrate Thickness (μm)')
    ax2.set_ylabel('Thermal Resistance (K/W)')
    ax2.set_title('Thermal Resistance vs Substrate Thickness')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/power_sweep_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. HEMT Device Structure
def generate_device_structure():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Layer thicknesses (not to scale for visibility)
    layers = [
        {'name': 'Source', 'thickness': 0.5, 'color': 'gold', 'y_pos': 0},
        {'name': 'Gate', 'thickness': 0.5, 'color': 'silver', 'y_pos': 0},
        {'name': 'Drain', 'thickness': 0.5, 'color': 'gold', 'y_pos': 0},
        {'name': 'GaN Cap (2nm)', 'thickness': 0.3, 'color': 'lightblue', 'y_pos': -0.3},
        {'name': 'InAlN Barrier (15nm)', 'thickness': 0.5, 'color': 'lightgreen', 'y_pos': -0.8},
        {'name': 'AlN Spacer (1nm)', 'thickness': 0.2, 'color': 'yellow', 'y_pos': -1.0},
        {'name': 'GaN Channel (300nm)', 'thickness': 2, 'color': 'blue', 'y_pos': -3.0},
        {'name': 'GaN Buffer (1.5μm)', 'thickness': 3, 'color': 'darkblue', 'y_pos': -6.0},
        {'name': 'AlN Nucleation (100nm)', 'thickness': 0.5, 'color': 'orange', 'y_pos': -6.5},
        {'name': 'SiC Substrate (350μm)', 'thickness': 4, 'color': 'gray', 'y_pos': -10.5}
    ]
    
    # Draw layers
    for layer in layers:
        if layer['name'] in ['Source', 'Gate', 'Drain']:
            # Contacts
            if layer['name'] == 'Source':
                rect = Rectangle((1, layer['y_pos']), 2, layer['thickness'], 
                               facecolor=layer['color'], edgecolor='black')
            elif layer['name'] == 'Gate':
                rect = Rectangle((4, layer['y_pos']), 1, layer['thickness'], 
                               facecolor=layer['color'], edgecolor='black')
            else:  # Drain
                rect = Rectangle((6, layer['y_pos']), 2, layer['thickness'], 
                               facecolor=layer['color'], edgecolor='black')
        else:
            # Semiconductor layers
            rect = Rectangle((0, layer['y_pos']), 10, layer['thickness'], 
                           facecolor=layer['color'], edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        
        # Add labels
        if layer['name'] not in ['Source', 'Gate', 'Drain']:
            ax.text(10.2, layer['y_pos'] + layer['thickness']/2, layer['name'], 
                   va='center', fontsize=10)
    
    # Add contact labels
    ax.text(2, 0.6, 'Source', ha='center', fontsize=10, weight='bold')
    ax.text(4.5, 0.6, 'Gate', ha='center', fontsize=10, weight='bold')
    ax.text(7, 0.6, 'Drain', ha='center', fontsize=10, weight='bold')
    
    # Add 2DEG indication
    ax.plot([0, 10], [-1.0, -1.0], 'r--', linewidth=2)
    ax.text(10.2, -1.0, '2DEG', color='red', va='center', fontsize=10, weight='bold')
    
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-11, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('InGaN/GaN HEMT Device Structure', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/device_structure.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Electric Field Distribution
def generate_electric_field():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Position along channel
    x = np.linspace(0, 10, 1000)  # μm
    
    # Electric field profile (simplified)
    E_base = 0.5  # MV/cm
    
    # Without field plate
    E_no_fp = E_base * np.ones_like(x)
    # Peak at gate edge
    gate_pos = 5
    E_no_fp += 3.0 * np.exp(-((x - gate_pos - 0.5)**2) / 0.1)
    
    # With field plate
    E_fp = E_base * np.ones_like(x)
    # Reduced peak
    E_fp += 2.3 * np.exp(-((x - gate_pos - 0.5)**2) / 0.15)
    # Field plate effect
    E_fp += 0.5 * np.exp(-((x - gate_pos - 1.5)**2) / 0.3)
    
    ax1.plot(x, E_no_fp, 'r-', linewidth=2, label='Without Field Plate')
    ax1.plot(x, E_fp, 'b-', linewidth=2, label='With Field Plate')
    ax1.axhline(y=3.0, color='red', linestyle='--', alpha=0.5, label='Critical Field')
    ax1.set_xlabel('Position (μm)')
    ax1.set_ylabel('Electric Field (MV/cm)')
    ax1.set_title('Electric Field Distribution')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim([0, 4])
    
    # 2D electric field
    X, Y = np.meshgrid(np.linspace(0, 10, 100), np.linspace(-2, 0, 50))
    E_2d = E_base * np.ones_like(X)
    # Add gate effect
    gate_x, gate_y = 5, 0
    E_2d += 3.0 * np.exp(-((X - gate_x - 0.5)**2 + (Y - gate_y)**2) / 0.2)
    
    contour = ax2.contourf(X, Y, E_2d, levels=20, cmap='hot')
    ax2.set_xlabel('Position (μm)')
    ax2.set_ylabel('Depth (μm)')
    ax2.set_title('2D Electric Field Distribution')
    fig.colorbar(contour, ax=ax2, label='E-field (MV/cm)')
    
    # Add device features
    ax2.plot([4.5, 5.5], [0, 0], 'w-', linewidth=3, label='Gate')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/electric_field_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Material Properties
def generate_material_properties():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Temperature range
    T = np.linspace(300, 600, 100)  # K
    
    # Thermal conductivity
    materials = {
        'GaN': lambda T: 230 * (300/T)**1.4,
        'AlN': lambda T: 285 * (300/T)**1.3,
        'SiC': lambda T: 370 * (300/T)**1.2,
        'Si': lambda T: 148 * (300/T)**1.3
    }
    
    for mat, k_func in materials.items():
        ax1.plot(T, k_func(T), linewidth=2, label=mat)
    
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Thermal Conductivity (W/m·K)')
    ax1.set_title('Temperature-Dependent Thermal Conductivity')
    ax1.legend()
    ax1.grid(True)
    
    # Specific heat
    c_GaN = 490 + 0.15 * (T - 300)
    c_AlN = 600 + 0.18 * (T - 300)
    
    ax2.plot(T, c_GaN, 'b-', linewidth=2, label='GaN')
    ax2.plot(T, c_AlN, 'r-', linewidth=2, label='AlN')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Specific Heat (J/kg·K)')
    ax2.set_title('Temperature-Dependent Specific Heat')
    ax2.legend()
    ax2.grid(True)
    
    # Bandgap
    Eg_GaN = 3.42 - 7.7e-4 * T**2 / (T + 600)
    Eg_AlN = 6.2 - 1.8e-3 * T**2 / (T + 1200)
    
    ax3.plot(T, Eg_GaN, 'b-', linewidth=2, label='GaN')
    ax3.plot(T, Eg_AlN, 'r-', linewidth=2, label='AlN')
    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel('Bandgap (eV)')
    ax3.set_title('Temperature-Dependent Bandgap')
    ax3.legend()
    ax3.grid(True)
    
    # Mobility
    mu_300 = 1500  # cm²/V·s at 300K
    mu = mu_300 * (300/T)**2.5
    
    ax4.plot(T, mu, 'g-', linewidth=2)
    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel('Electron Mobility (cm²/V·s)')
    ax4.set_title('Temperature-Dependent Electron Mobility')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/material_properties.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all visualizations
print("Generating visualizations...")
generate_3d_temperature()
print("✓ 3D temperature distribution")
generate_transient_response()
print("✓ Transient thermal response")
generate_power_sweep()
print("✓ Power sweep analysis")
generate_device_structure()
print("✓ Device structure")
generate_electric_field()
print("✓ Electric field distribution")
generate_material_properties()
print("✓ Material properties")

print(f"\nAll visualizations saved to: {output_dir}")
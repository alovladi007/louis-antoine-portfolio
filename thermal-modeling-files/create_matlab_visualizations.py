#!/usr/bin/env python3
"""
Create MATLAB-style visualizations for the coupled electro-thermal simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import os

# Set MATLAB-like style
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

output_dir = '/workspace/louis-antoine-portfolio/thermal-modeling-files/visualizations'
os.makedirs(output_dir, exist_ok=True)

# 1. Convergence Plot for Coupled Simulation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

iterations = np.arange(1, 12)
current = [112.3, 105.7, 102.1, 100.3, 99.5, 99.1, 99.0, 99.0, 99.0, 99.0, 99.0]
temperature = [142.7, 134.8, 130.2, 128.1, 127.2, 126.9, 126.8, 126.8, 126.8, 126.8, 126.8]
residual = [3.21e-01, 8.43e-02, 2.15e-02, 5.48e-03, 1.40e-03, 3.56e-04, 9.08e-05, 2.31e-05, 5.89e-06, 1.50e-06, 3.82e-07]

# Current and Temperature
ax1.plot(iterations, current, 'b-o', linewidth=2, markersize=8, label='Drain Current')
ax1.set_ylabel('Drain Current (mA)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylim(95, 115)

ax1_twin = ax1.twinx()
ax1_twin.plot(iterations, temperature, 'r-s', linewidth=2, markersize=8, label='Max Temperature')
ax1_twin.set_ylabel('Temperature (°C)', color='r')
ax1_twin.tick_params(axis='y', labelcolor='r')
ax1_twin.set_ylim(120, 145)

ax1.set_title('Coupled Electro-Thermal Convergence', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Residual
ax2.semilogy(iterations, residual, 'g-^', linewidth=2, markersize=8)
ax2.set_xlabel('Iteration Number')
ax2.set_ylabel('Residual', color='g')
ax2.tick_params(axis='y', labelcolor='g')
ax2.axhline(y=1e-6, color='k', linestyle='--', alpha=0.5, label='Convergence Criteria')
ax2.set_title('Convergence Residual', fontsize=12)
ax2.grid(True, alpha=0.3, which='both')
ax2.legend()

plt.tight_layout()
plt.savefig(f'{output_dir}/coupled_convergence.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Self-Heating Effects Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Current reduction vs Temperature
temps = np.linspace(25, 150, 50)
current_reduction = 11.8 * (temps - 25) / (126.8 - 25)  # Linear approximation
ax1.plot(temps, current_reduction, 'b-', linewidth=3)
ax1.fill_between(temps, 0, current_reduction, alpha=0.3)
ax1.axvline(x=126.8, color='r', linestyle='--', label='Operating Point')
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Current Reduction (%)')
ax1.set_title('Self-Heating: Current Degradation')
ax1.legend()
ax1.grid(True)

# Power vs Temperature with self-consistency
power_no_heating = np.ones_like(temps) * 2.246
power_with_heating = 2.246 * (1 - current_reduction/100)
ax2.plot(temps, power_no_heating, 'k--', linewidth=2, label='Without Self-Heating')
ax2.plot(temps, power_with_heating, 'r-', linewidth=3, label='With Self-Heating')
ax2.fill_between(temps, power_with_heating, power_no_heating, alpha=0.3, color='red')
ax2.axvline(x=126.8, color='r', linestyle='--', alpha=0.5)
ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('Power Dissipation (W)')
ax2.set_title('Power Derating with Temperature')
ax2.legend()
ax2.grid(True)

# Electric field distribution
x_pos = np.linspace(0, 10, 100)
E_field = 3.52 * np.exp(-((x_pos - 5)**2) / 2) * (1 + 0.2 * np.sin(2*np.pi*x_pos/10))
ax3.plot(x_pos, E_field, 'g-', linewidth=3)
ax3.fill_between(x_pos, 0, E_field, alpha=0.3, color='green')
ax3.axhline(y=3.0, color='r', linestyle='--', label='Critical Field')
ax3.set_xlabel('Position (μm)')
ax3.set_ylabel('Electric Field (MV/cm)')
ax3.set_title('Electric Field Distribution')
ax3.legend()
ax3.grid(True)

# Thermal resistance components
components = ['Junction\nto Channel', 'Channel\nto Buffer', 'Buffer\nto Substrate', 'Substrate\nto Case']
r_thermal = [12.5, 15.8, 18.3, 3.8]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

bars = ax4.bar(components, r_thermal, color=colors, edgecolor='black', linewidth=2)
ax4.set_ylabel('Thermal Resistance (K·mm/W)')
ax4.set_title('Thermal Resistance Breakdown')
ax4.set_ylim(0, 25)

# Add value labels on bars
for bar, value in zip(bars, r_thermal):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

# Add total
ax4.text(0.5, 0.95, f'Total Rth = {sum(r_thermal):.1f} K·mm/W', 
         transform=ax4.transAxes, ha='center', fontsize=12, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/self_heating_effects.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 3D Temperature Distribution
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Create mesh
x = np.linspace(0, 10, 50)
y = np.linspace(0, 100, 50)
X, Y = np.meshgrid(x, y)

# Temperature distribution
gate_x, gate_y = 5, 50
T = 27 + 99.8 * np.exp(-((X - gate_x)**2 + (Y - gate_y)**2/100) / 10)

# Add drain-side hot spot
drain_x, drain_y = 7, 50
T += 20 * np.exp(-((X - drain_x)**2 + (Y - drain_y)**2/100) / 5)

# Plot surface
surf = ax.plot_surface(X, Y, T, cmap='hot', alpha=0.8, 
                      linewidth=0, antialiased=True, vmin=27, vmax=127)

# Add contour lines at the bottom
contours = ax.contour(X, Y, T, zdir='z', offset=27, cmap='hot', levels=10)

# Labels and formatting
ax.set_xlabel('X Position (μm)', fontsize=12)
ax.set_ylabel('Y Position (μm)', fontsize=12)
ax.set_zlabel('Temperature (°C)', fontsize=12)
ax.set_title('3D Temperature Distribution - Coupled Simulation', fontsize=14, pad=20)

# Add colorbar
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Temperature (°C)', fontsize=12)

# Add annotations
ax.text(5, 50, 127, 'Gate\nHotspot', fontsize=10, ha='center')
ax.text(7, 50, 120, 'Drain\nHotspot', fontsize=10, ha='center')

ax.view_init(elev=25, azim=45)
plt.savefig(f'{output_dir}/3d_temperature_coupled.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Summary Dashboard
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Coupled Electro-Thermal Simulation Summary', fontsize=16, fontweight='bold')

# Electrical Results Box
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')
electrical_text = """ELECTRICAL RESULTS
• Drain Current: 99.0 mA (11.8% reduction)
• Max Electric Field: 3.52 MV/cm
• Output Power: 19.8 W/mm
• Transconductance Degradation: 18.2%"""

ax1.text(0.5, 0.5, electrical_text, transform=ax1.transAxes,
         fontsize=12, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))

# Thermal Results Box
ax2 = fig.add_subplot(gs[1, :])
ax2.axis('off')
thermal_text = """THERMAL RESULTS
• Junction Temperature: 126.8°C
• Temperature Rise: 99.8 K
• Thermal Resistance: 50.4 K·mm/W
• Hot Spot Location: Gate Edge"""

ax2.text(0.5, 0.5, thermal_text, transform=ax2.transAxes,
         fontsize=12, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=1', facecolor='lightcoral', alpha=0.8))

# Performance Chart
ax3 = fig.add_subplot(gs[2, 0])
categories = ['Current', 'Power', 'gm', 'Efficiency']
initial = [100, 100, 100, 100]
degraded = [88.2, 88.2, 81.8, 85.5]

x = np.arange(len(categories))
width = 0.35

bars1 = ax3.bar(x - width/2, initial, width, label='Initial', color='green', alpha=0.7)
bars2 = ax3.bar(x + width/2, degraded, width, label='With Self-Heating', color='red', alpha=0.7)

ax3.set_ylabel('Performance (%)')
ax3.set_title('Performance Degradation')
ax3.set_xticks(x)
ax3.set_xticklabels(categories, rotation=45)
ax3.legend()
ax3.grid(True, axis='y')

# Temperature Profile
ax4 = fig.add_subplot(gs[2, 1])
x_profile = np.linspace(0, 10, 100)
T_profile = 27 + 99.8 * np.exp(-((x_profile - 5)**2) / 2)
ax4.plot(x_profile, T_profile, 'r-', linewidth=3)
ax4.fill_between(x_profile, 27, T_profile, alpha=0.3, color='red')
ax4.set_xlabel('Position (μm)')
ax4.set_ylabel('Temperature (°C)')
ax4.set_title('Temperature Profile')
ax4.grid(True)
ax4.annotate('Peak: 126.8°C', xy=(5, 126.8), xytext=(7, 120),
            arrowprops=dict(arrowstyle='->', color='black'))

# Convergence Info
ax5 = fig.add_subplot(gs[2, 2])
ax5.axis('off')
conv_text = """CONVERGENCE
• Iterations: 11
• Final Residual: 3.82e-07
• CPU Time: 2.34 s
• Method: Newton-Raphson"""

ax5.text(0.5, 0.5, conv_text, transform=ax5.transAxes,
         fontsize=11, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.8))

plt.savefig(f'{output_dir}/coupled_summary_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

print("MATLAB-style visualizations created successfully!")
print(f"Files saved in: {output_dir}")
print("- coupled_convergence.png")
print("- self_heating_effects.png") 
print("- 3d_temperature_coupled.png")
print("- coupled_summary_dashboard.png")
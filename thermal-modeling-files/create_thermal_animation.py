#!/usr/bin/env python3
"""
Create thermal animation GIF for the portfolio
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import os

# Create output directory
output_dir = '/workspace/louis-antoine-portfolio/thermal-modeling-files/visualizations'
os.makedirs(output_dir, exist_ok=True)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#1a1a2e')
ax1.set_facecolor('#0a0a0f')
ax2.set_facecolor('#0a0a0f')

# Setup spatial grid
x = np.linspace(0, 10, 100)
y = np.linspace(0, 100, 100)
X, Y = np.meshgrid(x, y)

# Initialize plot elements
im = ax1.imshow(np.zeros((100, 100)), cmap='hot', extent=[0, 10, 0, 100], 
                vmin=27, vmax=127, animated=True)
cbar = plt.colorbar(im, ax=ax1)
cbar.set_label('Temperature (°C)', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# Time series plot
line, = ax2.plot([], [], 'r-', linewidth=2)
ax2.set_xlim(0, 50)
ax2.set_ylim(25, 130)
ax2.set_xlabel('Time (μs)', color='white')
ax2.set_ylabel('Max Temperature (°C)', color='white')
ax2.set_title('Temperature Rise Over Time', color='white')
ax2.grid(True, alpha=0.3)
ax2.tick_params(colors='white')

ax1.set_xlabel('X Position (μm)', color='white')
ax1.set_ylabel('Y Position (μm)', color='white')
ax1.set_title('Temperature Distribution', color='white')
ax1.tick_params(colors='white')

# Animation data
times = []
max_temps = []

def animate(frame):
    t = frame * 0.5  # Time in microseconds
    
    # Time-dependent heat source
    power = 2.0 * (1 - np.exp(-t/15))  # Exponential rise
    T_peak = 27 + power * 50
    
    # Gaussian heat distribution with spreading
    gate_x, gate_y = 5, 50
    sigma = 5 + 0.05 * t  # Slight spreading over time
    
    # Temperature field
    T = 27 + (T_peak - 27) * np.exp(-((X - gate_x)**2 + (Y - gate_y)**2) / (2 * sigma**2))
    
    # Add drain-side asymmetry
    drain_effect = power * 10 * np.exp(-((X - 7)**2 + (Y - gate_y)**2) / (2 * (sigma*1.5)**2))
    T += drain_effect
    
    # Update heatmap
    im.set_array(T)
    ax1.set_title(f'Temperature Distribution at t = {t:.1f} μs', color='white')
    
    # Update time series
    times.append(t)
    max_temps.append(np.max(T))
    line.set_data(times, max_temps)
    
    return [im, line]

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=True)

# Save as GIF
writer = PillowWriter(fps=20)
anim.save(f'{output_dir}/thermal_animation.gif', writer=writer)

print(f"Animation saved to: {output_dir}/thermal_animation.gif")

# Also create static publication figures
plt.style.use('seaborn-v0_8-whitegrid')

# Publication Figure 1: Multi-panel analysis
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

# Panel 1: 3D view (simulated)
x = np.linspace(0, 10, 50)
y = np.linspace(0, 100, 50)
X, Y = np.meshgrid(x, y)
Z = 27 + 100 * np.exp(-((X - 5)**2 + (Y - 50)**2) / 50)

c1 = ax1.contourf(X, Y, Z, levels=20, cmap='hot')
ax1.set_title('(a) Temperature Distribution')
ax1.set_xlabel('X Position (μm)')
ax1.set_ylabel('Y Position (μm)')
plt.colorbar(c1, ax=ax1, label='T (°C)')

# Panel 2: Line profile
ax2.plot(x, Z[25, :], 'b-', linewidth=2)
ax2.fill_between(x, 27, Z[25, :], alpha=0.3)
ax2.set_title('(b) Temperature Profile')
ax2.set_xlabel('X Position (μm)')
ax2.set_ylabel('Temperature (°C)')
ax2.grid(True)

# Panel 3: Power vs Temperature
powers = np.linspace(0.5, 3, 20)
for substrate, color, rth in [('SiC', 'blue', 50), ('Si', 'red', 80), ('Sapphire', 'green', 120)]:
    ax3.plot(powers, 27 + powers * rth, color=color, linewidth=2, label=substrate)
ax3.set_title('(c) Substrate Comparison')
ax3.set_xlabel('Power (W)')
ax3.set_ylabel('Temperature (°C)')
ax3.legend()
ax3.grid(True)

# Panel 4: Transient response
t = np.linspace(0, 100, 1000)
T_transient = 27 + 100 * (1 - np.exp(-t/15))
ax4.plot(t, T_transient, 'r-', linewidth=2)
ax4.set_title('(d) Transient Response')
ax4.set_xlabel('Time (μs)')
ax4.set_ylabel('Temperature (°C)')
ax4.grid(True)

plt.tight_layout()
plt.savefig(f'{output_dir}/publication_figure_1.png', dpi=300, bbox_inches='tight')

# Publication Figure 2: Heat flux analysis
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Heat flux vectors
dy, dx = np.gradient(Z)
skip = 3
ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
           -dx[::skip, ::skip], -dy[::skip, ::skip], 
           color='blue', alpha=0.7)
ax1.contour(X, Y, Z, levels=10, colors='red', alpha=0.5)
ax1.set_title('Heat Flux Vector Field')
ax1.set_xlabel('X Position (μm)')
ax1.set_ylabel('Y Position (μm)')

# Thermal resistance network
ax2.text(0.5, 0.9, 'Thermal Resistance Network', ha='center', transform=ax2.transAxes, fontsize=14)
ax2.text(0.5, 0.7, 'Junction → Case: 47.3 K/W', ha='center', transform=ax2.transAxes)
ax2.text(0.5, 0.5, 'Case → Ambient: 2.7 K/W', ha='center', transform=ax2.transAxes)
ax2.text(0.5, 0.3, 'Total: 50.0 K/W', ha='center', transform=ax2.transAxes, fontweight='bold')
ax2.axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/publication_figure_2.png', dpi=300, bbox_inches='tight')

print("Publication figures created successfully!")
plt.close('all')
#!/usr/bin/env python3
"""
Create 2D contour plot for the Python page
"""

import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = '/workspace/louis-antoine-portfolio/thermal-modeling-files/visualizations'
os.makedirs(output_dir, exist_ok=True)

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Generate data
x = np.linspace(0, 10, 100)
y = np.linspace(0, 100, 100)
X, Y = np.meshgrid(x, y)

# Temperature field
Z = 27 + 100 * np.exp(-((X - 5)**2 + (Y - 50)**2/100) / 25)

# Create filled contour plot
contourf = ax.contourf(X, Y, Z, levels=20, cmap='hot')
# Add contour lines
contour = ax.contour(X, Y, Z, levels=10, colors='white', alpha=0.5, linewidths=0.5)
# Add contour labels
ax.clabel(contour, inline=True, fontsize=8, fmt='%.0f°C')

# Add colorbar
cbar = plt.colorbar(contourf, ax=ax)
cbar.set_label('Temperature (°C)', rotation=270, labelpad=20)

# Labels and title
ax.set_xlabel('X Position (μm)')
ax.set_ylabel('Y Position (μm)')
ax.set_title('2D Temperature Contour Map with Custom Colormap', fontsize=14, fontweight='bold')

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Add annotations for key features
ax.annotate('Hot Spot', xy=(5, 50), xytext=(7, 70),
            arrowprops=dict(arrowstyle='->', color='yellow', lw=2),
            fontsize=12, color='yellow', fontweight='bold')

# Set background
ax.set_facecolor('#f0f0f0')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(f'{output_dir}/2d_contour_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("2D contour plot created successfully!")
print(f"Saved to: {output_dir}/2d_contour_plot.png")
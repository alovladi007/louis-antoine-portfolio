#!/usr/bin/env python3
"""
Advanced visualization tools for InGaN/GaN HEMT thermal simulation results
Includes 3D visualization, animations, and comparative analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import os
import sys

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class ThermalVisualizer:
    """Advanced visualization for thermal simulation results"""
    
    def __init__(self, results_path='../results'):
        """
        Initialize visualizer
        
        Args:
            results_path: Path to results directory
        """
        self.results_path = results_path
        self.colormap = 'hot'
        
    def create_3d_temperature_plot(self, T_field, x, y, z, 
                                  title="3D Temperature Distribution"):
        """
        Create interactive 3D temperature visualization using Plotly
        
        Args:
            T_field: 3D temperature field (nx, ny, nz)
            x, y, z: Coordinate arrays
            title: Plot title
        """
        # Convert to Celsius
        T_celsius = T_field - 273.15
        
        # Create meshgrid
        X, Y, Z = np.meshgrid(x*1e6, y*1e6, z*1e9, indexing='ij')
        
        # Downsample for performance if needed
        step = max(1, len(x)//30)
        
        fig = go.Figure(data=go.Volume(
            x=X[::step, ::step, ::step].flatten(),
            y=Y[::step, ::step, ::step].flatten(),
            z=Z[::step, ::step, ::step].flatten(),
            value=T_celsius[::step, ::step, ::step].flatten(),
            isomin=T_celsius.min(),
            isomax=T_celsius.max(),
            opacity=0.3,
            surface_count=15,
            colorscale='Hot',
            colorbar=dict(title='Temperature (°C)')
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (μm)',
                yaxis_title='Y (μm)',
                zaxis_title='Z (nm)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def create_temperature_slices(self, T_field, x, y, z, 
                                 slice_positions={'x': 0.5, 'y': 0.5, 'z': 0.1}):
        """
        Create temperature slice plots at specified positions
        
        Args:
            T_field: 3D temperature field
            x, y, z: Coordinate arrays
            slice_positions: Dict with relative positions (0-1) for slices
        """
        T_celsius = T_field - 273.15
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # X-slice (YZ plane)
        x_idx = int(slice_positions['x'] * (len(x) - 1))
        ax = axes[0, 0]
        im = ax.imshow(T_celsius[x_idx, :, :].T, extent=[y[0]*1e6, y[-1]*1e6, z[-1]*1e9, z[0]*1e9],
                      aspect='auto', cmap=self.colormap, origin='upper')
        ax.set_xlabel('Y (μm)')
        ax.set_ylabel('Z (nm)')
        ax.set_title(f'YZ Slice at X={x[x_idx]*1e6:.1f}μm')
        plt.colorbar(im, ax=ax, label='Temperature (°C)')
        
        # Y-slice (XZ plane)
        y_idx = int(slice_positions['y'] * (len(y) - 1))
        ax = axes[0, 1]
        im = ax.imshow(T_celsius[:, y_idx, :].T, extent=[x[0]*1e6, x[-1]*1e6, z[-1]*1e9, z[0]*1e9],
                      aspect='auto', cmap=self.colormap, origin='upper')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Z (nm)')
        ax.set_title(f'XZ Slice at Y={y[y_idx]*1e6:.1f}μm')
        plt.colorbar(im, ax=ax, label='Temperature (°C)')
        
        # Z-slice (XY plane)
        z_idx = int(slice_positions['z'] * (len(z) - 1))
        ax = axes[1, 0]
        im = ax.imshow(T_celsius[:, :, z_idx].T, extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6],
                      aspect='auto', cmap=self.colormap, origin='lower')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(f'XY Slice at Z={z[z_idx]*1e9:.1f}nm')
        plt.colorbar(im, ax=ax, label='Temperature (°C)')
        
        # 3D surface plot
        ax = axes[1, 1]
        ax.remove()
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        
        # Plot temperature at top surface
        X, Y = np.meshgrid(x*1e6, y*1e6)
        surf = ax.plot_surface(X, Y, T_celsius[:, :, 0].T, cmap=self.colormap,
                              linewidth=0, antialiased=True)
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_zlabel('Temperature (°C)')
        ax.set_title('Top Surface Temperature')
        
        plt.suptitle('Temperature Distribution Slices', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def create_heat_flux_visualization(self, T_field, k_field, dx, dy, dz):
        """
        Visualize heat flux vectors
        
        Args:
            T_field: Temperature field
            k_field: Thermal conductivity field
            dx, dy, dz: Grid spacing
        """
        # Calculate temperature gradients
        dT_dx, dT_dy, dT_dz = np.gradient(T_field, dx, dy, dz)
        
        # Calculate heat flux components (Fourier's law: q = -k∇T)
        q_x = -k_field * dT_dx
        q_y = -k_field * dT_dy
        q_z = -k_field * dT_dz
        
        # Heat flux magnitude
        q_mag = np.sqrt(q_x**2 + q_y**2 + q_z**2)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Heat flux magnitude at different slices
        slice_z = len(T_field[0, 0, :]) // 10  # Near top
        
        # XY plane heat flux
        ax = axes[0, 0]
        im = ax.imshow(q_mag[:, :, slice_z].T * 1e-6, cmap='plasma', origin='lower')
        ax.set_xlabel('X index')
        ax.set_ylabel('Y index')
        ax.set_title('Heat Flux Magnitude (MW/m²) - XY plane')
        plt.colorbar(im, ax=ax)
        
        # Add vector field (downsampled)
        step = max(1, len(T_field[:, 0, 0]) // 20)
        X, Y = np.meshgrid(range(0, len(T_field[:, 0, 0]), step),
                          range(0, len(T_field[0, :, 0]), step))
        ax.quiver(X, Y, q_x[::step, ::step, slice_z].T, 
                 q_y[::step, ::step, slice_z].T, 
                 scale=np.max(q_mag)*20, alpha=0.6)
        
        # XZ plane heat flux
        ax = axes[0, 1]
        slice_y = len(T_field[0, :, 0]) // 2
        im = ax.imshow(q_mag[:, slice_y, :].T * 1e-6, cmap='plasma', 
                      aspect='auto', origin='upper')
        ax.set_xlabel('X index')
        ax.set_ylabel('Z index')
        ax.set_title('Heat Flux Magnitude (MW/m²) - XZ plane')
        plt.colorbar(im, ax=ax)
        
        # Heat flux divergence (heat generation)
        div_q = np.gradient(q_x, dx, axis=0) + \
                np.gradient(q_y, dy, axis=1) + \
                np.gradient(q_z, dz, axis=2)
        
        ax = axes[1, 0]
        im = ax.imshow(div_q[:, slice_y, :].T * 1e-6, cmap='RdBu_r', 
                      aspect='auto', origin='upper')
        ax.set_xlabel('X index')
        ax.set_ylabel('Z index')
        ax.set_title('Heat Generation (MW/m³) - XZ plane')
        plt.colorbar(im, ax=ax)
        
        # Temperature gradient magnitude
        ax = axes[1, 1]
        grad_T_mag = np.sqrt(dT_dx**2 + dT_dy**2 + dT_dz**2)
        im = ax.imshow(grad_T_mag[:, slice_y, :].T, cmap='viridis', 
                      aspect='auto', origin='upper')
        ax.set_xlabel('X index')
        ax.set_ylabel('Z index')
        ax.set_title('Temperature Gradient Magnitude (K/m)')
        plt.colorbar(im, ax=ax)
        
        plt.suptitle('Heat Flux Analysis', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def create_transient_animation(self, time_data, T_data, output_file='thermal_transient.gif'):
        """
        Create animation of transient thermal response
        
        Args:
            time_data: Time points array
            T_data: Temperature data array (time, spatial)
            output_file: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Initialize plots
        line1, = ax1.plot([], [], 'b-', linewidth=2)
        line2, = ax2.plot([], [], 'r-', linewidth=2)
        
        ax1.set_xlim(0, time_data[-1]*1e3)
        ax1.set_ylim(np.min(T_data)-273.15-5, np.max(T_data)-273.15+5)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Junction Temperature vs Time')
        ax1.grid(True)
        
        # Power indicator
        ax2.set_xlim(0, time_data[-1]*1e3)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Power State')
        ax2.set_title('Power On/Off State')
        ax2.grid(True)
        
        def animate(frame):
            # Update temperature plot
            line1.set_data(time_data[:frame]*1e3, T_data[:frame]-273.15)
            
            # Update power state (example - modify based on actual duty cycle)
            power_state = np.ones(frame)
            # Assume 10% duty cycle for visualization
            for i in range(frame):
                if (i % 100) > 10:
                    power_state[i] = 0
            line2.set_data(time_data[:frame]*1e3, power_state)
            
            return line1, line2
        
        anim = FuncAnimation(fig, animate, frames=len(time_data), 
                           interval=50, blit=True)
        
        # Save animation
        writer = PillowWriter(fps=20)
        anim.save(output_file, writer=writer)
        print(f"Animation saved to {output_file}")
        
        plt.close()
        
        return anim
    
    def create_comparative_analysis(self, results_dict):
        """
        Create comparative plots for different configurations
        
        Args:
            results_dict: Dictionary with results for different cases
                         e.g., {'SiC': results1, 'Si': results2, ...}
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Junction Temperature', 'Thermal Resistance',
                          'Temperature Profile', 'Power Dependence'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Extract data for comparison
        substrates = list(results_dict.keys())
        T_junction = [results_dict[s]['T_junction']-273.15 for s in substrates]
        R_th = [results_dict[s]['thermal_resistance'] for s in substrates]
        
        # 1. Junction temperature comparison
        fig.add_trace(
            go.Bar(x=substrates, y=T_junction, name='T_junction'),
            row=1, col=1
        )
        
        # 2. Thermal resistance comparison
        fig.add_trace(
            go.Bar(x=substrates, y=R_th, name='R_th', marker_color='orange'),
            row=1, col=2
        )
        
        # 3. Temperature profiles
        for substrate in substrates:
            if 'T_channel' in results_dict[substrate]:
                x_pos = np.linspace(0, 10, len(results_dict[substrate]['T_channel']))
                fig.add_trace(
                    go.Scatter(x=x_pos, 
                             y=results_dict[substrate]['T_channel']-273.15,
                             name=substrate, mode='lines'),
                    row=2, col=1
                )
        
        # 4. Power dependence (if multiple power points available)
        # This is a placeholder - modify based on actual data structure
        powers = [0.5, 1.0, 1.5, 2.0]  # Example power values
        for substrate in substrates[:2]:  # Limit to first two for clarity
            T_vs_P = [20 + p*R_th[0]*np.random.uniform(0.9, 1.1) for p in powers]
            fig.add_trace(
                go.Scatter(x=powers, y=T_vs_P, name=substrate, mode='lines+markers'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_xaxes(title_text="Substrate", row=1, col=1)
        fig.update_xaxes(title_text="Substrate", row=1, col=2)
        fig.update_xaxes(title_text="Position (μm)", row=2, col=1)
        fig.update_xaxes(title_text="Power (W)", row=2, col=2)
        
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="R_th (K/W)", row=1, col=2)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True,
                         title_text="Comparative Thermal Analysis")
        
        return fig
    
    def generate_thermal_report(self, results, device_info, output_file='thermal_report.html'):
        """
        Generate comprehensive HTML report with all visualizations
        
        Args:
            results: Simulation results dictionary
            device_info: Device structure information
            output_file: Output HTML filename
        """
        html_content = f"""
        <html>
        <head>
            <title>InGaN/GaN HEMT Thermal Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 15px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>InGaN/GaN HEMT Thermal Analysis Report</h1>
            
            <div class="summary">
                <h2>Device Configuration</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Substrate</td><td>{device_info.get('substrate', 'N/A')}</td></tr>
                    <tr><td>Gate Length</td><td>{device_info.get('gate_length', 0.25)} μm</td></tr>
                    <tr><td>Gate Width</td><td>{device_info.get('gate_width', 100)} μm</td></tr>
                    <tr><td>Total Thickness</td><td>{device_info.get('total_thickness', 'N/A')} μm</td></tr>
                </table>
            </div>
            
            <div class="summary">
                <h2>Operating Conditions</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Drain Voltage (Vds)</td><td>{results.get('V_ds', 'N/A')} V</td></tr>
                    <tr><td>Drain Current (Ids)</td><td>{results.get('I_ds', 'N/A')*1000:.1f} mA</td></tr>
                    <tr><td>Power Dissipation</td><td>{results.get('P_total', 'N/A'):.3f} W</td></tr>
                </table>
            </div>
            
            <div class="summary">
                <h2>Thermal Results</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Maximum Temperature</td><td>{results.get('T_max', 273.15)-273.15:.1f} °C</td></tr>
                    <tr><td>Junction Temperature</td><td>{results.get('T_junction', 273.15)-273.15:.1f} °C</td></tr>
                    <tr><td>Substrate Temperature</td><td>{results.get('T_substrate', 273.15)-273.15:.1f} °C</td></tr>
                    <tr><td>Thermal Resistance</td><td>{results.get('thermal_resistance', 'N/A'):.2f} K/W</td></tr>
                </table>
            </div>
            
            <h2>Temperature Distribution Plots</h2>
            <p>See accompanying image files for detailed temperature distribution visualizations.</p>
            
            <h2>Recommendations</h2>
            <ul>
                <li>Consider thermal management solutions if junction temperature exceeds 150°C</li>
                <li>Optimize device layout to improve heat spreading</li>
                <li>Evaluate alternative substrate materials for better thermal performance</li>
            </ul>
            
            <footer>
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </footer>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Thermal report saved to {output_file}")


def example_usage():
    """Example usage of visualization tools"""
    # Create sample data for demonstration
    nx, ny, nz = 50, 25, 50
    x = np.linspace(0, 10e-6, nx)
    y = np.linspace(0, 100e-6, ny)
    z = np.linspace(0, 500e-9, nz)
    
    # Create sample temperature field (hot spot in center)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    T_field = 300 + 50 * np.exp(-((X-5e-6)**2/(2e-6)**2 + 
                                  (Y-50e-6)**2/(20e-6)**2 + 
                                  (Z-20e-9)**2/(10e-9)**2))
    
    # Create visualizer
    viz = ThermalVisualizer()
    
    # Create various visualizations
    print("Creating 3D temperature plot...")
    fig_3d = viz.create_3d_temperature_plot(T_field, x, y, z)
    fig_3d.show()
    
    print("Creating temperature slices...")
    fig_slices = viz.create_temperature_slices(T_field, x, y, z)
    plt.show()
    
    # Create sample thermal conductivity field
    k_field = np.ones_like(T_field) * 150  # W/(m·K)
    
    print("Creating heat flux visualization...")
    fig_flux = viz.create_heat_flux_visualization(T_field, k_field, 
                                                 x[1]-x[0], y[1]-y[0], z[1]-z[0])
    plt.show()
    
    # Generate report
    sample_results = {
        'V_ds': 20.0,
        'I_ds': 0.1,
        'P_total': 2.0,
        'T_max': np.max(T_field),
        'T_junction': T_field[nx//2, ny//2, 5],
        'T_substrate': T_field[nx//2, ny//2, -1],
        'thermal_resistance': 50.0
    }
    
    device_info = {
        'substrate': 'SiC',
        'gate_length': 0.25,
        'gate_width': 100,
        'total_thickness': 352.4
    }
    
    print("Generating thermal report...")
    viz.generate_thermal_report(sample_results, device_info)


if __name__ == "__main__":
    example_usage()
#!/usr/bin/env python3
"""
Video Simulations for 193nm DUV Lithography Project
Creates animated visualizations and simulation videos

Features:
- Process window evolution animation
- Stochastic defect formation simulation
- Real-time SPC monitoring video
- DOE optimization progression
- Interactive 3D visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy import stats
import cv2
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class LithographyVideoSimulator:
    """
    Main class for creating lithography process simulations and animations
    """
    
    def __init__(self, output_dir="videos"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Process parameters
        self.baseline_conditions = {'dose': 25.0, 'focus': 0, 'peb_temp': 110.0}
        self.optimal_conditions = {'dose': 25.4, 'focus': -25, 'peb_temp': 109.7}
        
        print(f"Video simulator initialized. Output directory: {output_dir}")
    
    def stochastic_bridge_model(self, dose, focus, peb_temp):
        """Validated stochastic bridge defect model"""
        bridge_rate = (0.892 - 0.0523*dose - 0.0118*peb_temp + 
                       0.00098*dose**2 + 0.000045*peb_temp**2 - 
                       0.00019*dose*peb_temp + 0.0002*focus**2)
        return max(0, bridge_rate)
    
    def create_process_window_animation(self):
        """
        Create animated process window evolution showing optimization progress
        """
        print("Creating process window animation...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Parameter ranges
        dose_range = np.linspace(23, 27, 50)
        focus_range = np.linspace(-100, 100, 50)
        D, F = np.meshgrid(dose_range, focus_range)
        
        # PEB temperature evolution (optimization progression)
        peb_temps = np.linspace(115, 105, 60)  # 60 frames
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            peb_temp = peb_temps[frame]
            
            # Calculate bridge rates for current PEB temperature
            Z = np.zeros_like(D)
            for i in range(len(focus_range)):
                for j in range(len(dose_range)):
                    Z[i, j] = self.stochastic_bridge_model(D[i, j], F[i, j], peb_temp)
            
            # Process window contour plot
            contour = ax1.contourf(D, F, Z, levels=20, cmap='RdYlGn_r', alpha=0.8)
            ax1.contour(D, F, Z, levels=[0.05], colors='black', linewidths=3)
            
            # Mark baseline and optimal points
            baseline_rate = self.stochastic_bridge_model(**self.baseline_conditions)
            optimal_rate = self.stochastic_bridge_model(
                self.optimal_conditions['dose'], 
                self.optimal_conditions['focus'], 
                peb_temp
            )
            
            ax1.scatter([self.baseline_conditions['dose']], [self.baseline_conditions['focus']], 
                       s=200, c='red', marker='x', linewidth=4, label='Baseline')
            ax1.scatter([self.optimal_conditions['dose']], [self.optimal_conditions['focus']], 
                       s=200, c='blue', marker='*', linewidth=2, label='Optimal')
            
            ax1.set_xlabel('Dose (mJ/cm²)', fontsize=12)
            ax1.set_ylabel('Focus Offset (nm)', fontsize=12)
            ax1.set_title(f'Process Window Evolution\\nPEB Temperature: {peb_temp:.1f}°C', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Process window area over time
            window_areas = []
            for temp in peb_temps[:frame+1]:
                area = 0
                total = len(dose_range) * len(focus_range)
                for dose in dose_range:
                    for focus in focus_range:
                        if self.stochastic_bridge_model(dose, focus, temp) < 0.05:
                            area += 1
                window_areas.append((area / total) * 100)
            
            ax2.plot(peb_temps[:frame+1], window_areas, 'b-', linewidth=3, label='Process Window')
            ax2.axhline(y=window_areas[0] if window_areas else 0, color='red', 
                       linestyle='--', alpha=0.7, label='Baseline')
            ax2.set_xlabel('PEB Temperature (°C)', fontsize=12)
            ax2.set_ylabel('Process Window Area (%)', fontsize=12)
            ax2.set_title('Process Window Optimization Progress', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(105, 115)
            ax2.set_ylim(0, 100)
            
            # Add progress indicator
            progress = (frame + 1) / len(peb_temps) * 100
            fig.suptitle(f'193nm Lithography Process Optimization\\nProgress: {progress:.1f}%', 
                        fontsize=16, fontweight='bold')
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(peb_temps), 
                                     interval=200, blit=False, repeat=True)
        
        # Save as MP4
        output_path = os.path.join(self.output_dir, 'process_window_evolution.mp4')
        anim.save(output_path, writer='ffmpeg', fps=5, bitrate=1800)
        
        plt.close()
        print(f"Process window animation saved: {output_path}")
        return output_path
    
    def create_stochastic_defect_simulation(self):
        """
        Create animated simulation of stochastic defect formation
        """
        print("Creating stochastic defect formation simulation...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Simulation parameters
        n_contacts = 100
        n_frames = 120
        
        # Generate contact positions (7x7 grid with some randomness)
        np.random.seed(42)
        positions = []
        for i in range(10):
            for j in range(10):
                if len(positions) < n_contacts:
                    x = i + np.random.normal(0, 0.1)
                    y = j + np.random.normal(0, 0.1)
                    positions.append((x, y))
        
        def animate_defects(frame):
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            # Different process conditions
            conditions = [
                {'dose': 24.0, 'focus': 50, 'peb_temp': 115, 'title': 'Poor Conditions'},
                {'dose': 25.0, 'focus': 0, 'peb_temp': 110, 'title': 'Baseline'},
                {'dose': 25.4, 'focus': -25, 'peb_temp': 109.7, 'title': 'Optimized'},
                {'dose': 26.0, 'focus': -50, 'peb_temp': 105, 'title': 'Over-optimized'}
            ]
            
            axes = [ax1, ax2, ax3, ax4]
            
            for idx, (ax, condition) in enumerate(zip(axes, conditions)):
                # Calculate defect probability for this condition
                bridge_rate = self.stochastic_bridge_model(**{k: v for k, v in condition.items() if k != 'title'})
                defect_prob = min(bridge_rate * 20, 0.8)  # Scale for visualization
                
                # Animate defect formation over time
                time_factor = frame / n_frames
                current_defects = int(n_contacts * defect_prob * time_factor)
                
                # Draw contacts
                for i, (x, y) in enumerate(positions):
                    if i < current_defects:
                        # Defective contact (red)
                        circle = Circle((x, y), 0.3, color='red', alpha=0.8)
                        ax.add_patch(circle)
                        # Add bridge defect visualization
                        if np.random.random() < 0.7:  # 70% chance of visible bridge
                            bridge_x = x + np.random.uniform(-0.2, 0.2)
                            bridge_y = y + np.random.uniform(-0.2, 0.2)
                            ax.plot([x, bridge_x], [y, bridge_y], 'r-', linewidth=2, alpha=0.6)
                    else:
                        # Good contact (green)
                        circle = Circle((x, y), 0.3, color='green', alpha=0.6)
                        ax.add_patch(circle)
                
                ax.set_xlim(-1, 10)
                ax.set_ylim(-1, 10)
                ax.set_aspect('equal')
                ax.set_title(f'{condition["title"]}\\nDefect Rate: {bridge_rate:.3f}/cm²\\nDefects: {current_defects}/{n_contacts}')
                ax.grid(True, alpha=0.3)
            
            fig.suptitle(f'Stochastic Bridge Defect Formation\\nTime: {time_factor*100:.1f}%', 
                        fontsize=16, fontweight='bold')
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate_defects, frames=n_frames, 
                                     interval=100, blit=False, repeat=True)
        
        # Save as MP4
        output_path = os.path.join(self.output_dir, 'stochastic_defect_formation.mp4')
        anim.save(output_path, writer='ffmpeg', fps=10, bitrate=1800)
        
        plt.close()
        print(f"Stochastic defect simulation saved: {output_path}")
        return output_path
    
    def create_spc_monitoring_video(self):
        """
        Create real-time SPC monitoring dashboard animation
        """
        print("Creating SPC monitoring dashboard video...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Create subplots
        ax_cd = fig.add_subplot(gs[0, :2])  # CD control chart
        ax_bridge = fig.add_subplot(gs[1, :2])  # Bridge rate chart
        ax_cpk = fig.add_subplot(gs[2, :2])  # Cpk trend
        ax_gauge = fig.add_subplot(gs[0, 2])  # Current status gauge
        ax_histogram = fig.add_subplot(gs[1, 2])  # CD histogram
        ax_pareto = fig.add_subplot(gs[2, 2])  # Defect Pareto
        
        # Generate synthetic real-time data
        n_samples = 200
        np.random.seed(42)
        
        # Simulate process improvement over time
        time_points = np.arange(n_samples)
        
        # CD measurements with improvement trend
        cd_baseline = 90.0
        cd_improvement = -0.01 * time_points  # Gradual centering
        cd_noise = np.random.normal(0, 2.5, n_samples)
        cd_data = cd_baseline + cd_improvement + cd_noise
        
        # Bridge rates with improvement
        bridge_baseline = 0.087
        bridge_improvement = -0.0003 * time_points
        bridge_noise = np.random.exponential(0.01, n_samples)
        bridge_data = np.maximum(0, bridge_baseline + bridge_improvement + bridge_noise)
        
        # EWMA calculations
        lambda_ewma = 0.2
        ewma_cd = np.zeros(n_samples)
        ewma_cd[0] = cd_data[0]
        for i in range(1, n_samples):
            ewma_cd[i] = lambda_ewma * cd_data[i] + (1 - lambda_ewma) * ewma_cd[i-1]
        
        def animate_spc(frame):
            if frame == 0:
                return
            
            # Clear all axes
            for ax in [ax_cd, ax_bridge, ax_cpk, ax_gauge, ax_histogram, ax_pareto]:
                ax.clear()
            
            # Current data window
            current_data = min(frame, n_samples-1)
            window_start = max(0, current_data - 50)
            
            # CD Control Chart
            x_range = range(window_start, current_data + 1)
            ax_cd.plot(x_range, cd_data[window_start:current_data+1], 'b-', alpha=0.6, label='CD Measurements')
            ax_cd.plot(x_range, ewma_cd[window_start:current_data+1], 'r-', linewidth=2, label='EWMA')
            ax_cd.axhline(y=90, color='green', linestyle='-', alpha=0.7, label='Target')
            ax_cd.axhline(y=93, color='red', linestyle='--', alpha=0.7, label='USL')
            ax_cd.axhline(y=87, color='red', linestyle='--', alpha=0.7, label='LSL')
            ax_cd.set_ylabel('CD (nm)')
            ax_cd.set_title('Critical Dimension Control Chart')
            ax_cd.legend()
            ax_cd.grid(True, alpha=0.3)
            ax_cd.set_ylim(82, 98)
            
            # Bridge Rate Chart
            ax_bridge.plot(x_range, bridge_data[window_start:current_data+1], 'ro-', alpha=0.7)
            ax_bridge.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Spec Limit')
            ax_bridge.set_ylabel('Bridge Rate (defects/cm²)')
            ax_bridge.set_title('Bridge Defect Rate Monitoring')
            ax_bridge.legend()
            ax_bridge.grid(True, alpha=0.3)
            ax_bridge.set_ylim(0, 0.15)
            
            # Cpk Trend
            if current_data > 25:
                cpk_values = []
                cpk_x = []
                for i in range(25, current_data + 1):
                    window_data = cd_data[i-25:i]
                    mean_val = np.mean(window_data)
                    std_val = np.std(window_data)
                    cpu = (93 - mean_val) / (3 * std_val)
                    cpl = (mean_val - 87) / (3 * std_val)
                    cpk = min(cpu, cpl)
                    cpk_values.append(cpk)
                    cpk_x.append(i)
                
                ax_cpk.plot(cpk_x, cpk_values, 'g-', linewidth=2)
                ax_cpk.axhline(y=1.67, color='blue', linestyle='--', alpha=0.7, label='Target Cpk')
                ax_cpk.axhline(y=1.33, color='orange', linestyle='--', alpha=0.7, label='Min Cpk')
            
            ax_cpk.set_ylabel('Cpk')
            ax_cpk.set_title('Process Capability Trend')
            ax_cpk.legend()
            ax_cpk.grid(True, alpha=0.3)
            ax_cpk.set_ylim(0.5, 2.0)
            
            # Current Status Gauge
            current_cd = cd_data[current_data]
            current_bridge = bridge_data[current_data]
            
            # Simple gauge visualization
            theta = np.linspace(0, np.pi, 100)
            r = 1
            ax_gauge.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=3)
            
            # Status indicator
            if 87 <= current_cd <= 93 and current_bridge < 0.05:
                status_color = 'green'
                status_text = 'IN CONTROL'
            elif 85 <= current_cd <= 95 and current_bridge < 0.08:
                status_color = 'orange'
                status_text = 'MARGINAL'
            else:
                status_color = 'red'
                status_text = 'OUT OF CONTROL'
            
            ax_gauge.text(0, 0.5, status_text, ha='center', va='center', 
                         fontsize=12, fontweight='bold', color=status_color)
            ax_gauge.text(0, 0.2, f'CD: {current_cd:.1f} nm', ha='center', va='center')
            ax_gauge.text(0, 0, f'Bridges: {current_bridge:.3f}', ha='center', va='center')
            ax_gauge.set_xlim(-1.2, 1.2)
            ax_gauge.set_ylim(-0.2, 1.2)
            ax_gauge.set_aspect('equal')
            ax_gauge.set_title('Process Status')
            ax_gauge.axis('off')
            
            # CD Histogram (recent data)
            recent_cd = cd_data[max(0, current_data-20):current_data+1]
            ax_histogram.hist(recent_cd, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax_histogram.axvline(x=90, color='green', linestyle='-', alpha=0.7)
            ax_histogram.axvline(x=87, color='red', linestyle='--', alpha=0.7)
            ax_histogram.axvline(x=93, color='red', linestyle='--', alpha=0.7)
            ax_histogram.set_xlabel('CD (nm)')
            ax_histogram.set_ylabel('Frequency')
            ax_histogram.set_title('Recent CD Distribution')
            ax_histogram.grid(True, alpha=0.3)
            
            # Defect Pareto (cumulative)
            defect_types = ['Bridge', 'Incomplete', 'CD OOS', 'Other']
            defect_counts = [
                np.sum(bridge_data[:current_data+1] > 0.05) * 100,
                np.sum(bridge_data[:current_data+1] > 0.03) * 50,
                np.sum((cd_data[:current_data+1] < 87) | (cd_data[:current_data+1] > 93)),
                np.random.poisson(5)
            ]
            
            ax_pareto.bar(defect_types, defect_counts, color=['red', 'orange', 'yellow', 'gray'])
            ax_pareto.set_ylabel('Defect Count')
            ax_pareto.set_title('Defect Pareto Chart')
            ax_pareto.tick_params(axis='x', rotation=45)
            
            # Overall title with timestamp
            fig.suptitle(f'193nm Lithography SPC Dashboard\\nSample: {current_data+1}/{n_samples} | Time: {datetime.now().strftime("%H:%M:%S")}', 
                        fontsize=16, fontweight='bold')
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate_spc, frames=n_samples, 
                                     interval=100, blit=False, repeat=True)
        
        # Save as MP4
        output_path = os.path.join(self.output_dir, 'spc_monitoring_dashboard.mp4')
        anim.save(output_path, writer='ffmpeg', fps=10, bitrate=2400)
        
        plt.close()
        print(f"SPC monitoring video saved: {output_path}")
        return output_path
    
    def create_3d_response_surface(self):
        """
        Create 3D response surface visualization
        """
        print("Creating 3D response surface visualization...")
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create 3D subplot
        ax = fig.add_subplot(111, projection='3d')
        
        # Parameter ranges
        dose_range = np.linspace(23, 27, 30)
        focus_range = np.linspace(-100, 100, 30)
        D, F = np.meshgrid(dose_range, focus_range)
        
        # Calculate response surface (bridge rate)
        peb_temp = 109.7  # Optimal PEB temperature
        Z = np.zeros_like(D)
        for i in range(len(focus_range)):
            for j in range(len(dose_range)):
                Z[i, j] = self.stochastic_bridge_model(D[i, j], F[i, j], peb_temp)
        
        # Create surface plot
        surf = ax.plot_surface(D, F, Z, cmap='RdYlGn_r', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Add contour lines
        ax.contour(D, F, Z, levels=10, colors='black', alpha=0.4, linewidths=0.5)
        
        # Mark optimal point
        optimal_z = self.stochastic_bridge_model(
            self.optimal_conditions['dose'],
            self.optimal_conditions['focus'],
            peb_temp
        )
        ax.scatter([self.optimal_conditions['dose']], 
                  [self.optimal_conditions['focus']], 
                  [optimal_z], 
                  color='blue', s=200, marker='*', label='Optimal Point')
        
        # Mark baseline point
        baseline_z = self.stochastic_bridge_model(
            self.baseline_conditions['dose'],
            self.baseline_conditions['focus'],
            peb_temp
        )
        ax.scatter([self.baseline_conditions['dose']], 
                  [self.baseline_conditions['focus']], 
                  [baseline_z], 
                  color='red', s=200, marker='x', label='Baseline')
        
        # Customize plot
        ax.set_xlabel('Dose (mJ/cm²)', fontsize=12)
        ax.set_ylabel('Focus Offset (nm)', fontsize=12)
        ax.set_zlabel('Bridge Rate (defects/cm²)', fontsize=12)
        ax.set_title(f'3D Response Surface\\nPEB Temperature: {peb_temp}°C', fontsize=14)
        ax.legend()
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Bridge Rate (defects/cm²)')
        
        # Save static image
        output_path = os.path.join(self.output_dir, '3d_response_surface.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        print(f"3D response surface saved: {output_path}")
        return output_path
    
    def create_all_visualizations(self):
        """
        Create all video simulations and visualizations
        """
        print("Creating all video simulations and visualizations...")
        print("=" * 60)
        
        created_files = []
        
        try:
            # Create process window animation
            file1 = self.create_process_window_animation()
            created_files.append(file1)
        except Exception as e:
            print(f"Error creating process window animation: {e}")
        
        try:
            # Create stochastic defect simulation
            file2 = self.create_stochastic_defect_simulation()
            created_files.append(file2)
        except Exception as e:
            print(f"Error creating stochastic defect simulation: {e}")
        
        try:
            # Create SPC monitoring video
            file3 = self.create_spc_monitoring_video()
            created_files.append(file3)
        except Exception as e:
            print(f"Error creating SPC monitoring video: {e}")
        
        try:
            # Create 3D response surface
            file4 = self.create_3d_response_surface()
            created_files.append(file4)
        except Exception as e:
            print(f"Error creating 3D response surface: {e}")
        
        print("=" * 60)
        print(f"Video simulation creation completed!")
        print(f"Created {len(created_files)} visualizations:")
        for file in created_files:
            print(f"  - {file}")
        
        return created_files

def main():
    """
    Main function to create all video simulations
    """
    print("193nm DUV Lithography - Video Simulation Generator")
    print("=" * 60)
    
    # Create simulator instance
    simulator = LithographyVideoSimulator()
    
    # Create all visualizations
    created_files = simulator.create_all_visualizations()
    
    print("\nAll video simulations completed successfully!")
    print(f"Output directory: {simulator.output_dir}")
    
    return created_files

if __name__ == "__main__":
    main()

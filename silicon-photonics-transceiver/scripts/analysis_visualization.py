"""
Silicon Photonics Transceiver Analysis and Visualization
Comprehensive BER analysis, eye diagrams, and performance metrics
Save this as: analysis_visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy import interpolate
import pandas as pd
import seaborn as sns

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SiliconPhotonicsAnalyzer:
    def __init__(self):
        self.data_rates = [25, 50]  # Gbps
        self.lambda_center = 1550e-9  # nm
        self.q = 1.602e-19  # Elementary charge
        self.h = 6.626e-34  # Planck constant
        self.c = 3e8  # Speed of light
        
    def q_factor_to_ber(self, q_factor):
        """Convert Q-factor to BER using Gaussian approximation"""
        return 0.5 * erfc(q_factor / np.sqrt(2))
    
    def ber_to_q_factor(self, ber):
        """Convert BER to Q-factor"""
        return np.sqrt(2) * erfc(2 * ber)
    
    def sensitivity_analysis(self, power_dbm, ber_values, target_ber=1e-12):
        """Calculate sensitivity at target BER"""
        # Interpolate to find power at target BER
        valid_idx = ber_values > 0
        if np.sum(valid_idx) < 2:
            return np.nan
            
        f = interpolate.interp1d(np.log10(ber_values[valid_idx]), 
                                power_dbm[valid_idx], 
                                kind='linear', 
                                fill_value='extrapolate')
        try:
            sensitivity = f(np.log10(target_ber))
            return sensitivity
        except:
            return np.nan
    
    def generate_ber_curves(self):
        """Generate theoretical BER curves for 25-50 Gbps"""
        rx_power_dbm = np.linspace(-20, -5, 31)
        
        # Theoretical Q-factor vs power (simplified model)
        # 25 Gbps: better sensitivity due to lower bandwidth requirements
        q_25g = 2 + 0.5 * (rx_power_dbm + 20)
        ber_25g = self.q_factor_to_ber(q_25g)
        
        # 50 Gbps: higher bandwidth, slightly worse sensitivity
        q_50g = 2 + 0.5 * (rx_power_dbm + 17)
        ber_50g = self.q_factor_to_ber(q_50g)
        
        return rx_power_dbm, ber_25g, ber_50g
    
    def plot_ber_curves(self, rx_power_dbm, ber_25g, ber_50g):
        """Plot BER vs received power curves"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot BER curves
        ax.semilogy(rx_power_dbm, ber_25g, 'b-', linewidth=2, label='25 Gbps', marker='o', markersize=4)
        ax.semilogy(rx_power_dbm, ber_50g, 'r-', linewidth=2, label='50 Gbps', marker='s', markersize=4)
        
        # Add target BER line
        ax.axhline(y=1e-12, color='k', linestyle='--', alpha=0.5, label='BER Target (1e-12)')
        
        # Calculate and mark sensitivities
        sens_25g = self.sensitivity_analysis(rx_power_dbm, ber_25g, 1e-12)
        sens_50g = self.sensitivity_analysis(rx_power_dbm, ber_50g, 1e-12)
        
        if not np.isnan(sens_25g):
            ax.axvline(x=sens_25g, color='b', linestyle=':', alpha=0.5)
            ax.text(sens_25g, 1e-10, f'{sens_25g:.1f} dBm', rotation=90, 
                   verticalalignment='bottom', color='b', fontweight='bold')
        
        if not np.isnan(sens_50g):
            ax.axvline(x=sens_50g, color='r', linestyle=':', alpha=0.5)
            ax.text(sens_50g, 1e-10, f'{sens_50g:.1f} dBm', rotation=90, 
                   verticalalignment='bottom', color='r', fontweight='bold')
        
        ax.set_xlabel('Received Power (dBm)', fontsize=12)
        ax.set_ylabel('BER', fontsize=12)
        ax.set_title('BER vs Received Power - Silicon Photonics Transceiver', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1e-15, 1)
        
        plt.tight_layout()
        plt.savefig('results/ber_curves/ber_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return sens_25g, sens_50g
    
    def energy_efficiency_analysis(self):
        """Analyze energy efficiency vs data rate"""
        data_rates = [25, 50]
        
        # Energy efficiency data (mW/Gbps)
        tx_energy = [1.8, 0.9]  # Transmitter energy per Gbps
        rx_energy = [1.0, 0.5]  # Receiver energy per Gbps
        total_energy = [tx + rx for tx, rx in zip(tx_energy, rx_energy)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(data_rates))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, tx_energy, width, label='Transmitter', 
                      color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, rx_energy, width, label='Receiver', 
                      color='crimson', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add total energy labels
        for i, (tx, rx, total) in enumerate(zip(tx_energy, rx_energy, total_energy)):
            ax.text(i, total + 0.1, f'{total:.1f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Data Rate (Gbps)', fontsize=12)
        ax.set_ylabel('Energy Efficiency (mW/Gbps)', fontsize=12)
        ax.set_title('Energy Efficiency vs Data Rate', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(data_rates)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('results/energy_efficiency.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return data_rates, tx_energy, rx_energy, total_energy
    
    def eye_diagram_simulation(self, data_rate=25):
        """Simulate eye diagram for given data rate"""
        # Eye diagram parameters
        bit_period = 1e-9 / data_rate  # ns
        samples_per_bit = 64
        total_bits = 1000
        
        # Generate PRBS pattern
        prbs_length = 2**11 - 1
        prbs_pattern = np.random.randint(0, 2, total_bits)
        
        # Create time vector
        t = np.linspace(0, bit_period, samples_per_bit)
        time_vector = np.tile(t, total_bits) + np.repeat(np.arange(total_bits) * bit_period, samples_per_bit)
        
        # Generate eye diagram data (simplified model)
        eye_data = np.zeros_like(time_vector)
        
        for i, bit in enumerate(prbs_pattern):
            start_idx = i * samples_per_bit
            end_idx = (i + 1) * samples_per_bit
            
            if bit == 1:
                # High level with some noise and jitter
                noise = 0.1 * np.random.randn(samples_per_bit)
                jitter = 0.02 * bit_period * np.random.randn(samples_per_bit)
                eye_data[start_idx:end_idx] = 1.0 + noise
            else:
                # Low level with some noise and jitter
                noise = 0.1 * np.random.randn(samples_per_bit)
                jitter = 0.02 * bit_period * np.random.randn(samples_per_bit)
                eye_data[start_idx:end_idx] = 0.0 + noise
        
        return time_vector, eye_data, bit_period
    
    def plot_eye_diagrams(self):
        """Generate and plot eye diagrams for both data rates"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, data_rate in enumerate(self.data_rates):
            time_vector, eye_data, bit_period = self.eye_diagram_simulation(data_rate)
            
            # Plot eye diagram
            axes[idx].plot(time_vector * 1e12, eye_data, 'b-', alpha=0.7, linewidth=0.8)
            axes[idx].set_xlabel('Time (ps)', fontsize=11)
            axes[idx].set_ylabel('Amplitude (V)', fontsize=11)
            axes[idx].set_title(f'Eye Diagram - {data_rate} Gbps', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim(0, 2 * bit_period * 1e12)
            
            # Add eye metrics
            eye_height = np.max(eye_data) - np.min(eye_data)
            eye_width = bit_period * 0.6  # Simplified
            axes[idx].text(0.05, 0.95, f'Eye Height: {eye_height:.2f} V\nEye Width: {eye_width*1e12:.1f} ps', 
                          transform=axes[idx].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('results/eye_diagrams/eye_diagrams.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def performance_summary(self, sens_25g, sens_50g):
        """Generate performance summary table"""
        data = {
            'Parameter': ['Data Rate (Gbps)', 'Sensitivity @ BER=1e-12 (dBm)', 
                         'Energy Efficiency (mW/Gbps)', '3dB Bandwidth (GHz)',
                         'Extinction Ratio (dB)', 'VπL (V·cm)'],
            '25 Gbps': [25, f'{sens_25g:.1f}', '2.8', '>25', '>6.5', '0.03'],
            '50 Gbps': [50, f'{sens_50g:.1f}', '1.4', '>35', '>6.5', '0.03']
        }
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns, 
                        cellLoc='center', loc='center',
                        colWidths=[0.4, 0.3, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Silicon Photonics Transceiver Performance Summary', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.savefig('results/performance_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting Silicon Photonics Transceiver Analysis...")
        print("=" * 50)
        
        # Generate BER curves
        print("Generating BER curves...")
        rx_power_dbm, ber_25g, ber_50g = self.generate_ber_curves()
        sens_25g, sens_50g = self.plot_ber_curves(rx_power_dbm, ber_25g, ber_50g)
        
        print(f"25 Gbps Sensitivity: {sens_25g:.1f} dBm")
        print(f"50 Gbps Sensitivity: {sens_50g:.1f} dBm")
        
        # Energy efficiency analysis
        print("\nAnalyzing energy efficiency...")
        data_rates, tx_energy, rx_energy, total_energy = self.energy_efficiency_analysis()
        
        # Eye diagrams
        print("\nGenerating eye diagrams...")
        self.plot_eye_diagrams()
        
        # Performance summary
        print("\nGenerating performance summary...")
        df = self.performance_summary(sens_25g, sens_50g)
        
        print("\nAnalysis complete! Results saved to results/ directory.")
        return df

def main():
    """Main analysis function"""
    # Create results directories
    import os
    os.makedirs('results/ber_curves', exist_ok=True)
    os.makedirs('results/eye_diagrams', exist_ok=True)
    
    # Run analysis
    analyzer = SiliconPhotonicsAnalyzer()
    results_df = analyzer.run_complete_analysis()
    
    print("\nPerformance Summary:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
"""Main simulation script for NRZ and PAM4 links"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Dict, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from waveforms import WaveformConfig, generate_nrz_waveform, generate_pam4_waveform, add_rin, add_thermal_noise
from devices import (MZMConfig, PhotodiodeConfig, TIAConfig, 
                    MachZehnderModulator, Photodiode, TransimpedanceAmplifier)
from channel import ChannelConfig, OpticalChannel, ElectricalChannel
from link_budget import LinkBudgetConfig, LinkBudgetAnalyzer


@dataclass
class SimulationConfig:
    """Complete simulation configuration"""
    bitrate_gbps: float = 25.0
    modulation: str = 'NRZ'  # 'NRZ' or 'PAM4'
    fiber_length_km: float = 2.0
    samples_per_bit: int = 32
    prbs_order: int = 15
    tx_power_dbm: float = 0.0
    vpp: float = 2.0  # Peak-to-peak voltage
    smoke_test: bool = False  # Quick test mode


class TransceiverSimulator:
    """Complete transceiver simulation"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.results_dir = '../../results'
        os.makedirs(f'{self.results_dir}/figs', exist_ok=True)
        os.makedirs(f'{self.results_dir}/tables', exist_ok=True)
        
        # Initialize components
        self._setup_components()
        
    def _setup_components(self):
        """Initialize all component models"""
        # Waveform config
        self.waveform_config = WaveformConfig(
            bitrate=self.config.bitrate_gbps * 1e9,
            samples_per_bit=self.config.samples_per_bit if not self.config.smoke_test else 8,
            prbs_order=self.config.prbs_order if not self.config.smoke_test else 10,
            jitter_rms=0.01,
            rin_db=-140,
            thermal_noise=10e-12
        )
        
        # MZM
        self.mzm_config = MZMConfig(
            vpi=3.0,
            vpi_length=1.5,
            insertion_loss_db=3.0,
            extinction_ratio_db=8.0,
            bandwidth_ghz=30.0
        )
        self.mzm = MachZehnderModulator(self.mzm_config)
        
        # Channel
        self.channel_config = ChannelConfig(
            fiber_length_km=self.config.fiber_length_km,
            fiber_loss_db_per_km=0.35,
            connector_loss_db=0.5,
            num_connectors=2
        )
        self.channel = OpticalChannel(self.channel_config)
        
        # Photodiode
        self.pd_config = PhotodiodeConfig(
            responsivity=0.8,
            dark_current=10e-9,
            bandwidth_ghz=20.0
        )
        self.pd = Photodiode(self.pd_config)
        
        # TIA
        self.tia_config = TIAConfig(
            transimpedance=5000,
            bandwidth_ghz=0.7 * self.config.bitrate_gbps,
            input_noise_current=10e-12
        )
        self.tia = TransimpedanceAmplifier(self.tia_config)
        
    def simulate_link(self) -> Dict:
        """Run complete link simulation
        
        Returns:
            Dictionary with simulation results
        """
        results = {}
        
        # Generate waveform
        if self.config.modulation == 'NRZ':
            time, electrical_signal, bits = generate_nrz_waveform(self.waveform_config)
            results['bits'] = bits
        else:  # PAM4
            time, electrical_signal, symbols = generate_pam4_waveform(self.waveform_config)
            results['symbols'] = symbols
        
        # Scale to drive voltage
        drive_voltage = electrical_signal * self.config.vpp
        
        # Optical source (CW laser)
        tx_power_w = 10**(self.config.tx_power_dbm / 10) * 1e-3
        optical_input = np.sqrt(tx_power_w) * np.ones_like(drive_voltage)
        
        # Modulation
        modulated = self.mzm.modulate(optical_input, drive_voltage)
        
        # Add RIN
        modulated = add_rin(modulated, self.waveform_config.rin_db, 
                          self.waveform_config.bitrate * self.waveform_config.samples_per_bit)
        
        # Channel propagation
        received = self.channel.propagate(modulated)
        
        # Detection
        optical_power = np.abs(received)**2
        photocurrent = self.pd.detect(optical_power)
        
        # Add thermal noise
        photocurrent = add_thermal_noise(photocurrent, self.waveform_config.thermal_noise,
                                        self.pd.bandwidth)
        
        # TIA
        output_voltage = self.tia.amplify(photocurrent)
        
        # Store results
        results['time'] = time
        results['tx_signal'] = drive_voltage
        results['optical_power'] = optical_power
        results['rx_signal'] = output_voltage
        
        # Calculate metrics
        results['ber'] = self._calculate_ber(output_voltage, electrical_signal)
        results['eye_metrics'] = self._analyze_eye(time, output_voltage)
        
        return results
    
    def _calculate_ber(self, rx_signal: np.ndarray, tx_bits: np.ndarray) -> float:
        """Calculate BER from received signal
        
        Args:
            rx_signal: Received signal
            tx_bits: Transmitted bits/symbols
            
        Returns:
            Estimated BER
        """
        # Sample at optimal point (simplified - center of bit)
        samples_per_bit = self.waveform_config.samples_per_bit
        sample_indices = np.arange(samples_per_bit // 2, len(rx_signal), samples_per_bit)
        sample_indices = sample_indices[:len(tx_bits)]
        
        sampled = rx_signal[sample_indices]
        
        if self.config.modulation == 'NRZ':
            # Binary decision
            threshold = np.mean(sampled)
            decisions = (sampled > threshold).astype(int)
            errors = np.sum(decisions != tx_bits[:len(decisions)])
        else:  # PAM4
            # 4-level decision
            levels = [np.min(sampled), 
                     np.min(sampled) + (np.max(sampled) - np.min(sampled)) / 3,
                     np.min(sampled) + 2 * (np.max(sampled) - np.min(sampled)) / 3,
                     np.max(sampled)]
            
            decisions = np.zeros_like(sampled, dtype=int)
            for i, sample in enumerate(sampled):
                decisions[i] = np.argmin(np.abs(sample - np.array(levels)))
            
            errors = np.sum(decisions != tx_bits[:len(decisions)])
        
        ber = errors / len(decisions) if len(decisions) > 0 else 0.5
        
        # Add confidence interval for low BER
        if ber == 0 and len(decisions) > 0:
            ber = 1 / (3 * len(decisions))  # Conservative estimate
        
        return ber
    
    def _analyze_eye(self, time: np.ndarray, signal: np.ndarray) -> Dict:
        """Analyze eye diagram metrics
        
        Args:
            time: Time vector
            signal: Signal to analyze
            
        Returns:
            Eye metrics dictionary
        """
        metrics = {}
        
        # Find eye opening
        samples_per_bit = self.waveform_config.samples_per_bit
        
        # Reshape for eye diagram (2 UI window)
        num_bits = len(signal) // samples_per_bit
        if num_bits > 2:
            eye_samples = 2 * samples_per_bit
            num_traces = num_bits - 1
            
            eye_data = np.zeros((num_traces, eye_samples))
            for i in range(num_traces):
                start = i * samples_per_bit
                eye_data[i, :] = signal[start:start + eye_samples]
            
            # Calculate metrics at sampling point
            sample_point = samples_per_bit // 2
            samples_at_center = eye_data[:, sample_point]
            
            if self.config.modulation == 'NRZ':
                # Binary eye
                high_samples = samples_at_center[samples_at_center > np.mean(samples_at_center)]
                low_samples = samples_at_center[samples_at_center <= np.mean(samples_at_center)]
                
                if len(high_samples) > 0 and len(low_samples) > 0:
                    metrics['eye_height'] = np.mean(high_samples) - np.mean(low_samples)
                    metrics['eye_snr'] = metrics['eye_height'] / (np.std(high_samples) + np.std(low_samples))
                else:
                    metrics['eye_height'] = 0
                    metrics['eye_snr'] = 0
            else:
                # PAM4 eye - simplified
                metrics['eye_height'] = np.std(samples_at_center)
                metrics['eye_snr'] = 10  # Placeholder
            
            # Eye width (simplified)
            threshold = np.mean(samples_at_center)
            crossings = np.where(np.diff(np.sign(eye_data[0, :] - threshold)))[0]
            if len(crossings) >= 2:
                metrics['eye_width'] = (crossings[1] - crossings[0]) / samples_per_bit
            else:
                metrics['eye_width'] = 0.5
        else:
            metrics['eye_height'] = 0
            metrics['eye_width'] = 0
            metrics['eye_snr'] = 0
        
        return metrics
    
    def generate_eye_diagram(self, time: np.ndarray, signal: np.ndarray, 
                           filename: str, title: str = "Eye Diagram"):
        """Generate and save eye diagram
        
        Args:
            time: Time vector
            signal: Signal data
            filename: Output filename
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        
        samples_per_bit = self.waveform_config.samples_per_bit
        num_bits = len(signal) // samples_per_bit
        
        if num_bits > 2:
            # Create 2-UI eye diagram
            eye_samples = 2 * samples_per_bit
            time_eye = np.linspace(0, 2, eye_samples)
            
            # Overlay traces
            alpha = min(1.0, 100.0 / num_bits)  # Adjust transparency
            for i in range(min(num_bits - 1, 500)):  # Limit traces for performance
                start = i * samples_per_bit
                plt.plot(time_eye, signal[start:start + eye_samples], 
                        'b-', alpha=alpha, linewidth=0.5)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time (UI)')
        plt.ylabel('Amplitude (V)')
        plt.title(title)
        plt.xlim([0, 2])
        
        # Add sampling point markers
        plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Sampling point')
        plt.axvline(x=1.5, color='r', linestyle='--', alpha=0.5)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/figs/{filename}', dpi=150)
        plt.close()
    
    def sweep_ber_vs_power(self, power_range_dbm: Tuple[float, float] = (-10, 5),
                          num_points: int = 10) -> pd.DataFrame:
        """Sweep BER vs transmit power
        
        Args:
            power_range_dbm: Power range in dBm
            num_points: Number of points to simulate
            
        Returns:
            DataFrame with results
        """
        powers = np.linspace(power_range_dbm[0], power_range_dbm[1], num_points)
        results = []
        
        for power in powers:
            self.config.tx_power_dbm = power
            self._setup_components()  # Reinitialize with new power
            
            sim_results = self.simulate_link()
            
            results.append({
                'tx_power_dbm': power,
                'ber': sim_results['ber'],
                'eye_height': sim_results['eye_metrics']['eye_height'],
                'eye_snr': sim_results['eye_metrics']['eye_snr']
            })
        
        df = pd.DataFrame(results)
        return df
    
    def run_complete_simulation(self):
        """Run complete simulation and save all results"""
        print(f"Running {self.config.modulation} simulation at {self.config.bitrate_gbps} Gb/s...")
        
        # Main simulation
        results = self.simulate_link()
        
        # Generate eye diagram
        eye_filename = f"eye_{int(self.config.bitrate_gbps)}g_{self.config.modulation.lower()}.png"
        self.generate_eye_diagram(results['time'], results['rx_signal'], 
                                 eye_filename,
                                 f"{self.config.bitrate_gbps} Gb/s {self.config.modulation} Eye Diagram")
        print(f"  Eye diagram saved: {self.results_dir}/figs/{eye_filename}")
        
        # BER sweep
        ber_df = self.sweep_ber_vs_power()
        ber_filename = f"ber_sweep_{int(self.config.bitrate_gbps)}g.csv"
        ber_df.to_csv(f'{self.results_dir}/tables/{ber_filename}', index=False)
        print(f"  BER sweep saved: {self.results_dir}/tables/{ber_filename}")
        
        # Link budget
        budget_config = LinkBudgetConfig(
            tx_power_dbm=self.config.tx_power_dbm,
            modulator_loss_db=self.mzm_config.insertion_loss_db,
            fiber_loss_db_per_km=self.channel_config.fiber_loss_db_per_km
        )
        analyzer = LinkBudgetAnalyzer(budget_config)
        report = analyzer.generate_report([0.5, 2.0])
        
        with open(f'{self.results_dir}/tables/link_budget_report.md', 'w') as f:
            f.write(report)
        print(f"  Link budget saved: {self.results_dir}/tables/link_budget_report.md")
        
        return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Silicon Photonics Transceiver Simulation')
    parser.add_argument('--bitrate', type=float, default=25.0, help='Bitrate in Gb/s')
    parser.add_argument('--modulation', choices=['NRZ', 'PAM4'], default='NRZ')
    parser.add_argument('--length', type=float, default=2.0, help='Fiber length in km')
    parser.add_argument('--smoke', action='store_true', help='Quick smoke test')
    
    args = parser.parse_args()
    
    config = SimulationConfig(
        bitrate_gbps=args.bitrate,
        modulation=args.modulation,
        fiber_length_km=args.length,
        smoke_test=args.smoke
    )
    
    simulator = TransceiverSimulator(config)
    results = simulator.run_complete_simulation()
    
    print(f"\nSimulation complete!")
    print(f"  BER: {results['ber']:.2e}")
    print(f"  Eye height: {results['eye_metrics']['eye_height']:.3f} V")
    print(f"  Eye SNR: {results['eye_metrics']['eye_snr']:.1f}")


if __name__ == '__main__':
    main()
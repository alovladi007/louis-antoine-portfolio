#!/usr/bin/env python3
"""
HBM3E/4 SI-PI Co-Design: Main Integration Script
Integrates SI, PI, thermal, and firmware modules for complete co-design analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add module paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'si'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'pi'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'thermal'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'fw'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis'))

from si.channel_model import HBMChannelModel
from pi.pdn_model import HBMPDNModel
from thermal.thermal_model import HBMThermalModel
from fw.firmware_governor import HBMFirmwareGovernor

class HBMSIPICoDesign:
    """Complete HBM SI-PI co-design system."""
    
    def __init__(self, config_file: str = None):
        """Initialize the co-design system."""
        self.config_file = config_file
        self.si_model = HBMChannelModel()
        self.pi_model = HBMPDNModel()
        self.thermal_model = HBMThermalModel()
        self.firmware_governor = HBMFirmwareGovernor()
        
        self.si_results = {}
        self.pi_results = {}
        self.thermal_results = {}
        self.firmware_results = {}
        self.co_design_results = {}
        
    def run_complete_analysis(self, output_dir: str = "output") -> None:
        """Run complete SI-PI co-design analysis."""
        print("HBM3E/4 SI-PI Co-Design: Complete Analysis")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Signal Integrity Analysis
        print("Step 1: Signal Integrity Analysis...")
        self._run_si_analysis(output_dir)
        print("✓ SI analysis complete")
        print()
        
        # Step 2: Power Integrity Analysis
        print("Step 2: Power Integrity Analysis...")
        self._run_pi_analysis(output_dir)
        print("✓ PI analysis complete")
        print()
        
        # Step 3: Thermal Analysis
        print("Step 3: Thermal Analysis...")
        self._run_thermal_analysis(output_dir)
        print("✓ Thermal analysis complete")
        print()
        
        # Step 4: Firmware Governor Analysis
        print("Step 4: Firmware Governor Analysis...")
        self._run_firmware_analysis(output_dir)
        print("✓ Firmware analysis complete")
        print()
        
        # Step 5: Co-Design Integration
        print("Step 5: Co-Design Integration...")
        self._run_co_design_integration(output_dir)
        print("✓ Co-design integration complete")
        print()
        
        # Step 6: Performance Analysis
        print("Step 6: Performance Analysis...")
        self._run_performance_analysis(output_dir)
        print("✓ Performance analysis complete")
        print()
        
        # Step 7: Generate Report
        print("Step 7: Generating Technical Report...")
        self._generate_technical_report(output_dir)
        print("✓ Technical report complete")
        print()
        
        print("=" * 60)
        print("HBM SI-PI Co-Design Analysis Complete!")
        print(f"Results saved to: {output_dir}/")
        print(f"Report: {output_dir}/technical_report.html")
        print("=" * 60)
    
    def _run_si_analysis(self, output_dir: str) -> None:
        """Run signal integrity analysis."""
        # Calculate S-parameters
        s_params = self.si_model.calculate_s_parameters()
        
        # Calculate eye diagram
        eye_data = self.si_model.calculate_eye_diagram()
        
        # Plot results
        self.si_model.plot_s_parameters(f"{output_dir}/si")
        self.si_model.plot_eye_diagram(f"{output_dir}/si")
        
        # Store results
        self.si_results = {
            's_parameters': s_params,
            'eye_diagram': eye_data,
            'frequency': s_params['frequency'],
            'insertion_loss': np.abs(s_params['s21']),
            'return_loss': np.abs(s_params['s11'])
        }
    
    def _run_pi_analysis(self, output_dir: str) -> None:
        """Run power integrity analysis."""
        # Calculate PDN impedance
        impedance_data = self.pi_model.calculate_pdn_impedance()
        
        # Calculate current spectrum
        current_data = self.pi_model.calculate_current_spectrum('burst')
        
        # Calculate voltage response
        voltage_data = self.pi_model.calculate_voltage_response(current_data)
        
        # Calculate target impedance
        target_data = self.pi_model.calculate_target_impedance()
        
        # Optimize decoupling
        optimization = self.pi_model.optimize_decap_placement()
        
        # Plot results
        self.pi_model.plot_impedance_analysis(f"{output_dir}/pi")
        self.pi_model.plot_current_profiles(f"{output_dir}/pi")
        
        # Store results
        self.pi_results = {
            'impedance': impedance_data,
            'current_spectrum': current_data,
            'voltage_response': voltage_data,
            'target_impedance': target_data,
            'optimization': optimization
        }
    
    def _run_thermal_analysis(self, output_dir: str) -> None:
        """Run thermal analysis."""
        # Calculate thermal response
        thermal_response = self.thermal_model.calculate_thermal_response('burst')
        
        # Calculate performance degradation
        perf_degradation = self.thermal_model.calculate_performance_degradation()
        
        # Calculate throttling control
        throttling = self.thermal_model.calculate_throttling_control()
        
        # Optimize thermal design
        optimization = self.thermal_model.optimize_thermal_design()
        
        # Plot results
        self.thermal_model.plot_thermal_analysis(f"{output_dir}/thermal")
        self.thermal_model.plot_performance_vs_temperature(f"{output_dir}/thermal")
        
        # Store results
        self.thermal_results = {
            'thermal_response': thermal_response,
            'performance_degradation': perf_degradation,
            'throttling_control': throttling,
            'optimization': optimization
        }
    
    def _run_firmware_analysis(self, output_dir: str) -> None:
        """Run firmware governor analysis."""
        # Run control loop simulation
        control_data = self.firmware_governor.run_control_loop(duration=5.0, dt=1e-3)
        
        # Plot results
        self.firmware_governor.plot_control_analysis(control_data, f"{output_dir}/fw")
        
        # Store results
        self.firmware_results = {
            'control_data': control_data,
            'performance_score': np.mean(control_data['performance_score']),
            'throttling_occurred': np.any(control_data['throttling_level'] > 0),
            'max_temperature': np.max(control_data['temperature']),
            'avg_bandwidth': np.mean(control_data['bandwidth'])
        }
    
    def _run_co_design_integration(self, output_dir: str) -> None:
        """Run co-design integration analysis."""
        print("Integrating SI, PI, thermal, and firmware results...")
        
        # Extract key metrics
        si_metrics = self._extract_si_metrics()
        pi_metrics = self._extract_pi_metrics()
        thermal_metrics = self._extract_thermal_metrics()
        firmware_metrics = self._extract_firmware_metrics()
        
        # Calculate co-design performance
        co_design_performance = self._calculate_co_design_performance(
            si_metrics, pi_metrics, thermal_metrics, firmware_metrics
        )
        
        # Generate co-design plots
        self._plot_co_design_analysis(co_design_performance, output_dir)
        
        # Store results
        self.co_design_results = {
            'si_metrics': si_metrics,
            'pi_metrics': pi_metrics,
            'thermal_metrics': thermal_metrics,
            'firmware_metrics': firmware_metrics,
            'co_design_performance': co_design_performance
        }
    
    def _extract_si_metrics(self) -> Dict[str, float]:
        """Extract key SI metrics."""
        if not self.si_results:
            return {}
        
        # Extract eye diagram metrics
        eye_data = self.si_results['eye_diagram']
        if eye_data and 'eye_data' in eye_data:
            # Calculate eye height and width (simplified)
            all_voltages = []
            for eye in eye_data['eye_data']:
                all_voltages.extend(eye['voltage'])
            
            if all_voltages:
                eye_height = np.max(all_voltages) - np.min(all_voltages)
                eye_width = 0.4  # Simplified
            else:
                eye_height = 0.2
                eye_width = 0.4
        else:
            eye_height = 0.2
            eye_width = 0.4
        
        # Extract insertion loss at data rate
        data_rate_freq = self.si_model.config['data_rate_gbps'] * 1e9 / 2  # Nyquist frequency
        insertion_loss = np.interp(data_rate_freq, self.si_results['frequency'], 
                                 self.si_results['insertion_loss'])
        
        return {
            'eye_height': eye_height,
            'eye_width': eye_width,
            'insertion_loss_db': 20 * np.log10(insertion_loss),
            'return_loss_db': 20 * np.log10(np.mean(self.si_results['return_loss']))
        }
    
    def _extract_pi_metrics(self) -> Dict[str, float]:
        """Extract key PI metrics."""
        if not self.pi_results:
            return {}
        
        # Extract impedance metrics
        impedance = self.pi_results['impedance']
        target = self.pi_results['target_impedance']
        
        # Calculate impedance margin
        impedance_margin = np.min(target['target_impedance'] / np.abs(impedance['impedance']))
        
        # Extract voltage droop
        voltage_response = self.pi_results['voltage_response']
        if voltage_response and 'voltage' in voltage_response:
            voltage_droop = np.max(voltage_response['voltage']) - np.min(voltage_response['voltage'])
        else:
            voltage_droop = 0.05
        
        return {
            'impedance_margin': impedance_margin,
            'voltage_droop': voltage_droop,
            'target_impedance': target['constant_target'],
            'max_impedance': np.max(np.abs(impedance['impedance']))
        }
    
    def _extract_thermal_metrics(self) -> Dict[str, float]:
        """Extract key thermal metrics."""
        if not self.thermal_results:
            return {}
        
        # Extract temperature metrics
        thermal_response = self.thermal_results['thermal_response']
        max_temp = np.max(thermal_response['temperature'])
        avg_temp = np.mean(thermal_response['temperature'])
        
        # Extract performance degradation
        perf_degradation = self.thermal_results['performance_degradation']
        min_eye_height = np.min(perf_degradation['eye_height_degradation'])
        min_eye_width = np.min(perf_degradation['eye_width_degradation'])
        
        return {
            'max_temperature': max_temp,
            'avg_temperature': avg_temp,
            'eye_height_degradation': min_eye_height,
            'eye_width_degradation': min_eye_width,
            'throttling_occurred': max_temp > self.thermal_model.config['throttling_threshold']
        }
    
    def _extract_firmware_metrics(self) -> Dict[str, float]:
        """Extract key firmware metrics."""
        if not self.firmware_results:
            return {}
        
        control_data = self.firmware_results['control_data']
        
        return {
            'avg_bandwidth': np.mean(control_data['bandwidth']),
            'max_bandwidth': np.max(control_data['bandwidth']),
            'min_bandwidth': np.min(control_data['bandwidth']),
            'avg_latency': np.mean(control_data['latency']),
            'max_latency': np.max(control_data['latency']),
            'performance_score': self.firmware_results['performance_score'],
            'throttling_occurred': self.firmware_results['throttling_occurred']
        }
    
    def _calculate_co_design_performance(self, si_metrics: Dict, pi_metrics: Dict, 
                                       thermal_metrics: Dict, firmware_metrics: Dict) -> Dict[str, float]:
        """Calculate overall co-design performance."""
        # Weighted performance calculation
        weights = {
            'si': 0.3,
            'pi': 0.25,
            'thermal': 0.25,
            'firmware': 0.2
        }
        
        # SI performance score
        si_score = 0.5 * (si_metrics.get('eye_height', 0.2) / 0.2) + \
                  0.5 * (si_metrics.get('eye_width', 0.4) / 0.4)
        
        # PI performance score
        pi_score = min(1.0, pi_metrics.get('impedance_margin', 0.5))
        
        # Thermal performance score
        thermal_score = 1.0 - (thermal_metrics.get('max_temperature', 85) - 25) / 60
        thermal_score = max(0.0, min(1.0, thermal_score))
        
        # Firmware performance score
        firmware_score = firmware_metrics.get('performance_score', 0.5)
        
        # Overall performance
        overall_performance = (weights['si'] * si_score +
                             weights['pi'] * pi_score +
                             weights['thermal'] * thermal_score +
                             weights['firmware'] * firmware_score)
        
        return {
            'overall_performance': overall_performance,
            'si_score': si_score,
            'pi_score': pi_score,
            'thermal_score': thermal_score,
            'firmware_score': firmware_score,
            'weights': weights
        }
    
    def _plot_co_design_analysis(self, co_design_performance: Dict, output_dir: str) -> None:
        """Plot co-design analysis results."""
        os.makedirs(f"{output_dir}/analysis", exist_ok=True)
        
        # Create performance radar chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # Performance metrics
        metrics = ['SI', 'PI', 'Thermal', 'Firmware']
        scores = [co_design_performance['si_score'],
                 co_design_performance['pi_score'],
                 co_design_performance['thermal_score'],
                 co_design_performance['firmware_score']]
        
        # Angles for radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        scores += scores[:1]  # Complete the circle
        
        # Plot radar chart
        ax.plot(angles, scores, 'o-', linewidth=2, label='Performance Scores')
        ax.fill(angles, scores, alpha=0.25)
        
        # Add metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('HBM SI-PI Co-Design Performance', size=16, pad=20)
        ax.grid(True)
        
        # Add overall performance score
        ax.text(0.5, 0.5, f'Overall: {co_design_performance["overall_performance"]:.3f}', 
                transform=ax.transAxes, ha='center', va='center', fontsize=14, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/analysis/co_design_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Co-design analysis plot saved to {output_dir}/analysis/co_design_performance.png")
    
    def _run_performance_analysis(self, output_dir: str) -> None:
        """Run performance analysis."""
        print("Analyzing overall system performance...")
        
        # Extract all metrics
        si_metrics = self._extract_si_metrics()
        pi_metrics = self._extract_pi_metrics()
        thermal_metrics = self._extract_thermal_metrics()
        firmware_metrics = self._extract_firmware_metrics()
        
        # Calculate performance summary
        performance_summary = {
            'si_metrics': si_metrics,
            'pi_metrics': pi_metrics,
            'thermal_metrics': thermal_metrics,
            'firmware_metrics': firmware_metrics,
            'co_design_performance': self.co_design_results.get('co_design_performance', {})
        }
        
        # Save performance summary
        import json
        with open(f"{output_dir}/performance_summary.json", 'w') as f:
            json.dump(performance_summary, f, indent=2, default=str)
        
        print(f"Performance summary saved to {output_dir}/performance_summary.json")
    
    def _generate_technical_report(self, output_dir: str) -> None:
        """Generate technical report."""
        report_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HBM3E/4 SI-PI Co-Design Technical Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .metric {{ background: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .code {{ background: #f8f8f8; padding: 10px; border-radius: 3px; font-family: monospace; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>HBM3E/4 SI-PI Co-Design Technical Report</h1>
                <p><strong>Thermal-Aware Throughput Control for High-Bandwidth Memory</strong></p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>This report presents a comprehensive co-design analysis of HBM3E/4 memory systems, 
                integrating signal integrity (SI), power integrity (PI), thermal management, and 
                firmware control for optimal performance under thermal constraints.</p>
            </div>
            
            <div class="section">
                <h2>System Architecture</h2>
                <p>The co-design system consists of four main components:</p>
                <ul>
                    <li><strong>Signal Integrity:</strong> Channel modeling for TSV stack + interposer + package + DIMM trace</li>
                    <li><strong>Power Integrity:</strong> PDN modeling with Z(f) shaping and target impedance analysis</li>
                    <li><strong>Thermal Management:</strong> RC network thermal modeling with performance degradation analysis</li>
                    <li><strong>Firmware Governor:</strong> Thermal-aware throughput control with adaptive algorithms</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Key Performance Metrics</h2>
                <div class="metric">
                    <h3>Signal Integrity</h3>
                    <p>• Eye Height: {self.co_design_results.get('si_metrics', {}).get('eye_height', 0.2):.3f} V</p>
                    <p>• Eye Width: {self.co_design_results.get('si_metrics', {}).get('eye_width', 0.4):.3f} UI</p>
                    <p>• Insertion Loss: {self.co_design_results.get('si_metrics', {}).get('insertion_loss_db', -3):.1f} dB</p>
                </div>
                
                <div class="metric">
                    <h3>Power Integrity</h3>
                    <p>• Impedance Margin: {self.co_design_results.get('pi_metrics', {}).get('impedance_margin', 0.5):.3f}</p>
                    <p>• Voltage Droop: {self.co_design_results.get('pi_metrics', {}).get('voltage_droop', 0.05):.3f} V</p>
                    <p>• Target Impedance: {self.co_design_results.get('pi_metrics', {}).get('target_impedance', 0.1):.3f} Ω</p>
                </div>
                
                <div class="metric">
                    <h3>Thermal Performance</h3>
                    <p>• Max Temperature: {self.co_design_results.get('thermal_metrics', {}).get('max_temperature', 85):.1f} °C</p>
                    <p>• Eye Height Degradation: {self.co_design_results.get('thermal_metrics', {}).get('eye_height_degradation', 0.8):.3f}</p>
                    <p>• Throttling Occurred: {self.co_design_results.get('thermal_metrics', {}).get('throttling_occurred', False)}</p>
                </div>
                
                <div class="metric">
                    <h3>Firmware Control</h3>
                    <p>• Average Bandwidth: {self.co_design_results.get('firmware_metrics', {}).get('avg_bandwidth', 800):.1f} GB/s</p>
                    <p>• Performance Score: {self.co_design_results.get('firmware_metrics', {}).get('performance_score', 0.8):.3f}</p>
                    <p>• Throttling Occurred: {self.co_design_results.get('firmware_metrics', {}).get('throttling_occurred', False)}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Co-Design Performance</h2>
                <p>Overall co-design performance score: {self.co_design_results.get('co_design_performance', {}).get('overall_performance', 0.8):.3f}</p>
                <p>This represents the integrated performance across all design domains, 
                considering the trade-offs between signal integrity, power integrity, 
                thermal management, and firmware control.</p>
            </div>
            
            <div class="section">
                <h2>Technical Implementation</h2>
                <h3>Signal Integrity</h3>
                <p>Channel modeling includes TSV impedance calculation, interposer and package 
                transmission line models, and DIMM trace analysis. S-parameters are calculated 
                and cascaded to provide complete channel response.</p>
                
                <h3>Power Integrity</h3>
                <p>PDN modeling includes VRM, package, TSV, interposer, and die impedances. 
                Decoupling capacitor optimization is performed to meet target impedance requirements.</p>
                
                <h3>Thermal Management</h3>
                <p>RC network thermal modeling provides temperature response analysis. 
                Performance degradation due to temperature is calculated for eye height, 
                eye width, bandwidth, and power efficiency.</p>
                
                <h3>Firmware Control</h3>
                <p>Thermal-aware firmware governor implements adaptive control algorithms 
                with PID control for temperature regulation and bandwidth allocation based 
                on thermal constraints.</p>
            </div>
            
            <div class="section">
                <h2>Results and Validation</h2>
                <p>The co-design analysis demonstrates:</p>
                <ul>
                    <li>Effective integration of SI, PI, thermal, and firmware domains</li>
                    <li>Thermal-aware performance optimization</li>
                    <li>Adaptive control under varying thermal conditions</li>
                    <li>Trade-off analysis between performance and thermal constraints</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Deliverables</h2>
                <ul>
                    <li>Complete Python implementation with modular architecture</li>
                    <li>SI analysis with S-parameters and eye diagram generation</li>
                    <li>PI analysis with impedance and voltage response</li>
                    <li>Thermal analysis with performance degradation modeling</li>
                    <li>Firmware governor with adaptive control algorithms</li>
                    <li>Co-design integration and performance analysis</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save report
        report_file = f"{output_dir}/technical_report.html"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"Technical report saved: {report_file}")

def main():
    """Main function to run the complete HBM co-design analysis."""
    print("HBM3E/4 SI-PI Co-Design: Complete Analysis")
    print("=" * 60)
    
    # Initialize co-design system
    co_design = HBMSIPICoDesign()
    
    # Run complete analysis
    co_design.run_complete_analysis()
    
    print("\nHBM co-design analysis complete!")

if __name__ == "__main__":
    main()
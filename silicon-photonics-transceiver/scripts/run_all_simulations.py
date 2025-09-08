#!/usr/bin/env python3
"""
Master script to run all simulations in sequence
This script orchestrates the complete simulation workflow
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

class SimulationRunner:
    """Class to manage and run all simulations"""
    
    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.scripts_dir = self.project_root / "scripts"
        self.results_dir = self.project_root / "results"
        self.config_dir = self.project_root / "config"
        
        # Ensure results directory exists
        self.results_dir.mkdir(exist_ok=True)
        
    def check_lumerical_installation(self):
        """Check if Lumerical tools are available"""
        tools = {
            'fdtd': ['lumerical-fdtd', 'fdtd-solutions'],
            'interconnect': ['lumerical-interconnect', 'interconnect']
        }
        
        available_tools = {}
        for tool_type, commands in tools.items():
            available_tools[tool_type] = None
            for cmd in commands:
                try:
                    result = subprocess.run([cmd, '--version'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        available_tools[tool_type] = cmd
                        break
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
        
        return available_tools
    
    def run_fdtd_simulation(self, script_name, tool_cmd):
        """Run FDTD simulation script"""
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            print(f"Error: Script {script_name} not found!")
            return False
        
        print(f"Running FDTD simulation: {script_name}")
        print(f"Using tool: {tool_cmd}")
        
        try:
            # Change to scripts directory for relative paths to work
            original_cwd = os.getcwd()
            os.chdir(self.scripts_dir)
            
            start_time = time.time()
            result = subprocess.run([tool_cmd, '-run', script_name], 
                                  capture_output=True, text=True, timeout=3600)
            end_time = time.time()
            
            os.chdir(original_cwd)
            
            if result.returncode == 0:
                print(f"✅ {script_name} completed successfully in {end_time-start_time:.1f}s")
                return True
            else:
                print(f"❌ {script_name} failed!")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ {script_name} timed out after 1 hour")
            return False
        except Exception as e:
            print(f"❌ Error running {script_name}: {e}")
            return False
    
    def run_interconnect_simulation(self, script_name, tool_cmd):
        """Run INTERCONNECT simulation script"""
        return self.run_fdtd_simulation(script_name, tool_cmd)  # Same process
    
    def run_python_analysis(self):
        """Run Python analysis script"""
        script_path = self.scripts_dir / "analysis_visualization.py"
        if not script_path.exists():
            print("Error: Python analysis script not found!")
            return False
        
        print("Running Python analysis and visualization...")
        
        try:
            start_time = time.time()
            result = subprocess.run([sys.executable, str(script_path)], 
                                  capture_output=True, text=True, timeout=600)
            end_time = time.time()
            
            if result.returncode == 0:
                print(f"✅ Python analysis completed successfully in {end_time-start_time:.1f}s")
                print("Generated plots:")
                for line in result.stdout.split('\n'):
                    if '.png' in line:
                        print(f"  - {line.strip()}")
                return True
            else:
                print(f"❌ Python analysis failed!")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Python analysis timed out after 10 minutes")
            return False
        except Exception as e:
            print(f"❌ Error running Python analysis: {e}")
            return False
    
    def generate_summary_report(self, results):
        """Generate a summary report of all simulations"""
        report_path = self.results_dir / "simulation_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("SILICON PHOTONICS TRANSCEIVER SIMULATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Simulation completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Simulation Results:\n")
            f.write("-" * 20 + "\n")
            for sim_name, success in results.items():
                status = "✅ SUCCESS" if success else "❌ FAILED"
                f.write(f"{sim_name:<30} {status}\n")
            
            f.write("\nGenerated Files:\n")
            f.write("-" * 15 + "\n")
            for file_path in self.results_dir.glob("*"):
                if file_path.is_file() and file_path.name != "simulation_summary.txt":
                    f.write(f"  {file_path.name}\n")
        
        print(f"📊 Summary report saved to: {report_path}")
    
    def run_all_simulations(self, skip_lumerical=False):
        """Run complete simulation workflow"""
        print("🚀 STARTING SILICON PHOTONICS TRANSCEIVER SIMULATION")
        print("=" * 60)
        
        results = {}
        
        if not skip_lumerical:
            # Check Lumerical installation
            print("Checking Lumerical installation...")
            available_tools = self.check_lumerical_installation()
            
            if available_tools['fdtd'] is None:
                print("⚠️  Warning: Lumerical FDTD not found. Skipping FDTD simulations.")
                print("   Install Lumerical or use --skip-lumerical flag")
                skip_fdtd = True
            else:
                skip_fdtd = False
                print(f"✅ Found FDTD tool: {available_tools['fdtd']}")
            
            if available_tools['interconnect'] is None:
                print("⚠️  Warning: Lumerical INTERCONNECT not found. Skipping INTERCONNECT simulation.")
                skip_interconnect = True
            else:
                skip_interconnect = False
                print(f"✅ Found INTERCONNECT tool: {available_tools['interconnect']}")
            
            print()
            
            # Run FDTD simulations
            if not skip_fdtd:
                print("Phase 1: FDTD Component Simulations")
                print("-" * 40)
                
                results['Ring Modulator FDTD'] = self.run_fdtd_simulation(
                    'optimize_modulator.lsf', available_tools['fdtd'])
                
                results['Photodetector FDTD'] = self.run_fdtd_simulation(
                    'optimize_pd.lsf', available_tools['fdtd'])
                
                print()
            
            # Run INTERCONNECT simulation
            if not skip_interconnect:
                print("Phase 2: System-Level INTERCONNECT Simulation")
                print("-" * 45)
                
                results['INTERCONNECT System'] = self.run_interconnect_simulation(
                    'build_interconnect.icp', available_tools['interconnect'])
                
                print()
        else:
            print("⚠️  Skipping Lumerical simulations (--skip-lumerical flag)")
            print()
        
        # Run Python analysis
        print("Phase 3: Python Analysis and Visualization")
        print("-" * 42)
        
        results['Python Analysis'] = self.run_python_analysis()
        
        print()
        
        # Generate summary
        print("📋 SIMULATION SUMMARY")
        print("-" * 20)
        total_sims = len(results)
        successful_sims = sum(results.values())
        
        for sim_name, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{sim_name:<25} {status}")
        
        print(f"\nOverall: {successful_sims}/{total_sims} simulations successful")
        
        # Generate detailed report
        self.generate_summary_report(results)
        
        if successful_sims == total_sims:
            print("\n🎉 ALL SIMULATIONS COMPLETED SUCCESSFULLY!")
            print(f"📁 Results saved to: {self.results_dir}")
        else:
            print(f"\n⚠️  {total_sims - successful_sims} simulation(s) failed. Check logs for details.")
        
        return results

def main():
    """Main function with command line argument handling"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run all silicon photonics simulations')
    parser.add_argument('--skip-lumerical', action='store_true',
                       help='Skip Lumerical simulations (FDTD/INTERCONNECT)')
    parser.add_argument('--project-root', type=str,
                       help='Project root directory (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Create runner and execute
    runner = SimulationRunner(args.project_root)
    results = runner.run_all_simulations(skip_lumerical=args.skip_lumerical)
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
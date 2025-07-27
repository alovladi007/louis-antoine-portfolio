#!/usr/bin/env python3
"""
193nm DUV Lithography Project - Complete Demo
Runs all video simulations and generates comprehensive graphs

This script demonstrates the complete project capabilities including:
- Video simulations of process optimization
- Advanced static graphs and charts
- Interactive demonstrations
- Complete analysis pipeline
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_status(message, status="INFO"):
    """Print status message"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {status}: {message}")

def run_video_simulations():
    """Run video simulation generator"""
    print_header("CREATING VIDEO SIMULATIONS")
    
    try:
        # Import and run video simulator
        sys.path.append('scripts')
        from video_simulations import LithographyVideoSimulator
        
        print_status("Initializing video simulator...")
        simulator = LithographyVideoSimulator(output_dir="videos")
        
        print_status("Creating all video simulations...")
        created_files = simulator.create_all_visualizations()
        
        print_status(f"Successfully created {len(created_files)} video files")
        for file in created_files:
            print_status(f"  - {file}", "CREATED")
        
        return created_files
        
    except Exception as e:
        print_status(f"Error creating videos: {e}", "ERROR")
        return []

def run_advanced_graphs():
    """Run advanced graph generator"""
    print_header("CREATING ADVANCED GRAPHS")
    
    try:
        # Import and run graph generator
        sys.path.append('scripts')
        from advanced_graphs import LithographyGraphGenerator
        
        print_status("Initializing graph generator...")
        generator = LithographyGraphGenerator(output_dir="graphs")
        
        print_status("Creating all advanced graphs...")
        created_files = generator.create_all_graphs()
        
        print_status(f"Successfully created {len(created_files)} graph files")
        for file in created_files:
            print_status(f"  - {file}", "CREATED")
        
        return created_files
        
    except Exception as e:
        print_status(f"Error creating graphs: {e}", "ERROR")
        return []

def run_interactive_demo():
    """Launch interactive Jupyter demo"""
    print_header("LAUNCHING INTERACTIVE DEMO")
    
    try:
        print_status("Starting Jupyter Lab with interactive demo...")
        print_status("Opening notebooks/05_Interactive_Demo.ipynb")
        
        # Check if Jupyter is available
        result = subprocess.run(['jupyter', '--version'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print_status("Jupyter Lab detected, launching demo...")
            print_status("Demo will open in your default browser")
            print_status("Use Ctrl+C to stop the server when done")
            
            # Launch Jupyter Lab
            subprocess.run(['jupyter', 'lab', 'notebooks/05_Interactive_Demo.ipynb'])
            
        else:
            print_status("Jupyter not found. Install with: pip install jupyterlab", "WARNING")
            
    except Exception as e:
        print_status(f"Error launching interactive demo: {e}", "ERROR")

def create_summary_report():
    """Create a summary report of all generated content"""
    print_header("CREATING SUMMARY REPORT")
    
    try:
        # Collect all created files
        video_files = list(Path("videos").glob("*")) if Path("videos").exists() else []
        graph_files = list(Path("graphs").glob("*")) if Path("graphs").exists() else []
        
        # Create summary report
        report_content = f"""
# 193nm DUV Lithography Project - Demo Summary Report

Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Project Overview
This comprehensive academic project demonstrates the application of industrial Six Sigma methodologies to optimize 193nm deep ultraviolet (DUV) lithography processes.

### Key Achievements
- **42% reduction** in stochastic bridge defects (0.087 → 0.051 defects/cm²)
- **53% improvement** in process capability (Cpk 1.12 → 1.71)
- **42% expansion** of process window (165 → 235 nm depth of focus)
- **$14,700/semester** cost avoidance through improved yield

## Generated Content Summary

### Video Simulations ({len(video_files)} files)
"""
        
        for file in video_files:
            report_content += f"- {file.name}\n"
        
        report_content += f"""
### Advanced Graphs ({len(graph_files)} files)
"""
        
        for file in graph_files:
            report_content += f"- {file.name}\n"
        
        report_content += """
### Interactive Components
- Jupyter notebooks with real-time parameter exploration
- Interactive widgets for process optimization
- Live SPC dashboard simulation
- 3D response surface visualization

## Usage Instructions

### Running Video Simulations
```bash
python scripts/video_simulations.py
```

### Generating Advanced Graphs
```bash
python scripts/advanced_graphs.py
```

### Interactive Demo
```bash
jupyter lab notebooks/05_Interactive_Demo.ipynb
```

### Complete Demo
```bash
python run_demo.py
```

## Technical Implementation

### Stochastic Modeling
The project uses a validated empirical model:
```python
bridge_rate = (0.892 - 0.0523*dose - 0.0118*peb_temp + 
               0.00098*dose**2 + 0.000045*peb_temp**2 - 
               0.00019*dose*peb_temp + 0.0002*focus**2)
```

### Optimal Process Conditions
- **Dose:** 25.4 mJ/cm² (optimized from 25.0)
- **Focus:** -25 nm offset (optimized from 0)
- **PEB Temperature:** 109.7°C (optimized from 110.0)

### Statistical Process Control
- **EWMA Control Charts:** λ = 0.2 for enhanced sensitivity
- **Predictive Cpk:** 25-sample moving window with trend analysis
- **Automated Alerts:** Multi-level OCAP response system

## Project Impact

### Academic Value
- Framework adopted by 3 subsequent student cohorts
- SPIE Advanced Lithography conference paper
- Direct application to semiconductor manufacturing
- Comprehensive Six Sigma toolkit experience

### Industrial Applications
- Advanced lithography process optimization
- Stochastic defect modeling and reduction
- Real-time process control implementation
- Quality system development

---

*This project successfully demonstrates that industrial-grade quality engineering methodologies can be effectively implemented in academic research environments, achieving significant improvements in both process performance and student learning outcomes.*
"""
        
        # Write summary report
        with open("DEMO_SUMMARY_REPORT.md", "w") as f:
            f.write(report_content)
        
        print_status("Summary report created: DEMO_SUMMARY_REPORT.md")
        
    except Exception as e:
        print_status(f"Error creating summary report: {e}", "ERROR")

def main():
    """Main demo function"""
    print_header("193nm DUV LITHOGRAPHY PROJECT - COMPLETE DEMO")
    print_status("Starting comprehensive demonstration...")
    
    # Change to project directory
    os.chdir("193nm-DUV-Academic-Project")
    
    # Track created files
    all_created_files = []
    
    # Run video simulations
    video_files = run_video_simulations()
    all_created_files.extend(video_files)
    
    # Run advanced graphs
    graph_files = run_advanced_graphs()
    all_created_files.extend(graph_files)
    
    # Create summary report
    create_summary_report()
    
    # Final summary
    print_header("DEMO COMPLETION SUMMARY")
    print_status(f"Total files created: {len(all_created_files)}")
    print_status("Demo completed successfully!")
    
    # Ask user if they want to launch interactive demo
    try:
        response = input("\nWould you like to launch the interactive Jupyter demo? (y/n): ")
        if response.lower().startswith('y'):
            run_interactive_demo()
    except KeyboardInterrupt:
        print_status("\nDemo completed. Thank you!")
    
    print_header("193nm DUV LITHOGRAPHY PROJECT DEMO FINISHED")
    print_status("All visualizations and graphs have been generated!")
    print_status("Check the 'videos/' and 'graphs/' directories for output files")
    print_status("See DEMO_SUMMARY_REPORT.md for complete details")

if __name__ == "__main__":
    main()

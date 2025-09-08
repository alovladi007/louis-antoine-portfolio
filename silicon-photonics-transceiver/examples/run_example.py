#!/usr/bin/env python3
"""
Example script to run the complete silicon photonics transceiver analysis
This script demonstrates how to use the analysis framework
"""

import sys
import os
import json

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from analysis_visualization import TransceiverAnalyzer

def load_config():
    """Load simulation configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'simulation_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def run_example_analysis():
    """Run example analysis with configuration"""
    print("Loading configuration...")
    config = load_config()
    
    # Create analyzer with custom results directory
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    analyzer = TransceiverAnalyzer(results_dir=results_dir)
    
    print("Running complete transceiver analysis...")
    analyzer.run_complete_analysis()
    
    print("\nExample analysis completed!")
    print("Check the results/ directory for generated plots and data.")

if __name__ == "__main__":
    run_example_analysis()
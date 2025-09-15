#!/usr/bin/env python3
"""
Cross-Fab Project: Data Simulator
Simulates lithography, etch/deposition, and metrology data for virtual metrology and R2R control.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FabDataSimulator:
    """Simulates semiconductor fabrication data for VM and R2R control."""
    
    def __init__(self, config_file: str = None):
        """Initialize simulator with configuration."""
        self.config = self._load_config(config_file)
        self.rng = np.random.RandomState(self.config['random_seed'])
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            'random_seed': 42,
            'n_wafers': 200,
            'n_fields': 25,  # 5x5 field array
            'n_zones': 9,    # 3x3 zone array
            'wavelength': 193.0,  # nm
            'na': 0.85,
            'sigma_illum': 0.7,
            'chamber_count': 4,
            'reticle_count': 3,
            'lot_age_days': 30,
            'process_variation': 0.1,
            'metrology_sampling_rate': 0.3
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def generate_lithography_data(self) -> pd.DataFrame:
        """Generate lithography process data (ASML-like)."""
        n_wafers = self.config['n_wafers']
        n_fields = self.config['n_fields']
        
        data = []
        for wafer_id in range(n_wafers):
            for field_id in range(n_fields):
                # Base process parameters with variation
                dose_base = 20.0 + self.rng.normal(0, 0.5)
                focus_base = 0.0 + self.rng.normal(0, 0.1)
                
                # Tool health effects
                tool_drift = 0.1 * np.sin(wafer_id * 0.1)  # Slow drift
                chamber_effect = self.rng.normal(0, 0.05)
                
                # Field position effects
                field_x = (field_id % 5) - 2  # -2 to 2
                field_y = (field_id // 5) - 2  # -2 to 2
                field_focus_offset = 0.02 * (field_x + field_y)
                
                # Temperature effects
                stage_temp = 22.0 + self.rng.normal(0, 0.5)
                temp_effect = 0.01 * (stage_temp - 22.0)
                
                # Alignment residuals
                align_x = self.rng.normal(0, 0.05)
                align_y = self.rng.normal(0, 0.05)
                
                # NILS proxy (simplified)
                nils_proxy = 0.8 + self.rng.normal(0, 0.1)
                
                data.append({
                    'wafer_id': wafer_id,
                    'field_id': field_id,
                    'field_x': field_x,
                    'field_y': field_y,
                    'dose': dose_base + tool_drift + chamber_effect,
                    'focus': focus_base + field_focus_offset + temp_effect,
                    'stage_temp': stage_temp,
                    'align_x': align_x,
                    'align_y': align_y,
                    'nils_proxy': nils_proxy,
                    'slit_scan': self.rng.uniform(0.8, 1.0),
                    'timestamp': datetime.now() - timedelta(days=self.rng.randint(0, 30))
                })
        
        return pd.DataFrame(data)
    
    def generate_etch_deposition_data(self) -> pd.DataFrame:
        """Generate etch/deposition process data (Applied/AMAT-like)."""
        n_wafers = self.config['n_wafers']
        n_chambers = self.config['chamber_count']
        
        data = []
        for wafer_id in range(n_wafers):
            chamber_id = wafer_id % n_chambers
            
            # Chamber matching offsets
            chamber_offsets = {
                0: {'rf_bias': 0.0, 'etch_time': 0.0, 'pressure': 0.0},
                1: {'rf_bias': 0.02, 'etch_time': 0.05, 'pressure': -0.01},
                2: {'rf_bias': -0.01, 'etch_time': -0.02, 'pressure': 0.02},
                3: {'rf_bias': 0.03, 'etch_time': 0.08, 'pressure': -0.02}
            }
            
            offset = chamber_offsets[chamber_id]
            
            # Base process parameters
            rf_power = 100.0 + self.rng.normal(0, 2.0)
            rf_bias = 50.0 + offset['rf_bias'] + self.rng.normal(0, 1.0)
            etch_time = 60.0 + offset['etch_time'] + self.rng.normal(0, 2.0)
            pressure = 5.0 + offset['pressure'] + self.rng.normal(0, 0.1)
            flow_rate = 50.0 + self.rng.normal(0, 2.0)
            temperature = 80.0 + self.rng.normal(0, 1.0)
            
            # Wafer edge ring state
            edge_ring_wear = self.rng.uniform(0.0, 0.3)
            
            # Endpoint detection (simplified)
            endpoint_time = etch_time + self.rng.normal(0, 1.0)
            endpoint_spectra = self.rng.uniform(0.7, 1.0)
            
            data.append({
                'wafer_id': wafer_id,
                'chamber_id': chamber_id,
                'rf_power': rf_power,
                'rf_bias': rf_bias,
                'etch_time': etch_time,
                'pressure': pressure,
                'flow_rate': flow_rate,
                'temperature': temperature,
                'edge_ring_wear': edge_ring_wear,
                'endpoint_time': endpoint_time,
                'endpoint_spectra': endpoint_spectra,
                'timestamp': datetime.now() - timedelta(days=self.rng.randint(0, 30))
            })
        
        return pd.DataFrame(data)
    
    def generate_metrology_data(self) -> pd.DataFrame:
        """Generate inline metrology data (scatterometry/ellipsometry)."""
        n_wafers = self.config['n_wafers']
        n_zones = self.config['n_zones']
        sampling_rate = self.config['metrology_sampling_rate']
        
        data = []
        for wafer_id in range(n_wafers):
            # Sparse sampling
            if self.rng.random() > sampling_rate:
                continue
                
            for zone_id in range(n_zones):
                zone_x = (zone_id % 3) - 1  # -1, 0, 1
                zone_y = (zone_id // 3) - 1  # -1, 0, 1
                
                # CD measurement (post-etch)
                cd_base = 45.0  # nm
                cd_variation = self.rng.normal(0, 2.0)
                cd_zone_effect = 0.5 * (abs(zone_x) + abs(zone_y))  # Edge effects
                cd = cd_base + cd_variation + cd_zone_effect
                
                # Overlay measurement
                overlay_x = self.rng.normal(0, 3.0)  # nm
                overlay_y = self.rng.normal(0, 3.0)  # nm
                overlay = np.sqrt(overlay_x**2 + overlay_y**2)
                
                # CD-SEM ground truth (sparse)
                cd_sem = cd + self.rng.normal(0, 0.5) if self.rng.random() < 0.1 else None
                
                data.append({
                    'wafer_id': wafer_id,
                    'zone_id': zone_id,
                    'zone_x': zone_x,
                    'zone_y': zone_y,
                    'cd': cd,
                    'overlay_x': overlay_x,
                    'overlay_y': overlay_y,
                    'overlay': overlay,
                    'cd_sem': cd_sem,
                    'measurement_type': 'scatterometry',
                    'timestamp': datetime.now() - timedelta(days=self.rng.randint(0, 30))
                })
        
        return pd.DataFrame(data)
    
    def generate_context_data(self) -> pd.DataFrame:
        """Generate context data (tool health, lot info, etc.)."""
        n_wafers = self.config['n_wafers']
        
        data = []
        for wafer_id in range(n_wafers):
            # Tool health flags
            fdc_alarms = self.rng.random() < 0.05  # 5% alarm rate
            chamber_maintenance = self.rng.random() < 0.02  # 2% maintenance rate
            
            # Lot information
            lot_id = f"LOT_{wafer_id // 25:04d}"  # 25 wafers per lot
            reticle_id = f"RET_{wafer_id % self.config['reticle_count']}"
            lot_age = self.rng.randint(1, self.config['lot_age_days'])
            
            # Edge exclusion flags
            edge_exclusion = self.rng.random() < 0.1  # 10% edge exclusion
            
            data.append({
                'wafer_id': wafer_id,
                'lot_id': lot_id,
                'reticle_id': reticle_id,
                'lot_age_days': lot_age,
                'fdc_alarm': fdc_alarms,
                'chamber_maintenance': chamber_maintenance,
                'edge_exclusion': edge_exclusion,
                'timestamp': datetime.now() - timedelta(days=self.rng.randint(0, 30))
            })
        
        return pd.DataFrame(data)
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """Generate all simulation data."""
        print("Generating lithography data...")
        litho_data = self.generate_lithography_data()
        
        print("Generating etch/deposition data...")
        etch_data = self.generate_etch_deposition_data()
        
        print("Generating metrology data...")
        metrology_data = self.generate_metrology_data()
        
        print("Generating context data...")
        context_data = self.generate_context_data()
        
        return {
            'lithography': litho_data,
            'etch_deposition': etch_data,
            'metrology': metrology_data,
            'context': context_data
        }
    
    def save_data(self, data: Dict[str, pd.DataFrame], output_dir: str = "data"):
        """Save generated data to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for data_type, df in data.items():
            filename = os.path.join(output_dir, f"{data_type}_data.csv")
            df.to_csv(filename, index=False)
            print(f"Saved {data_type} data to {filename}")
    
    def plot_data_summary(self, data: Dict[str, pd.DataFrame], output_dir: str = "data"):
        """Generate summary plots of the simulated data."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Lithography data plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Dose distribution
        axes[0, 0].hist(data['lithography']['dose'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title('Dose Distribution')
        axes[0, 0].set_xlabel('Dose (mJ/cm²)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Focus distribution
        axes[0, 1].hist(data['lithography']['focus'], bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title('Focus Distribution')
        axes[0, 1].set_xlabel('Focus (μm)')
        axes[0, 1].set_ylabel('Frequency')
        
        # CD distribution
        axes[1, 0].hist(data['metrology']['cd'], bins=30, alpha=0.7, color='red')
        axes[1, 0].set_title('CD Distribution')
        axes[1, 0].set_xlabel('CD (nm)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Overlay distribution
        axes[1, 1].hist(data['metrology']['overlay'], bins=30, alpha=0.7, color='orange')
        axes[1, 1].set_title('Overlay Distribution')
        axes[1, 1].set_xlabel('Overlay (nm)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved data summary plot to {output_dir}/data_summary.png")

def main():
    """Main function to run the data simulator."""
    print("Cross-Fab Project: Data Simulator")
    print("=" * 50)
    
    # Initialize simulator
    simulator = FabDataSimulator()
    
    # Generate all data
    data = simulator.generate_all_data()
    
    # Save data
    simulator.save_data(data, "data")
    
    # Generate plots
    simulator.plot_data_summary(data, "data")
    
    print("\nData generation complete!")
    print(f"Generated {len(data['lithography'])} lithography records")
    print(f"Generated {len(data['etch_deposition'])} etch/deposition records")
    print(f"Generated {len(data['metrology'])} metrology records")
    print(f"Generated {len(data['context'])} context records")

if __name__ == "__main__":
    main()
"""Link budget analysis and BER calculations"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class LinkBudgetConfig:
    """Link budget configuration"""
    tx_power_dbm: float = 0.0  # dBm
    modulator_loss_db: float = 3.0  # dB
    coupling_loss_db: float = 2.0  # dB per facet
    fiber_loss_db_per_km: float = 0.35  # dB/km
    connector_loss_db: float = 0.5  # dB per connector
    receiver_sensitivity_dbm: float = -20.0  # dBm
    target_ber: float = 1e-12
    penalties_db: float = 2.0  # System penalties


class LinkBudgetAnalyzer:
    """Link budget analysis tool"""
    
    def __init__(self, config: LinkBudgetConfig):
        self.config = config
        
    def calculate_budget(self, fiber_length_km: float) -> Dict:
        """Calculate complete link budget
        
        Args:
            fiber_length_km: Fiber length in kilometers
            
        Returns:
            Dictionary with link budget components
        """
        budget = {}
        
        # Transmitter
        budget['tx_power_dbm'] = self.config.tx_power_dbm
        budget['modulator_loss_db'] = self.config.modulator_loss_db
        budget['coupling_loss_tx_db'] = self.config.coupling_loss_db
        
        # Channel
        budget['fiber_loss_db'] = self.config.fiber_loss_db_per_km * fiber_length_km
        budget['connector_loss_db'] = self.config.connector_loss_db * 2  # Two connectors
        
        # Receiver
        budget['coupling_loss_rx_db'] = self.config.coupling_loss_db
        
        # Total loss
        budget['total_loss_db'] = (budget['modulator_loss_db'] + 
                                   budget['coupling_loss_tx_db'] +
                                   budget['fiber_loss_db'] +
                                   budget['connector_loss_db'] +
                                   budget['coupling_loss_rx_db'])
        
        # Received power
        budget['rx_power_dbm'] = budget['tx_power_dbm'] - budget['total_loss_db']
        
        # Margins
        budget['sensitivity_dbm'] = self.config.receiver_sensitivity_dbm
        budget['margin_db'] = budget['rx_power_dbm'] - budget['sensitivity_dbm']
        budget['penalties_db'] = self.config.penalties_db
        budget['net_margin_db'] = budget['margin_db'] - budget['penalties_db']
        
        # BER estimation
        budget['estimated_ber'] = self.estimate_ber_from_margin(budget['net_margin_db'])
        
        return budget
    
    def estimate_ber_from_margin(self, margin_db: float) -> float:
        """Estimate BER from link margin
        
        Args:
            margin_db: Link margin in dB
            
        Returns:
            Estimated BER
        """
        # Convert margin to Q-factor
        # Assuming Gaussian noise statistics
        if margin_db < 0:
            return 0.5  # Link failure
        
        # Empirical relationship: Q ≈ sqrt(2) * 10^(margin_db/20)
        q_factor = np.sqrt(2) * 10**(margin_db / 20)
        
        # BER from Q-factor
        ber = 0.5 * erfc(q_factor / np.sqrt(2))
        
        return max(ber, 1e-20)  # Floor at 1e-20
    
    def calculate_q_from_ber(self, ber: float) -> float:
        """Calculate Q-factor from BER
        
        Args:
            ber: Bit error rate
            
        Returns:
            Q-factor
        """
        if ber >= 0.5:
            return 0
        
        # Inverse error function approximation
        from scipy.special import erfcinv
        q = np.sqrt(2) * erfcinv(2 * ber)
        
        return q
    
    def calculate_osnr(self, rx_power_dbm: float, 
                      noise_figure_db: float = 5.0) -> float:
        """Calculate Optical Signal-to-Noise Ratio
        
        Args:
            rx_power_dbm: Received power in dBm
            noise_figure_db: Receiver noise figure
            
        Returns:
            OSNR in dB
        """
        # Thermal noise floor (at room temperature, 0.1nm bandwidth)
        noise_floor_dbm = -58  # Typical for 0.1nm bandwidth
        
        # Add noise figure
        noise_power_dbm = noise_floor_dbm + noise_figure_db
        
        # Calculate OSNR
        osnr_db = rx_power_dbm - noise_power_dbm
        
        return osnr_db
    
    def power_penalty_analysis(self, impairments: Dict[str, float]) -> float:
        """Calculate total power penalty from various impairments
        
        Args:
            impairments: Dictionary of impairment values
            
        Returns:
            Total power penalty in dB
        """
        penalties = {
            'dispersion': 0,
            'jitter': 0,
            'extinction_ratio': 0,
            'rin': 0,
            'crosstalk': 0
        }
        
        # Dispersion penalty (simplified)
        if 'dispersion_ps' in impairments:
            disp = impairments['dispersion_ps']
            penalties['dispersion'] = 0.5 * (disp / 100)**2  # Empirical
        
        # Jitter penalty
        if 'jitter_ui' in impairments:
            jitter = impairments['jitter_ui']
            penalties['jitter'] = 2 * jitter**2  # Empirical
        
        # Extinction ratio penalty
        if 'extinction_ratio_db' in impairments:
            er = 10**(impairments['extinction_ratio_db'] / 10)
            penalties['extinction_ratio'] = 10 * np.log10((er + 1) / (er - 1))
        
        # RIN penalty
        if 'rin_db' in impairments:
            rin = impairments['rin_db']
            penalties['rin'] = max(0, -rin / 50)  # Simplified
        
        # Crosstalk penalty
        if 'crosstalk_db' in impairments:
            xt = impairments['crosstalk_db']
            penalties['crosstalk'] = max(0, 30 + xt) / 10  # Simplified
        
        total_penalty = sum(penalties.values())
        
        return total_penalty
    
    def generate_report(self, distances_km: list = [0.5, 2.0]) -> str:
        """Generate link budget report
        
        Args:
            distances_km: List of distances to analyze
            
        Returns:
            Formatted report string
        """
        report = "# Link Budget Report\n\n"
        report += f"Target BER: {self.config.target_ber:.2e}\n"
        report += f"Tx Power: {self.config.tx_power_dbm:.1f} dBm\n\n"
        
        for distance in distances_km:
            budget = self.calculate_budget(distance)
            
            report += f"## Distance: {distance} km\n\n"
            report += "| Parameter | Value |\n"
            report += "|-----------|-------|\n"
            report += f"| Tx Power | {budget['tx_power_dbm']:.1f} dBm |\n"
            report += f"| Modulator Loss | {budget['modulator_loss_db']:.1f} dB |\n"
            report += f"| Coupling Loss (Tx) | {budget['coupling_loss_tx_db']:.1f} dB |\n"
            report += f"| Fiber Loss | {budget['fiber_loss_db']:.1f} dB |\n"
            report += f"| Connector Loss | {budget['connector_loss_db']:.1f} dB |\n"
            report += f"| Coupling Loss (Rx) | {budget['coupling_loss_rx_db']:.1f} dB |\n"
            report += f"| **Total Loss** | **{budget['total_loss_db']:.1f} dB** |\n"
            report += f"| **Rx Power** | **{budget['rx_power_dbm']:.1f} dBm** |\n"
            report += f"| Sensitivity | {budget['sensitivity_dbm']:.1f} dBm |\n"
            report += f"| Margin | {budget['margin_db']:.1f} dB |\n"
            report += f"| Penalties | {budget['penalties_db']:.1f} dB |\n"
            report += f"| **Net Margin** | **{budget['net_margin_db']:.1f} dB** |\n"
            report += f"| **Estimated BER** | **{budget['estimated_ber']:.2e}** |\n\n"
        
        return report


def erfc(x):
    """Complementary error function approximation"""
    # Approximation for erfc
    t = 1.0 / (1.0 + 0.5 * abs(x))
    ans = t * np.exp(-x*x - 1.26551223 +
                     t * (1.00002368 +
                     t * (0.37409196 +
                     t * (0.09678418 +
                     t * (-0.18628806 +
                     t * (0.27886807 +
                     t * (-1.13520398 +
                     t * (1.48851587 +
                     t * (-0.82215223 +
                     t * (0.17087277))))))))))
    if x >= 0:
        return ans
    else:
        return 2.0 - ans
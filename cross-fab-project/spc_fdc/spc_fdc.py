#!/usr/bin/env python3
"""
Cross-Fab Project: SPC and FDC Integration Module
Implements multivariate T² and SPE charts with FDC alarm routing and root-cause analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
import joblib
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SPCFDCMonitor:
    """SPC and FDC monitoring system with multivariate control charts."""
    
    def __init__(self, config: Dict = None):
        """Initialize SPC/FDC monitor."""
        self.config = config or {
            'alpha': 0.05,  # Significance level
            'ucl_t2': 12.59,  # Upper control limit for T² (95% confidence)
            'ucl_spe': 0.05,  # Upper control limit for SPE
            'window_size': 50,  # Moving window size
            'alarm_threshold': 3,  # Consecutive violations for alarm
            'pca_components': 0.95,  # PCA variance explained
            'drift_detection_window': 20,  # Window for drift detection
            'drift_threshold': 2.0  # Drift detection threshold
        }
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.config['pca_components'])
        self.cov_estimator = EmpiricalCovariance()
        
        self.is_fitted = False
        self.control_limits = {}
        self.alarm_history = []
        self.drift_history = []
        
    def fit_control_limits(self, data: pd.DataFrame) -> None:
        """Fit control limits from historical data."""
        print("Fitting SPC control limits...")
        
        # Prepare data
        X = data.select_dtypes(include=[np.number]).fillna(0)
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Calculate T² statistics
        t2_stats = self._calculate_t2_statistics(X_scaled)
        
        # Calculate SPE statistics
        spe_stats = self._calculate_spe_statistics(X_scaled, X_pca)
        
        # Set control limits
        self.control_limits = {
            't2_mean': np.mean(t2_stats),
            't2_std': np.std(t2_stats),
            't2_ucl': np.percentile(t2_stats, 95),
            'spe_mean': np.mean(spe_stats),
            'spe_std': np.std(spe_stats),
            'spe_ucl': np.percentile(spe_stats, 95),
            'feature_names': X.columns.tolist()
        }
        
        self.is_fitted = True
        print("Control limits fitted successfully!")
    
    def _calculate_t2_statistics(self, X_scaled: np.ndarray) -> np.ndarray:
        """Calculate Hotelling's T² statistics."""
        # Calculate sample covariance matrix
        cov_matrix = np.cov(X_scaled.T)
        cov_inv = np.linalg.pinv(cov_matrix)
        
        # Calculate T² for each sample
        t2_stats = []
        for i in range(len(X_scaled)):
            x = X_scaled[i:i+1].T
            t2 = x.T @ cov_inv @ x
            t2_stats.append(t2[0, 0])
        
        return np.array(t2_stats)
    
    def _calculate_spe_statistics(self, X_scaled: np.ndarray, X_pca: np.ndarray) -> np.ndarray:
        """Calculate Squared Prediction Error (SPE) statistics."""
        # Reconstruct data from PCA
        X_reconstructed = self.pca.inverse_transform(X_pca)
        
        # Calculate SPE
        spe_stats = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
        
        return spe_stats
    
    def monitor_process(self, data: pd.DataFrame) -> Dict:
        """Monitor process and detect violations."""
        if not self.is_fitted:
            raise ValueError("Control limits must be fitted before monitoring")
        
        # Prepare data
        X = data.select_dtypes(include=[np.number]).fillna(0)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        # Calculate statistics
        t2_stats = self._calculate_t2_statistics(X_scaled)
        spe_stats = self._calculate_spe_statistics(X_scaled, X_pca)
        
        # Check for violations
        t2_violations = t2_stats > self.control_limits['t2_ucl']
        spe_violations = spe_stats > self.control_limits['spe_ucl']
        
        # Detect alarms
        alarms = self._detect_alarms(t2_violations, spe_violations)
        
        # Detect drift
        drift_info = self._detect_drift(t2_stats, spe_stats)
        
        # Root cause analysis
        root_causes = self._analyze_root_causes(X_scaled, t2_violations, spe_violations)
        
        monitoring_result = {
            't2_statistics': t2_stats,
            'spe_statistics': spe_stats,
            't2_violations': t2_violations,
            'spe_violations': spe_violations,
            'alarms': alarms,
            'drift_info': drift_info,
            'root_causes': root_causes,
            'timestamp': pd.Timestamp.now()
        }
        
        # Store history
        self.alarm_history.append(alarms)
        self.drift_history.append(drift_info)
        
        return monitoring_result
    
    def _detect_alarms(self, t2_violations: np.ndarray, spe_violations: np.ndarray) -> Dict:
        """Detect alarms based on consecutive violations."""
        alarms = {
            't2_alarm': False,
            'spe_alarm': False,
            'combined_alarm': False,
            'alarm_count': 0,
            'alarm_details': []
        }
        
        # Check for consecutive T² violations
        t2_consecutive = self._count_consecutive_violations(t2_violations)
        if t2_consecutive >= self.config['alarm_threshold']:
            alarms['t2_alarm'] = True
            alarms['alarm_count'] += 1
            alarms['alarm_details'].append(f"T² alarm: {t2_consecutive} consecutive violations")
        
        # Check for consecutive SPE violations
        spe_consecutive = self._count_consecutive_violations(spe_violations)
        if spe_consecutive >= self.config['alarm_threshold']:
            alarms['spe_alarm'] = True
            alarms['alarm_count'] += 1
            alarms['alarm_details'].append(f"SPE alarm: {spe_consecutive} consecutive violations")
        
        # Combined alarm
        if alarms['t2_alarm'] or alarms['spe_alarm']:
            alarms['combined_alarm'] = True
        
        return alarms
    
    def _count_consecutive_violations(self, violations: np.ndarray) -> int:
        """Count consecutive violations from the end."""
        count = 0
        for i in range(len(violations) - 1, -1, -1):
            if violations[i]:
                count += 1
            else:
                break
        return count
    
    def _detect_drift(self, t2_stats: np.ndarray, spe_stats: np.ndarray) -> Dict:
        """Detect process drift."""
        window_size = self.config['drift_detection_window']
        threshold = self.config['drift_threshold']
        
        drift_info = {
            't2_drift': False,
            'spe_drift': False,
            'drift_magnitude': 0.0,
            'drift_direction': 'stable'
        }
        
        if len(t2_stats) < window_size:
            return drift_info
        
        # Check T² drift
        recent_t2 = np.mean(t2_stats[-window_size:])
        historical_t2 = np.mean(t2_stats[:-window_size]) if len(t2_stats) > window_size else recent_t2
        
        t2_drift = abs(recent_t2 - historical_t2) / historical_t2 if historical_t2 > 0 else 0
        
        if t2_drift > threshold:
            drift_info['t2_drift'] = True
            drift_info['drift_magnitude'] = t2_drift
            drift_info['drift_direction'] = 'increasing' if recent_t2 > historical_t2 else 'decreasing'
        
        # Check SPE drift
        recent_spe = np.mean(spe_stats[-window_size:])
        historical_spe = np.mean(spe_stats[:-window_size]) if len(spe_stats) > window_size else recent_spe
        
        spe_drift = abs(recent_spe - historical_spe) / historical_spe if historical_spe > 0 else 0
        
        if spe_drift > threshold:
            drift_info['spe_drift'] = True
            drift_info['drift_magnitude'] = max(drift_info['drift_magnitude'], spe_drift)
            drift_info['drift_direction'] = 'increasing' if recent_spe > historical_spe else 'decreasing'
        
        return drift_info
    
    def _analyze_root_causes(self, X_scaled: np.ndarray, t2_violations: np.ndarray, 
                           spe_violations: np.ndarray) -> Dict:
        """Analyze root causes of violations."""
        root_causes = {
            'contributing_factors': [],
            'severity_scores': {},
            'recommendations': []
        }
        
        if not np.any(t2_violations) and not np.any(spe_violations):
            return root_causes
        
        # Find violating samples
        violating_samples = X_scaled[t2_violations | spe_violations]
        
        if len(violating_samples) == 0:
            return root_causes
        
        # Calculate feature contributions
        feature_names = self.control_limits['feature_names']
        mean_values = np.mean(X_scaled, axis=0)
        violation_means = np.mean(violating_samples, axis=0)
        
        # Calculate deviations
        deviations = np.abs(violation_means - mean_values)
        
        # Rank features by contribution
        feature_contributions = list(zip(feature_names, deviations))
        feature_contributions.sort(key=lambda x: x[1], reverse=True)
        
        # Identify top contributing factors
        top_factors = feature_contributions[:5]  # Top 5 contributing factors
        
        root_causes['contributing_factors'] = [
            {'feature': factor[0], 'deviation': factor[1]} 
            for factor in top_factors
        ]
        
        # Calculate severity scores
        for factor in top_factors:
            severity = min(factor[1] / np.std(X_scaled[:, feature_names.index(factor[0])]), 3.0)
            root_causes['severity_scores'][factor[0]] = severity
        
        # Generate recommendations
        for factor in top_factors[:3]:  # Top 3 factors
            if factor[1] > 2.0:  # High deviation
                root_causes['recommendations'].append(
                    f"High deviation in {factor[0]}: Check process parameters and tool health"
                )
            elif factor[1] > 1.0:  # Medium deviation
                root_causes['recommendations'].append(
                    f"Medium deviation in {factor[0]}: Monitor closely and consider adjustment"
                )
        
        return root_causes
    
    def plot_control_charts(self, monitoring_result: Dict, output_dir: str = "spc_fdc") -> None:
        """Plot control charts for monitoring results."""
        os.makedirs(output_dir, exist_ok=True)
        
        t2_stats = monitoring_result['t2_statistics']
        spe_stats = monitoring_result['spe_statistics']
        t2_violations = monitoring_result['t2_violations']
        spe_violations = monitoring_result['spe_violations']
        
        # Create control charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # T² control chart
        axes[0, 0].plot(t2_stats, 'b-', linewidth=1, alpha=0.7)
        axes[0, 0].axhline(y=self.control_limits['t2_ucl'], color='r', linestyle='--', 
                          label=f"UCL = {self.control_limits['t2_ucl']:.2f}")
        axes[0, 0].scatter(np.where(t2_violations)[0], t2_stats[t2_violations], 
                          color='red', s=50, alpha=0.8, label='Violations')
        axes[0, 0].set_title('Hotelling\'s T² Control Chart')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('T² Statistic')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # SPE control chart
        axes[0, 1].plot(spe_stats, 'g-', linewidth=1, alpha=0.7)
        axes[0, 1].axhline(y=self.control_limits['spe_ucl'], color='r', linestyle='--',
                          label=f"UCL = {self.control_limits['spe_ucl']:.2f}")
        axes[0, 1].scatter(np.where(spe_violations)[0], spe_stats[spe_violations],
                          color='red', s=50, alpha=0.8, label='Violations')
        axes[0, 1].set_title('SPE Control Chart')
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('SPE Statistic')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # T² histogram
        axes[1, 0].hist(t2_stats, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].axvline(x=self.control_limits['t2_ucl'], color='r', linestyle='--',
                          label=f"UCL = {self.control_limits['t2_ucl']:.2f}")
        axes[1, 0].set_title('T² Distribution')
        axes[1, 0].set_xlabel('T² Statistic')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # SPE histogram
        axes[1, 1].hist(spe_stats, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].axvline(x=self.control_limits['spe_ucl'], color='r', linestyle='--',
                          label=f"UCL = {self.control_limits['spe_ucl']:.2f}")
        axes[1, 1].set_title('SPE Distribution')
        axes[1, 1].set_xlabel('SPE Statistic')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'control_charts.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Control charts saved to {output_dir}/control_charts.png")
    
    def save_monitor(self, output_dir: str = "spc_fdc") -> None:
        """Save monitor configuration and state."""
        os.makedirs(output_dir, exist_ok=True)
        
        monitor_data = {
            'config': self.config,
            'control_limits': self.control_limits,
            'alarm_history': self.alarm_history,
            'drift_history': self.drift_history,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(monitor_data, os.path.join(output_dir, 'spc_fdc_monitor.pkl'))
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(self.pca, os.path.join(output_dir, 'pca.pkl'))
        
        print(f"Monitor saved to {output_dir}/")

def main():
    """Main function to demonstrate SPC/FDC system."""
    print("Cross-Fab Project: SPC and FDC Integration")
    print("=" * 50)
    
    # Initialize SPC/FDC monitor
    monitor = SPCFDCMonitor()
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Generate normal process data
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_samples
    )
    
    # Add some violations
    violation_indices = np.random.choice(n_samples, size=10, replace=False)
    normal_data[violation_indices] += np.random.normal(0, 2, (10, n_features))
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    data = pd.DataFrame(normal_data, columns=feature_names)
    
    # Fit control limits
    monitor.fit_control_limits(data)
    
    # Monitor process
    result = monitor.monitor_process(data)
    
    # Print results
    print(f"T² violations: {np.sum(result['t2_violations'])}")
    print(f"SPE violations: {np.sum(result['spe_violations'])}")
    print(f"Alarms: {result['alarms']}")
    print(f"Drift info: {result['drift_info']}")
    
    # Plot control charts
    monitor.plot_control_charts(result)
    
    # Save monitor
    monitor.save_monitor()
    
    print("SPC/FDC demonstration complete!")

if __name__ == "__main__":
    main()
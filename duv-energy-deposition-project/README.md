# DUV Energy Deposition: Monte Carlo vs. Double Gaussian

## Project Overview

This project implements and compares Monte Carlo and Double Gaussian models for energy deposition in DUV (Deep Ultraviolet) masks. The project includes partial coherence modeling, flare effects, swing curves analysis, and comprehensive performance evaluation.

## Features

- **Monte Carlo Simulation**: Particle-based energy deposition modeling
- **Double Gaussian Model**: Analytical PSF-based energy deposition
- **Partial Coherence Modeling**: NA/illumination effects with pupil cut-off
- **Flare Modeling**: Long-tail effects via third Gaussian
- **Swing Curves Analysis**: Duty cycle and pitch dependence
- **Comprehensive Analysis**: Statistical comparison and validation
- **Performance Metrics**: Execution time, memory usage, throughput
- **PDF Report Generation**: Figures, equations, and metrics

## Project Structure

```
duv-energy-deposition-project/
├── monte_carlo/
│   └── monte_carlo_simulation.py
├── double_gaussian/
│   └── double_gaussian_model.py
├── partial_coherence/
│   └── partial_coherence_model.py
├── flare_modeling/
│   └── flare_modeling.py
├── swing_curves/
│   └── swing_curves.py
├── analysis/
│   └── analysis.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd duv-energy-deposition-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Analysis

```bash
python main.py
```

### Run Individual Modules

```bash
# Monte Carlo simulation
python monte_carlo/monte_carlo_simulation.py

# Double Gaussian model
python double_gaussian/double_gaussian_model.py

# Partial coherence modeling
python partial_coherence/partial_coherence_model.py

# Flare modeling
python flare_modeling/flare_modeling.py

# Swing curves analysis
python swing_curves/swing_curves.py

# Comprehensive analysis
python analysis/analysis.py
```

## Configuration

The project can be configured through the `config` dictionary in each module:

```python
config = {
    'wavelength': 193e-9,            # DUV wavelength in m
    'numerical_aperture': 0.85,      # Numerical aperture
    'illumination_sigma': 0.7,       # Illumination sigma
    'pupil_cutoff': 0.95,            # Pupil cutoff
    'partial_coherence': 0.7,        # Partial coherence
    'flare_level': 0.02,             # Flare level
    'flare_sigma': 0.1,              # Flare sigma
    'mask_size': (1000, 1000),       # Mask size in pixels
    'pixel_size': 1e-9,              # Pixel size in m
    'psf_size': 100,                 # PSF size in pixels
    'fft_size': 2048,                # FFT size
    'monte_carlo_particles': 1000000, # Number of Monte Carlo particles
    'monte_carlo_batches': 100,      # Number of Monte Carlo batches
    'double_gaussian_sigma1': 0.1,   # First Gaussian sigma
    'double_gaussian_sigma2': 0.5,   # Second Gaussian sigma
    'double_gaussian_weight1': 0.8,  # First Gaussian weight
    'double_gaussian_weight2': 0.2,  # Second Gaussian weight
    'duty_cycle_range': (0.1, 0.9),  # Duty cycle range
    'duty_cycle_steps': 50,          # Number of duty cycle steps
    'pitch_range': (100e-9, 1000e-9), # Pitch range in m
    'pitch_steps': 50,               # Number of pitch steps
    'swing_curve_points': 100,       # Number of swing curve points
    'analysis_points': 1000,         # Number of analysis points
    'output_directory': 'output',    # Output directory
    'temporary_directory': 'temp',   # Temporary directory
    'cache_directory': 'cache',      # Cache directory
    'log_directory': 'logs',         # Log directory
    'result_directory': 'results'    # Result directory
}
```

## Technical Details

### Monte Carlo Simulation

- **Particles**: 1,000,000 particles
- **Batches**: 100 batches for uncertainty quantification
- **Energy Distribution**: Exponential distribution
- **Spatial Distribution**: Uniform distribution
- **Uncertainty**: Bootstrap, statistical, and systematic

### Double Gaussian Model

- **PSF**: Two Gaussian components
- **Sigma 1**: 0.1 (narrow component)
- **Sigma 2**: 0.5 (wide component)
- **Weight 1**: 0.8 (main component)
- **Weight 2**: 0.2 (secondary component)
- **Convolution**: FFT-based for efficiency

### Partial Coherence Modeling

- **Illumination Sigma**: 0.7
- **Pupil Cutoff**: 0.95
- **NA Effects**: Included in PSF calculation
- **Coherence Length**: Calculated from parameters

### Flare Modeling

- **Flare Level**: 0.02 (2% of main signal)
- **Flare Sigma**: 0.1 (long-tail component)
- **Third Gaussian**: Additional long-tail effect
- **Spatial Correlation**: Included in analysis

### Swing Curves Analysis

- **Duty Cycle Range**: 0.1 to 0.9
- **Pitch Range**: 100 nm to 1000 nm
- **Contrast Calculation**: (Imax - Imin) / (Imax + Imin)
- **NILS Calculation**: Normalized Image Log Slope

## Performance Metrics

- **Execution Time**: Total time for complete analysis
- **Memory Usage**: Peak memory consumption
- **Throughput**: Operations per second
- **GPU Utilization**: GPU usage percentage
- **Memory Bandwidth**: Data transfer rate
- **Compute Intensity**: Operations per byte

## Output Files

- `comprehensive_analysis.png`: Main analysis plots
- `duv_energy_deposition_report.txt`: Detailed report
- `model_analysis.png`: Model-specific analysis
- `swing_curves_analysis.png`: Swing curves plots
- Performance metrics and statistics

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Pandas
- Scikit-learn (optional)

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Contact

For questions or support, please contact the project maintainers.

## Changelog

### Version 1.0.0
- Initial release
- Monte Carlo simulation
- Double Gaussian model
- Partial coherence modeling
- Flare modeling
- Swing curves analysis
- Comprehensive analysis
- Performance metrics
- PDF report generation
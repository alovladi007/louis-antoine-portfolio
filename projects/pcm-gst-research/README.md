# PCM GST Research

Phase Change Memory (PCM) simulation framework for Ge-Sb-Te (GST) alloys.

## Features

- **JMAK Crystallization Kinetics**: Johnson-Mehl-Avrami-Kolmogorov model for phase transformation
- **Electro-thermal Device Model**: Coupled electrical and thermal dynamics with Joule heating
- **Threshold Switching**: I-V characteristics with hysteresis and state-dependent behavior
- **Reliability Analysis**: Retention (Arrhenius) and endurance (Weibull) models
- **Pulse Simulations**: RESET (amorphization) and SET (crystallization) operations
- **Monte Carlo Variability**: Statistical analysis with parameter variations
- **2D Switching Maps**: Parameter space exploration for optimization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pcm-gst-research.git
cd pcm-gst-research

# Install dependencies
pip install -r requirements.txt

# Or use make
make install
```

## Quick Start

```python
from pcm.device import PCMDevice
from pcm.utils import plot_pulse_results

# Create device
device = PCMDevice()

# Simulate RESET pulse
t_reset, V_reset = device.create_reset_pulse(duration_s=100e-9, voltage_V=3.5)
result = device.simulate_voltage_pulse(V_reset, t_reset, X0=1.0)

# Plot results
figs = plot_pulse_results(result, "RESET")

print(f"Final crystalline fraction: {result['X'][-1]:.3f}")
print(f"Peak temperature: {result['T_K'].max():.0f} K")
```

## Running the Complete Pipeline

```bash
# Run all simulations and generate reports
make run

# Or run Python script directly
python run_simulation.py

# Run tests
make test

# Clean generated files
make clean
```

## Project Structure

```
pcm-gst-research/
├── pcm/                    # Core simulation modules
│   ├── __init__.py        # Package initialization
│   ├── kinetics.py        # JMAK crystallization model
│   ├── device.py          # Electro-thermal device model
│   ├── iv.py              # I-V threshold switching
│   ├── reliability.py     # Retention and endurance
│   └── utils.py           # Utility functions
├── notebooks/             # Jupyter notebooks
│   ├── analysis.ipynb     # Interactive analysis
│   └── experiments.ipynb  # Experimental simulations
├── tests/                 # Unit tests
│   ├── test_kinetics.py
│   ├── test_device.py
│   └── test_reliability.py
├── data/                  # Output data (CSV, plots)
├── reports/               # Generated reports
│   └── figures/          # Report figures
├── params.json           # Configuration parameters
├── requirements.txt      # Python dependencies
├── Makefile             # Build automation
├── Dockerfile           # Container configuration
└── README.md            # This file
```

## Key Results

### Material Properties
- **Active layer**: 50 nm GST
- **Resistance ratio**: >1000× (amorphous/crystalline)
- **Melting temperature**: 900 K
- **Crystallization onset**: 430 K

### JMAK Kinetics
- **Activation energy**: 2.1 eV
- **Prefactor**: 10¹³ s⁻¹
- **Avrami exponent**: 3.0 (3D growth)

### Device Performance
- **RESET**: 100 ns @ 3.5 V → Amorphous (high R)
- **SET**: 500 ns @ 1.8 V → Crystalline (low R)
- **Threshold voltage**: ~1.2 V (state-dependent)
- **Hold current**: 50 µA

### Reliability
- **Retention**: Arrhenius behavior, 10 years @ <85°C
- **Endurance**: 10⁶ cycles (Weibull characteristic life)

## Theory Background

### JMAK Crystallization
The Johnson-Mehl-Avrami-Kolmogorov equation describes isothermal phase transformation:
```
X(t) = 1 - exp(-(kt)^n)
```
where `k = k₀ × exp(-Ea/kT)` is the temperature-dependent rate constant.

### Electro-thermal Model
Lumped thermal model with Joule heating:
```
C_th × dT/dt = P - (T - T_amb)/R_th
P = I² × R(X,T)
```

### Phase Change Memory Operation
- **RESET**: High-power pulse melts GST and rapid quenching creates amorphous phase
- **SET**: Medium-power pulse heats above crystallization temperature for controlled crystallization

## Docker Support

```bash
# Build Docker image
docker build -t pcm-gst-research .

# Run simulations in container
docker run -v $(pwd)/data:/app/data pcm-gst-research
```

## CI/CD

The project includes GitHub Actions workflow for:
- Automated testing with pytest
- Code quality checks
- Documentation generation
- Docker image building

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this code in your research, please cite:
```bibtex
@software{pcm_gst_research,
  title = {PCM GST Research: Phase Change Memory Simulation Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/pcm-gst-research}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions or collaborations, please open an issue on GitHub.
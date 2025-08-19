# PCM GST Research Framework

## Advanced Phase Change Memory Simulation for Ge-Sb-Te Materials

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### 🚀 Overview

Comprehensive simulation framework for Phase Change Memory (PCM) devices based on Ge-Sb-Te (GST) chalcogenide materials. This research project implements advanced materials science models and device physics to simulate and optimize next-generation non-volatile memory technology.

### ✨ Key Features

- **JMAK Crystallization Kinetics**: Temperature-dependent phase transformation modeling
- **Electro-thermal Device Simulation**: Coupled electrical and thermal dynamics
- **Threshold Switching**: Electronic switching with hysteresis modeling
- **Reliability Analysis**: Arrhenius retention and Weibull endurance models
- **Monte Carlo Variability**: Statistical analysis with process variations
- **Interactive Visualizations**: Real-time parameter exploration

### 📊 Performance Metrics

| Metric | Value | Unit |
|--------|-------|------|
| Switching Speed | <10 | ns |
| Resistance Ratio | >1000 | × |
| Endurance | 10⁹ | cycles |
| Retention @ 85°C | >10 | years |
| Power Consumption | <1 | pJ/bit |
| Cell Size | 50×50 | nm² |

### 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pcm-gst-research.git
cd pcm-gst-research

# Install dependencies
pip install -r requirements.txt
```

### 🚦 Quick Start

```python
from pcm.device import PCMDevice
from pcm.utils import plot_pulse_results

# Create device
device = PCMDevice()

# Simulate RESET pulse (amorphization)
t_reset, V_reset = device.create_reset_pulse(duration_s=100e-9, voltage_V=2.5)
result_reset = device.simulate_voltage_pulse(V_reset, t_reset, X0=1.0)

# Simulate SET pulse (crystallization)
t_set, V_set = device.create_set_pulse(duration_s=200e-9, voltage_V=0.9)
result_set = device.simulate_voltage_pulse(V_set, t_set, X0=0.0)

# Plot results
figs = plot_pulse_results(result_reset, "RESET")
```

### 📁 Project Structure

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
│   └── pcm_simulations.ipynb
├── data/                  # Simulation outputs
├── reports/               # Generated reports
├── tests/                 # Unit tests
├── run_simulations.py     # Main simulation script
├── params.json            # Configuration parameters
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### 🔬 Physics Models

#### JMAK Crystallization Kinetics

The Johnson-Mehl-Avrami-Kolmogorov equation describes isothermal phase transformation:

```
X(t) = 1 - exp(-(kt)^n)
```

where:
- `X`: Crystalline fraction
- `k = k0 * exp(-Ea/kT)`: Rate constant
- `n`: Avrami exponent (3.0 for 3D growth)
- `Ea`: Activation energy (1.8 eV for GST)

#### Electro-thermal Model

Lumped thermal model with Joule heating:

```
C_th * dT/dt = P - (T - T_amb)/R_th
P = I²R = V²/(R + R_series)²
```

#### Resistance Mixing

Percolation-based parallel conductivity model:

```
σ = (X^p)/ρ_c + ((1-X)^p)/ρ_a
R = L/(A*σ)
```

### 📈 Simulation Capabilities

1. **Material Properties**
   - GST composition optimization
   - Temperature-dependent properties
   - Phase-dependent electrical/thermal characteristics

2. **Device Operations**
   - RESET pulse: Melt-quench amorphization
   - SET pulse: Controlled crystallization
   - Multi-level cell operation

3. **Reliability Studies**
   - Data retention vs temperature
   - Cycling endurance analysis
   - Failure mechanism modeling

4. **Process Variations**
   - Monte Carlo simulations
   - Statistical distributions
   - Yield prediction

### 🎯 Applications

- **Storage Class Memory**: Bridge between DRAM and NAND
- **Neuromorphic Computing**: Analog synaptic weights
- **Edge AI**: Low-power inference accelerators
- **Automotive**: High-temperature non-volatile storage
- **Space**: Radiation-hard memory systems

### 📚 Theory Background

Phase Change Memory exploits the large resistance contrast between amorphous and crystalline phases of chalcogenide materials:

1. **Amorphous Phase** (High Resistance)
   - Disordered atomic structure
   - High resistivity (~1 Ω·m)
   - Metastable at room temperature

2. **Crystalline Phase** (Low Resistance)
   - Ordered FCC structure
   - Low resistivity (~10⁻³ Ω·m)
   - Thermodynamically stable

3. **Phase Transitions**
   - **RESET**: Joule heating above Tm (900K) → rapid quench → amorphous
   - **SET**: Heating above Tc (430K) → crystallization via nucleation/growth

### 🔧 Configuration

Edit `params.json` to customize simulation parameters:

```json
{
  "thickness_m": 5e-08,        // Active layer thickness (50 nm)
  "area_m2": 2.5e-15,          // Device area (50×50 nm²)
  "Ea_cryst_eV": 1.8,          // Crystallization activation energy
  "Tm_K": 900.0,               // Melting temperature
  "Tc_K": 430.0,               // Crystallization onset
  "reset_V_V": 2.5,            // RESET voltage
  "set_V_V": 0.9               // SET voltage
}
```

### 📊 Running Simulations

```bash
# Run complete simulation suite
python run_simulations.py

# Results will be saved in:
# - data/*.png      (figures)
# - data/*.csv      (data files)
# - reports/*.txt   (summary reports)
```

### 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pcm --cov-report=html
```

### 📖 References

1. **JMAK Model**: Avrami, M. (1939). "Kinetics of Phase Change"
2. **GST Properties**: Wuttig, M. & Yamada, N. (2007). "Phase-change materials for rewriteable data storage"
3. **Device Physics**: Burr, G.W. et al. (2010). "Phase change memory technology"
4. **Threshold Switching**: Ielmini, D. & Zhang, Y. (2007). "Analytical model for subthreshold conduction"
5. **Reliability**: Papandreou, N. et al. (2011). "Drift-tolerant multilevel phase-change memory"

### 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

### 👥 Authors

- **Louis Antoine** - *Initial work* - [Portfolio](https://louisantoine.com)

### 🙏 Acknowledgments

- Stanford Nanoelectronics Lab for GST material parameters
- IBM Research for threshold switching models
- IMEC for reliability data

### 📬 Contact

For questions or collaborations:
- Email: contact@louisantoine.com
- LinkedIn: [Louis Antoine](https://linkedin.com/in/louisantoine)
- GitHub: [@louisantoine](https://github.com/louisantoine)

---

*Last Updated: December 2024*
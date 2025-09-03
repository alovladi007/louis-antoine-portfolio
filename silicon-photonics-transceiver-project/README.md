# Silicon Photonics Transceiver — 25–50 Gb/s (NRZ & PAM4)

![CI](https://github.com/louis-antoine/silicon-photonics-transceiver/workflows/CI/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)

## 🚀 What This Proves I Can Do

- **Semiconductor-Optics-RF Packaging Crossover**: End-to-end system design from photonic devices to high-speed electronics
- **Device Modeling**: Physics-based behavioral models for MZM, ring modulators, photodiodes, and TIAs
- **Communication Systems**: NRZ/PAM4 signaling, BER analysis, eye diagrams, and link budget optimization
- **Optimization Engineering**: Pareto-optimal design space exploration for power-efficient transceivers
- **Product Storytelling**: Technical documentation, interactive dashboards, and portfolio presentation

## 📸 Quick Demo

```bash
git clone https://github.com/louis-antoine/silicon-photonics-transceiver
cd silicon-photonics-transceiver
make demo  # Generates eye_25g_nrz.png + ber_sweep_25g.csv in ~2-3 minutes
make site  # Builds static portfolio site to docs/site/
```

## 🎯 Project Overview

Complete design and simulation framework for silicon photonics transceivers operating at:
- **25 Gb/s NRZ** (standard data center interconnect)
- **50 Gb/s NRZ/PAM4** (next-gen high-density links)

Targeting O-band (1310 nm) operation for data center applications with:
- Link distances: 500m - 2km
- Target BER: < 10⁻¹²
- Power optimization for energy-efficient operation

## 📊 Key Results

After running `make demo`, you'll find:
- `results/figs/eye_25g_nrz.png` - Eye diagram showing signal quality
- `results/tables/ber_sweep_25g.csv` - BER vs. power sweep data
- `results/tables/link_budget_report.md` - Complete link budget analysis

Full simulation (`make simulate`) adds:
- 50 Gb/s NRZ and PAM4 eye diagrams
- Pareto optimization curves
- Comprehensive BER analysis

## 🛠️ Technical Features

### Device Models
- **Mach-Zehnder Modulator (MZM)**: Vπ·L = 1.5 V·cm, ER > 6 dB
- **Ring Modulator**: High-Q resonant modulation
- **Photodiode**: R = 0.8 A/W with noise modeling
- **Transimpedance Amplifier**: Behavioral model with bandwidth/noise trade-offs

### System Simulation
- PRBS pattern generation (2¹⁵-1)
- Jitter, RIN, and thermal noise modeling
- Oversampled time-domain simulation (32 samples/bit)
- NRZ and PAM4 signaling support

### Optimization
- Multi-objective optimization for power vs. margin
- Automated parameter sweeps
- Pareto frontier visualization

## 🏗️ Repository Structure

```
.
├── src/link/          # Core simulation engine
├── src/verilogA/      # Portable device models
├── src/fdtd/          # Optional EM simulations (Meep)
├── notebooks/         # Interactive analysis
├── web/               # Next.js portfolio site
├── docs/              # Technical documentation
└── results/           # Generated outputs
```

## 🚦 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ (for web interface)
- Make

### Installation

```bash
# Clone repository
git clone https://github.com/louis-antoine/silicon-photonics-transceiver
cd silicon-photonics-transceiver

# Setup Python environment
conda env create -f env/environment.yml
conda activate photonics

# Install Node dependencies (optional, for web)
cd web && npm install && cd ..
```

### Usage

```bash
# Run quick demo (2-3 minutes)
make demo

# Full simulation suite
make simulate

# Parameter optimization
make optimize

# Launch interactive dashboard
make dashboard

# Build portfolio website
make site

# Generate technical paper
make paper
```

## 📈 Performance Metrics

| Metric | 25 Gb/s NRZ | 50 Gb/s PAM4 |
|--------|-------------|--------------|
| BER | < 10⁻¹² | < 10⁻¹² |
| Power Consumption | < 100 mW | < 150 mW |
| Extinction Ratio | > 6 dB | > 4 dB |
| Link Distance | 2 km | 500 m |

## 📚 Documentation

- [Technical Paper](docs/paper.md) - Detailed system analysis
- [API Reference](docs/api.md) - Module documentation
- [Portfolio Site](https://louis-antoine.github.io/silicon-photonics-transceiver) - Interactive results

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Louis Antoine**
- LinkedIn: [louis-antoine](https://linkedin.com/in/louis-antoine-333199a0)
- GitHub: [@louis-antoine](https://github.com/alovladi007)
- Email: louis.antoine@uconn.edu

## 🙏 Acknowledgments

- University of Connecticut ECE Department
- Silicon photonics research community
- Open-source simulation tools contributors

---
*This project demonstrates end-to-end system design capabilities for high-speed optical interconnects, from device physics to system optimization.*
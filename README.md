# 193nm DUV Resist Process Window & Stochastic Defect Reduction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This comprehensive graduate research project demonstrates the application of industrial Six Sigma methodologies to optimize 193nm deep ultraviolet (DUV) lithography processes in an academic cleanroom setting. By integrating stochastic resist modeling, statistical process control, and advanced data analysis, we achieved significant improvements in both process performance and cost efficiency.

### 🏆 Key Achievements

- **42% reduction** in stochastic bridge defects (0.087 → 0.051 defects/cm²)
- **53% improvement** in process capability (Cpk 1.12 → 1.71)
- **42% expansion** of process window (165 → 235 nm depth of focus)
- **$14,700/semester** cost avoidance through improved yield

## 🔬 Technical Implementation

### DMAIC Framework
- **Define:** Project charter and CTQ requirements
- **Measure:** Comprehensive baseline analysis with Gage R&R
- **Analyze:** Stochastic modeling and root cause analysis
- **Improve:** DOE optimization with response surface methodology
- **Control:** Real-time SPC with predictive analytics

### Technologies Used
- **Python** - Simulation and statistical analysis
- **Minitab** - Statistical process control and capability studies
- **JMP** - Real-time SPC dashboards
- **PROLITH** - Lithography process modeling
- **Jupyter** - Interactive demonstrations and documentation

## 📊 Visualizations & Simulations

This project includes comprehensive visualizations and interactive simulations:

### Generated Visualizations
- **Process Optimization Simulation** - 4-panel analysis showing optimization progress
- **Stochastic Defect Analysis** - Bridge defect distribution and process windows
- **Process Capability Study** - Complete Six Sigma capability analysis
- **SPC Dashboard** - Real-time statistical process control monitoring
- **3D Response Surface** - Advanced parameter optimization visualization
- **DOE Analysis** - Design of experiments with interaction plots

### Interactive Components
- Real-time parameter exploration with Jupyter widgets
- Live process window visualization
- Stochastic simulation with instant feedback
- Predictive modeling interface

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Complete Demo
```bash
python run_demo.py
```

### Individual Components
```bash
# Generate advanced graphs
python scripts/advanced_graphs.py

# Create video simulations
python scripts/video_simulations.py

# Launch interactive demo
jupyter lab notebooks/05_Interactive_Demo.ipynb
```

## 📁 Project Structure

```
193nm-DUV-Academic-Project/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── run_demo.py                        # Master demo script
├── PROJECT_OVERVIEW.md                # Detailed technical overview
├── data/                              # Experimental datasets
│   ├── baseline_measurements.csv
│   └── doe_results.csv
├── notebooks/                         # Jupyter notebooks
│   ├── 01_Baseline_Analysis.ipynb
│   └── 05_Interactive_Demo.ipynb
├── scripts/                           # Analysis and simulation scripts
│   ├── advanced_graphs.py
│   ├── video_simulations.py
│   ├── prolith_wrapper.py
│   └── spc_dashboard.jsl
├── minitab/                          # Minitab project files
│   └── macros/
├── results/                          # Project deliverables
│   └── STAR_Summary.md
└── documents/                        # Documentation and procedures
    ├── Work_Instructions/
    └── ISO_9001_Templates/
```

## 📈 Results Summary

### Quantitative Improvements
| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Bridge Defect Rate | 0.087 defects/cm² | 0.051 defects/cm² | 42% reduction |
| Process Capability (Cpk) | 1.12 | 1.71 | 53% improvement |
| Depth of Focus | 165 nm | 235 nm | 42% expansion |
| First-Pass Yield | 67% | 91% | 36% improvement |

### Optimal Process Conditions
- **Dose:** 25.4 mJ/cm² (optimized from 25.0)
- **Focus:** -25 nm offset (optimized from 0)
- **PEB Temperature:** 109.7°C (optimized from 110.0)

## 🎓 Academic Impact

- Framework adopted by 3 subsequent student cohorts
- SPIE Advanced Lithography conference paper
- Direct application to semiconductor manufacturing
- Comprehensive Six Sigma toolkit demonstration

## 🏭 Industrial Applications

- Advanced lithography process optimization
- Stochastic defect modeling and reduction
- Real-time process control implementation
- Quality system development

## 📊 Interactive Demo

The project includes a comprehensive interactive demonstration accessible through Jupyter notebooks. The demo features:

- Real-time parameter sliders for process exploration
- Live stochastic simulation with instant visualization
- Process window analysis with immediate feedback
- Statistical analysis with real-time calculations

## 🤝 Contributing

This project serves as a template for applying industrial quality methodologies in academic research environments. Feel free to adapt the framework for your own lithography optimization projects.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍🔬 Author

**Louis Antoine**
- LinkedIn: [louis-antoine-333199a0](https://www.linkedin.com/in/louis-antoine-333199a0)
- GitHub: [@alovladi007](https://github.com/alovladi007)
- Email: alovladi@gmail.com

## �� Acknowledgments

- University of Connecticut Nanofabrication Facility
- ASML for equipment access and technical support
- Graduate research team members
- Faculty advisors and cleanroom staff

---

*This project successfully demonstrates that industrial-grade quality engineering methodologies can be effectively implemented in academic research environments, achieving significant improvements in both process performance and student learning outcomes.*

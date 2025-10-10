# roch-cv-quantum

**Continuous-Variable Quantum Optics Research Suite**

Three research-ready projects tailored to University of Rochester's strengths in integrated photonics, CV quantum optics, and quantum communications & computing.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Projects

### A) Integrated CV Squeezer (SiN/TFLN)
On-chip squeezed light generation via χ⁽²⁾ OPA and χ⁽³⁾ Kerr nonlinearities. Includes ring resonator models, thermal detuning, and loss budgets.

**Key outputs:**
- On-chip vs detected squeezing spectra
- Trade-off curves: Q-factor, coupling, pump power
- Platform comparison: SiN rings vs TFLN waveguides

### B) PIC-based CV-QKD Link Optimizer
Gaussian-modulated coherent-state protocol with finite-size corrections, PIC loss budgets, and detector noise models.

**Key outputs:**
- Secret key rate heatmaps vs distance, β, ξ
- PIC insertion loss budget analysis
- Detector NEP sensitivity curves

### C) Loss-Tolerant CV Cluster-State Compiler
Graph state compilation with loss-aware phase optimization for measurement-based quantum computing.

**Key outputs:**
- Nullifier variance vs loss depth
- Fabrication tolerance maps
- Optimal phase settings for target graphs

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run all demos
make demos

# Or run individually:
python -m project_squeezer.cli --platform SiN --loss_db 0.5
python -m project_cvqkd.cli --distance_km 25 --beta 0.96
python -m project_cluster.cli --graph square --modes 8
```

## Repository Structure

```
roch-cv-quantum/
├── common/              # Shared utilities
│   ├── gaussian.py      # Covariance matrices & symplectic transforms
│   ├── noise_models.py  # Optical loss, detector QE, electronic noise
│   ├── fisher_crb.py    # Fisher information & Cramér-Rao bounds
│   └── plotting.py      # Publication-ready plot styling
├── project_squeezer/    # (A) CV Squeezer
│   ├── sim/            # Kerr/OPA models, ring parameters
│   ├── notebooks/      # Interactive analysis
│   ├── paper/          # LaTeX writeup
│   └── cli.py          # Command-line interface
├── project_cvqkd/       # (B) CV-QKD Link
│   ├── sim/            # GMCS protocol, key rates, PIC budgets
│   ├── notebooks/      # Link budget analysis
│   ├── paper/          # LaTeX writeup
│   └── cli.py
├── project_cluster/     # (C) Cluster States
│   ├── sim/            # Graph generation, nullifiers, compilation
│   ├── notebooks/      # Loss tolerance analysis
│   ├── paper/          # LaTeX writeup
│   └── cli.py
└── docs/               # Documentation & Rochester alignment
```

## Reproducibility

- **Fixed seeds**: All stochastic simulations use configurable seeds
- **Metadata tracking**: Each result includes `metadata.json` with parameters
- **Version control**: Results reference exact code commits
- **Auto-generated figures**: LaTeX papers auto-include figures from `/results`

## Papers

Each project includes a publication-ready LaTeX template:

```bash
cd project_squeezer/paper && make
cd project_cvqkd/paper && make
cd project_cluster/paper && make
```

Generated PDFs:
- `project_squeezer/paper/main.pdf` - Squeezing trade-offs & measured spectra
- `project_cvqkd/paper/main.pdf` - Key rates vs chip/detector budgets
- `project_cluster/paper/main.pdf` - Nullifier analysis & compiled settings

## Rochester Alignment

See [docs/rochester_alignment.md](docs/rochester_alignment.md) for how each project maps to:
- Institute of Optics strengths (integrated photonics, nonlinear optics)
- QIST (Quantum Information Science & Technology) research areas
- Potential collaborations with labs/advisors

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run specific test suite
pytest common/tests/
pytest project_squeezer/tests/
pytest project_cvqkd/tests/
pytest project_cluster/tests/
```

## Citation

```bibtex
@software{roch_cv_quantum_2025,
  title = {roch-cv-quantum: Continuous-Variable Quantum Optics Research Suite},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/roch-cv-quantum}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## References

Key papers implemented:
1. Kerr squeezing: [Phys. Rev. Lett. 127, 150503 (2021)]
2. GMCS CV-QKD: [Rev. Mod. Phys. 84, 621 (2012)]
3. Cluster states: [Phys. Rev. A 73, 012316 (2006)]

Full bibliography available in each project's `paper/refs.bib`.

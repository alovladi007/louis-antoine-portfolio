# Cross-Fab Project: Virtual Metrology + Run-to-Run APC

## Overview

This project implements a comprehensive Virtual Metrology (VM) and Run-to-Run (R2R) Advanced Process Control (APC) pipeline for semiconductor manufacturing. The system predicts post-process CD and overlay per wafer/zone and automatically tunes recipes with feed-forward from lithography and feedback to etch/deposition processes.

## Key Features

- **Virtual Metrology**: CatBoost-based models with conformal prediction for calibrated confidence intervals
- **R2R Control**: Double-EWMA and Kalman filter-based process control
- **SPC/FDC Integration**: Multivariate T² and SPE monitoring with root-cause analysis
- **Interactive Dashboard**: Real-time visualization and monitoring
- **Data Simulation**: Realistic fab data generation for testing and validation

## Performance Targets

- CD Prediction MAE: ≤ ±1.5 nm
- Overlay Prediction MAE: ≤ ±2.0 nm
- CD 3σ Reduction: ≥ 25%
- Overlay P95 Improvement: ≥ 20%
- Metrology Sampling Reduction: 50-70%
- Metrology Queue Time Reduction: 30%

## Project Structure

```
cross-fab-project/
├── data/                   # Data simulation and generation
│   └── simulator.py       # Fab data simulator
├── vm/                    # Virtual Metrology
│   └── virtual_metrology.py
├── apc/                   # Run-to-Run Control
│   └── r2r_control.py
├── spc_fdc/              # SPC/FDC Integration
│   └── spc_fdc.py
├── dash/                 # Dashboard
│   └── dashboard.py
├── main.py              # Main integration script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cross-fab-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the complete pipeline:

```bash
python main.py
```

This will:
1. Generate simulation data
2. Train VM models
3. Run R2R control simulation
4. Perform SPC/FDC monitoring
5. Generate interactive dashboard
6. Create technical report

## Individual Module Usage

### Data Simulator
```python
from data.simulator import FabDataSimulator

simulator = FabDataSimulator()
data = simulator.generate_all_data()
simulator.save_data("output/data")
```

### Virtual Metrology
```python
from vm.virtual_metrology import VirtualMetrology

vm = VirtualMetrology()
vm.train_models(features, targets)
predictions = vm.predict_cd(features)
```

### R2R Control
```python
from apc.r2r_control import R2RController

controller = R2RController()
action = controller.calculate_control_action(cd_pred, overlay_pred)
```

### SPC/FDC Monitoring
```python
from spc_fdc.spc_fdc import SPCFDCMonitor

monitor = SPCFDCMonitor()
monitor.fit_control_limits(data)
result = monitor.monitor_process(data)
```

### Dashboard
```python
from dash.dashboard import CrossFabDashboard

dashboard = CrossFabDashboard()
dashboard.load_data("data")
html_file = dashboard.create_dashboard_html("output/dash")
```

## Configuration

The system can be configured through a JSON configuration file:

```json
{
    "random_seed": 42,
    "n_wafers": 200,
    "n_fields": 25,
    "n_zones": 9,
    "wavelength": 193.0,
    "na": 0.85,
    "sigma_illum": 0.7,
    "chamber_count": 4,
    "reticle_count": 3,
    "lot_age_days": 30,
    "process_variation": 0.1,
    "metrology_sampling_rate": 0.3
}
```

## Output Files

After running the pipeline, the following files will be generated:

- `output/data/` - Generated simulation data (CSV files)
- `output/vm/` - Trained VM models and feature importance
- `output/apc/` - R2R controller state and control history
- `output/spc_fdc/` - SPC/FDC monitoring results and control charts
- `output/dash/dashboard.html` - Interactive dashboard
- `output/technical_report.html` - Comprehensive technical report

## Technical Details

### Virtual Metrology
- Uses CatBoost gradient boosting for robust prediction
- Implements conformal prediction for calibrated confidence intervals
- Features include EWMA'd tool traces, per-field statistics, and chamber drift counters
- Supports both CD and overlay prediction

### R2R Control
- Double-EWMA algorithm for process control
- Kalman filter for state estimation
- Feed-forward from lithography to etch/deposition
- Chamber matching and recipe constraints

### SPC/FDC Integration
- Multivariate T² and SPE control charts
- Automatic alarm detection and routing
- Root-cause analysis via SHAP feature importance
- Drift detection and trend analysis

## Dependencies

- Python 3.8+
- NumPy, Pandas, Matplotlib
- Scikit-learn, SciPy
- CatBoost
- SHAP
- Plotly
- Joblib

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or support, please contact the project maintainers.

## Acknowledgments

This project was developed as part of advanced semiconductor manufacturing research and represents cutting-edge work in virtual metrology and process control.
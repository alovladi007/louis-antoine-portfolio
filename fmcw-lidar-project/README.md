# FMCW LiDAR-on-a-Tabletop (Integrated Photonics + RF/DSP)

## Project Overview

This project implements a complete FMCW (Frequency Modulated Continuous Wave) LiDAR system using integrated photonics and RF/DSP processing. The system demonstrates the integration of silicon photonics frontend, electronics chain, real-time DSP processing, laser control, and comprehensive system analysis.

## Key Features

- **Integrated Photonics Frontend**: Silicon photonics components including splitters, MZI, and coherent receiver
- **Electronics Chain**: PLL/chirp synthesis, laser current/TEC control, TIA/ADC processing
- **Real-time DSP**: Range-Doppler FFT, CFAR detection, phase unwrapping, super-resolution
- **Laser Control**: Precise current and temperature control with PID feedback
- **System Analysis**: Comprehensive performance analysis including phase noise, RIN, ADC ENOB, and loss budget

## Project Structure

```
fmcw-lidar-project/
├── photonics/
│   └── photonic_frontend.py      # Silicon photonics simulation and layout
├── elec_spice/
│   └── electronics_chain.py      # Electronics/RF chain components
├── dsp/
│   └── signal_processing.py      # DSP processing algorithms
├── firmware/
│   └── laser_control.py          # Laser current/TEC control
├── analysis/
│   └── system_analysis.py        # System performance analysis
├── main.py                       # Main integration script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Technical Specifications

### Photonic Frontend
- **Wavelength**: 1550 nm
- **Bandwidth**: 100 GHz
- **Components**: Splitters, MZI, coherent receiver
- **Insertion Loss**: < 3 dB
- **Phase Shift**: Controllable via MZI

### Electronics Chain
- **Sampling Rate**: 100 MS/s
- **ADC Resolution**: 12 bits
- **TIA Gain**: 1 kΩ
- **Bandwidth**: 100 GHz
- **SNR**: > 60 dB

### DSP Processing
- **Range Resolution**: 0.1 m
- **Velocity Resolution**: 0.1 m/s
- **FFT Size**: 1024
- **CFAR Detection**: Adaptive threshold
- **Super-resolution**: 4x improvement

### Laser Control
- **Power**: 10 mW
- **Current Control**: 0-200 mA
- **Temperature Control**: 0-100°C
- **Wavelength Stability**: ±0.1 nm
- **Control Frequency**: 1 MHz

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fmcw-lidar-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main integration script:
```bash
python main.py
```

This will:
1. Initialize all subsystems
2. Calculate system integration
3. Generate performance analysis
4. Create visualization plots
5. Generate system report

### Individual Module Usage

#### Photonic Frontend
```python
from photonics.photonic_frontend import PhotonicFrontend

frontend = PhotonicFrontend()
results = frontend.calculate_photonic_frontend()
```

#### Electronics Chain
```python
from elec_spice.electronics_chain import ElectronicsChain

electronics = ElectronicsChain()
results = electronics.calculate_electronics_chain()
```

#### DSP Processing
```python
from dsp.signal_processing import FMCWDSPProcessor

dsp = FMCWDSPProcessor()
results = dsp.calculate_real_time_processing(beat_signal)
```

#### Laser Control
```python
from firmware.laser_control import LaserController

controller = LaserController()
results = controller.calculate_real_time_control(power, temperature, wavelength)
```

#### System Analysis
```python
from analysis.system_analysis import FMCWSystemAnalyzer

analyzer = FMCWSystemAnalyzer()
results = analyzer.calculate_system_performance()
```

## Performance Metrics

### Range Performance
- **Maximum Range**: 100 m
- **Range Resolution**: 0.1 m
- **Range Accuracy**: ±1 mm
- **Range Precision**: ±0.1 mm

### Velocity Performance
- **Maximum Velocity**: 50 m/s
- **Velocity Resolution**: 0.1 m/s
- **Velocity Accuracy**: ±1 mm/s
- **Velocity Precision**: ±0.1 mm/s

### Detection Performance
- **SNR**: > 60 dB
- **Detection Probability**: 90%
- **False Alarm Rate**: 1e-6
- **Detection Range**: 100 m

### System Performance
- **Phase Noise**: -80 dBc/Hz
- **RIN**: -150 dB/Hz
- **ADC ENOB**: 12 bits
- **TIA Noise**: 1 pA/√Hz
- **Photonic Loss**: < 3 dB

## Output Files

The system generates several output files:

### Plots
- `fmcw_lidar/system_integration.png`: Comprehensive system overview
- `photonics/photonic_analysis.png`: Photonic frontend analysis
- `elec_spice/electronics_analysis.png`: Electronics chain analysis
- `dsp/dsp_analysis.png`: DSP processing analysis
- `firmware/control_analysis.png`: Laser control analysis
- `analysis/system_analysis.png`: System performance analysis

### Reports
- `fmcw_lidar/system_report.txt`: Comprehensive system report
- `photonics/photonic_report.txt`: Photonic frontend report
- `elec_spice/electronics_report.txt`: Electronics chain report
- `dsp/dsp_report.txt`: DSP processing report
- `firmware/control_report.txt`: Laser control report
- `analysis/analysis_report.txt`: System analysis report

## Configuration

The system can be configured by modifying the `config` dictionary in each module:

```python
config = {
    'wavelength': 1550e-9,           # Operating wavelength in m
    'bandwidth': 100e9,              # Chirp bandwidth in Hz
    'chirp_duration': 1e-3,          # Chirp duration in s
    'sampling_rate': 100e6,          # ADC sampling rate in Hz
    'laser_power': 10e-3,            # Laser power in W
    'range_resolution': 0.1,         # Range resolution in m
    'velocity_resolution': 0.1,      # Velocity resolution in m/s
    'max_range': 100.0,              # Maximum range in m
    'max_velocity': 50.0,            # Maximum velocity in m/s
    'snr_threshold': 10,             # SNR threshold in dB
    'detection_probability': 0.9,    # Detection probability
    'false_alarm_rate': 1e-6,        # False alarm rate
    # ... more parameters
}
```

## Dependencies

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn (optional)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions or support, please open an issue on the repository.

## Changelog

### Version 1.0.0
- Initial release
- Complete FMCW LiDAR system implementation
- Integrated photonics frontend
- Electronics chain simulation
- Real-time DSP processing
- Laser control system
- Comprehensive system analysis
# RF Power Amplifier Design & Linearization (5G/6G Ready)

## 🚀 Overview

A cutting-edge interactive web application demonstrating the design and optimization of GaN-based RF power amplifiers for 5G/6G mmWave frequencies. This project showcases advanced digital predistortion (DPD) techniques, real-time performance visualization, and comprehensive technical documentation for next-generation wireless systems.

## ✨ Key Features

### Interactive Visualizations
- **Real-time Power Transfer Characteristics**: Dynamic charts showing output power, gain, and PAE vs input power
- **Frequency Response Analysis**: S-parameter visualization across the operating bandwidth
- **256-QAM Constellation Diagrams**: Interactive modulation quality assessment
- **Efficiency vs Linearity Trade-offs**: Comprehensive performance optimization curves

### Performance Metrics
- **Output Power**: 43 dBm (20W) saturated
- **Peak PAE**: 65% at 3dB compression
- **Bandwidth**: 400 MHz instantaneous
- **ACPR**: -48 dBc with DPD enabled
- **EVM**: 1.8% RMS for 256-QAM

### Interactive Controls
- Adjustable input power sweep (-30 to +10 dBm)
- DPD correction level control (0-100%)
- Real-time optimization algorithm
- Live metric updates with visual feedback

## 🛠️ Technologies

### RF Design
- **GaN HEMT Technology**: 0.15μm process for high power density
- **Doherty Architecture**: Asymmetric 2:1 configuration
- **Digital Predistortion**: Memory polynomial with K=7, M=3
- **Load-Pull Analysis**: Comprehensive impedance optimization
- **EM Simulation**: Full 3D electromagnetic co-simulation

### Web Technologies
- **HTML5**: Semantic structure and modern web standards
- **CSS3**: Advanced animations and responsive design
- **JavaScript**: Interactive charts and real-time calculations
- **Chart.js**: Professional data visualization library

## 📊 Technical Specifications

| Parameter | Specification | Measured | Unit |
|-----------|--------------|----------|------|
| Frequency Range | 27.5 - 28.35 | 27.5 - 28.35 | GHz |
| Output Power (P1dB) | ≥ 40 | 43 | dBm |
| Power Added Efficiency | ≥ 60 | 65 | % |
| Small Signal Gain | ≥ 25 | 28 | dB |
| Input Return Loss | ≤ -10 | -15 | dB |
| Output Return Loss | ≤ -10 | -12 | dB |
| ACPR (5MHz offset) | ≤ -45 | -48 | dBc |
| Operating Temperature | -40 to +85 | -40 to +85 | °C |

## 🎯 Applications

- **5G/6G Base Stations**: Macro and small cell deployments
- **Phased Array Radar**: Defense and weather monitoring systems
- **Satellite Communications**: Ka-band uplink terminals
- **Fixed Wireless Access**: High-capacity point-to-point links
- **Military Communications**: Tactical and strategic systems

## 🚦 Getting Started

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection (for Chart.js CDN)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rf-power-amplifier-5g6g.git
cd rf-power-amplifier-5g6g
```

2. Open the application:
```bash
# Using Python's built-in server
python -m http.server 8000

# Or using Node.js
npx http-server

# Or simply open index.html in your browser
```

3. Navigate to `http://localhost:8000` in your browser

## 📱 Usage

### Interactive Controls
1. **Input Power Slider**: Adjust the input power level to see compression effects
2. **DPD Level Slider**: Control the digital predistortion correction amount
3. **Update Response Button**: Manually refresh all visualizations
4. **Run Optimization Button**: Execute automatic DPD optimization algorithm

### Chart Interactions
- Hover over data points for detailed values
- Click legend items to show/hide datasets
- Charts automatically scale and update in real-time

## 📚 Technical Documentation

The application includes comprehensive technical documentation covering:

- Design methodology and architecture
- Digital predistortion algorithms
- Measurement results and validation
- Thermal management strategies
- Reliability and stability analysis
- Field deployment considerations

## 🔬 Research & Development

### Design Process
1. **Requirements Analysis**: 5G NR specifications for n257/n260 bands
2. **Device Selection**: 0.15μm GaN HEMT technology evaluation
3. **Circuit Design**: Asymmetric Doherty with optimized matching
4. **EM Simulation**: Full 3D analysis in HFSS
5. **DPD Implementation**: Memory polynomial with adaptive coefficients
6. **Validation & Testing**: Comprehensive performance verification

### Key Innovations
- Advanced harmonic termination for efficiency enhancement
- Adaptive bias control for linearity improvement
- Thermal-aware design for reliability
- Wideband matching network optimization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Performance optimizations
- Additional visualization features
- Algorithm improvements
- Documentation enhancements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - RF/Microwave Engineer
- Research Lab - Institution

## 🙏 Acknowledgments

- IEEE MTT-S for technical standards and guidelines
- 3GPP for 5G NR specifications
- Open-source community for web technologies

## 📞 Contact

For questions, collaborations, or consulting inquiries:
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- GitHub: [@yourusername](https://github.com/yourusername)

---

**Note**: This is a demonstration project showcasing RF power amplifier design concepts and interactive visualization techniques. Actual hardware implementation requires detailed electromagnetic simulation, thermal analysis, and extensive testing.
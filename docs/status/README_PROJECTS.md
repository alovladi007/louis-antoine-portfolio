# Semiconductor Simulation Projects

This repository contains two comprehensive semiconductor-related projects:

## 1. Photolithography & Optical Metrology Simulation

A complete Python/MATLAB simulation framework for semiconductor photolithography processes.

### Features Implemented

#### ✅ Mask Pattern Generation (`src/mask_generation.py`)
- Multiple pattern types (line/space, contact holes, SRAM cells)
- Resolution test patterns
- Phase shift mask generation
- Defect injection capabilities
- Alignment marks
- GDS export support

#### ✅ Optical Proximity Correction (`src/opc.py`)
- Rule-based OPC with serifs and assist features
- Model-based OPC with iterative optimization
- Inverse Lithography Technology (ILT)
- Line end correction
- Corner correction
- OPC evaluation metrics (EPE, CD, pattern fidelity)

#### ✅ Defect Inspection Analysis (`src/defect_inspection.py`)
- Die-to-die inspection
- 10+ defect type classification
- Pattern fidelity analysis
- Line edge/width roughness measurement
- Comprehensive inspection reporting
- Defect visualization

#### ✅ Fourier Optics Simulation (`src/fourier_optics.py`)
- PSF and MTF calculations
- Multiple illumination modes (conventional, annular, quadrupole, dipole)
- Aberration modeling with Zernike polynomials
- Partially coherent imaging (Hopkins method)
- Focus-exposure matrix simulation
- Process window calculation

#### ✅ Monte Carlo Simulation (`src/monte_carlo.py`)
- Photon shot noise simulation
- Acid generation and diffusion
- Resist development modeling
- Line edge roughness prediction
- Process variation analysis
- Parallel simulation support

### Installation & Usage

```bash
cd photolithography-simulation
pip install -r requirements.txt

# Run examples
python src/mask_generation.py
python src/opc.py
python src/defect_inspection.py
python src/fourier_optics.py
python src/monte_carlo.py
```

### Key Technologies
- **Python**: Core simulation engine
- **NumPy/SciPy**: Numerical computations
- **OpenCV**: Image processing
- **Matplotlib**: Visualization
- **MATLAB**: Integration examples (pending)

---

## 2. Defect Detection & Yield Prediction with ML

Advanced machine learning system for semiconductor defect classification and yield prediction.

### Features Implemented

#### ✅ Data Generation (`src/data_generator.py`)
- Synthetic wafer map generation
- Multiple defect pattern types (particle, scratch, cluster, ring, edge)
- Realistic wafer image synthesis
- Inspection artifact simulation
- Complete dataset generation pipeline

#### ✅ CNN Models (`src/cnn_models.py`)
- Custom DefectCNN architecture
- ResNet-based transfer learning (ResNet18/34/50/101)
- EfficientNet classifier
- Attention mechanism CNN
- Multi-scale CNN for various defect sizes
- Model factory pattern

#### ✅ Bayesian Models (`src/bayesian_models.py`)
- Bayesian CNN with Pyro
- MC Dropout for uncertainty quantification
- Rare defect detector using GMM
- Active learning sample selection
- Uncertainty estimation methods

#### 🔄 Pending Implementation
- Yield prediction models
- Streamlit web interface
- Model training pipelines
- Real-time inference system

### Installation & Usage

```bash
cd defect-detection-ml
pip install -r requirements.txt

# Run examples
python src/data_generator.py
python src/cnn_models.py
python src/bayesian_models.py
```

### Key Technologies
- **PyTorch**: Deep learning framework
- **Pyro**: Probabilistic programming
- **Scikit-learn**: Classical ML algorithms
- **Streamlit**: Web interface (pending)
- **MLflow**: Experiment tracking

---

## Project Status

### Completed ✅
- Photolithography simulation core modules (5/6)
- ML project structure and data generation
- CNN architectures for defect classification
- Bayesian methods for rare defects
- Monte Carlo simulation engine

### In Progress 🔄
- Yield prediction models
- Streamlit web interface
- MATLAB integration examples
- Comprehensive documentation

### Repository Structure

```
/workspace/
├── photolithography-simulation/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── mask_generation.py
│   │   ├── opc.py
│   │   ├── defect_inspection.py
│   │   ├── fourier_optics.py
│   │   └── monte_carlo.py
│   ├── requirements.txt
│   └── matlab/  (pending)
│
└── defect-detection-ml/
    ├── src/
    │   ├── __init__.py
    │   ├── data_generator.py
    │   ├── cnn_models.py
    │   └── bayesian_models.py
    ├── requirements.txt
    └── streamlit_app/  (pending)
```

## Next Steps

1. Complete yield prediction models
2. Build Streamlit web interface
3. Add MATLAB integration examples
4. Create comprehensive test suite
5. Add real semiconductor dataset loaders
6. Implement model deployment pipeline

## Contributing

Both projects are actively being developed. Each module includes example usage in the `if __name__ == "__main__"` section for testing and demonstration.

## License

MIT License (or as specified by the repository owner)
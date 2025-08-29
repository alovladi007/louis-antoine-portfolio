# 🔬 Defect Detection & Yield Prediction with ML

A comprehensive machine learning system for semiconductor defect classification and yield prediction, featuring state-of-the-art CNN architectures, Bayesian methods, and interactive visualizations.

## 🚀 Quick Start

### Installation

```bash
# Navigate to project directory
cd defect-detection-ml

# Install dependencies
pip install -r requirements.txt
```

### Run the Demo

```bash
# Run comprehensive demo with visualizations
python demo.py

# This will:
# - Generate synthetic wafer defect maps
# - Create realistic wafer images
# - Demonstrate all CNN architectures
# - Show uncertainty quantification
# - Perform active learning
# - Generate sample datasets
```

### Launch Web Interface

```bash
# Start Streamlit application
streamlit run streamlit_app.py

# Open browser to http://localhost:8501
```

### Train a Model

```bash
# Train with default settings (DefectCNN, 30 epochs)
python train.py

# Train specific model
python train.py --model resnet --epochs 50 --batch_size 32

# Available models: defect_cnn, resnet, efficientnet, attention_cnn, multiscale_cnn, mc_dropout
```

## 📁 Project Structure

```
defect-detection-ml/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── data_generator.py           # Synthetic data generation
│   ├── cnn_models.py              # CNN architectures
│   ├── bayesian_models.py         # Bayesian ML & uncertainty
│   └── yield_prediction.py        # Yield prediction models
├── demo.py                        # Comprehensive demonstration
├── streamlit_app.py              # Web interface
├── train.py                      # Training pipeline
├── models/                       # Saved model weights
├── data/                        # Generated datasets
├── outputs/                     # Visualization outputs
└── requirements.txt            # Dependencies
```

## 🎯 Features

### Data Generation
- **Synthetic Wafer Maps**: 6 defect pattern types (particle, scratch, cluster, ring, edge, pattern)
- **Realistic Wafer Images**: Full wafer visualization with texture and artifacts
- **Defect Image Generation**: Individual defect images for CNN training
- **Complete Dataset Builder**: Automated dataset creation with train/test splits

### Model Architectures
1. **DefectCNN**: Custom CNN with batch normalization
2. **ResNet**: Transfer learning with ResNet-18/34/50/101
3. **EfficientNet**: Efficient architecture for resource-constrained deployment
4. **AttentionCNN**: Spatial attention mechanisms for defect localization
5. **MultiScaleCNN**: Multi-scale feature extraction for various defect sizes

### Advanced Features
- **Bayesian Uncertainty Quantification**: MC Dropout and Bayesian NNs
- **Rare Defect Detection**: GMM-based anomaly detection
- **Active Learning**: Intelligent sample selection strategies
- **Yield Prediction**: Process parameter optimization
- **Time Series Analysis**: Yield trend forecasting

## 💻 Usage Examples

### Generate Synthetic Data

```python
from src.data_generator import DefectDataGenerator, DefectType

# Create generator
generator = DefectDataGenerator(wafer_size=300, die_size=(10, 10))

# Generate wafer map with specific defect type
wafer_map = generator.generate_wafer_map(
    defect_type=DefectType.CLUSTER,
    defect_density=0.05
)

print(f"Yield Rate: {wafer_map.yield_rate:.1%}")
print(f"Defect Count: {wafer_map.defect_count}")

# Generate complete dataset
dataset = generator.generate_dataset(
    n_samples=1000,
    defect_classes=["particle", "scratch", "void", "bridge", "none"],
    train_split=0.8
)
```

### Train and Evaluate Models

```python
from src.cnn_models import create_model
import torch

# Create model
model = create_model('resnet', num_classes=5, resnet_version='resnet50')

# Train (see train.py for complete pipeline)
# ... training code ...

# Inference
input_tensor = torch.randn(1, 1, 224, 224)
output = model(input_tensor)
prediction = torch.argmax(output, dim=1)
```

### Uncertainty Quantification

```python
from src.bayesian_models import MCDropoutClassifier

# Create model with dropout
model = MCDropoutClassifier(num_classes=5, dropout_rate=0.5)

# Get predictions with uncertainty
mean_pred, uncertainty = model.predict_with_uncertainty(
    input_tensor, 
    n_samples=50  # Monte Carlo samples
)

print(f"Prediction confidence: {mean_pred.max():.2%}")
print(f"Uncertainty: {uncertainty.item():.4f}")
```

### Yield Prediction

```python
from src.yield_prediction import YieldPredictor, ProcessParameters

# Create predictor
predictor = YieldPredictor(model_type='random_forest')

# Train on historical data
predictor.train(X_features, y_yields)

# Predict yield for new wafer
process_params = ProcessParameters(
    temperature=150,
    pressure=500,
    exposure_time=50
)

features = predictor.extract_features(wafer_map, process_params)
yield_pred, confidence = predictor.predict(features.reshape(1, -1))

print(f"Predicted Yield: {yield_pred[0]:.1%} ± {confidence[0]:.2%}")
```

## 📊 Web Interface Features

The Streamlit app provides:

1. **Overview Dashboard**: Project metrics and capabilities
2. **Data Generation**: Interactive wafer map and defect image creation
3. **Model Testing**: Load models and run inference
4. **Analytics**: Yield trends and defect statistics
5. **Predictions**: Yield forecasting with confidence intervals
6. **Documentation**: Complete API reference and examples

## 🎓 Training Pipeline

The `train.py` script provides a complete training pipeline with:

- Automatic data generation or loading
- Train/validation/test splits
- Model checkpointing
- Early stopping
- Learning rate scheduling
- Comprehensive evaluation metrics
- Confusion matrix visualization
- Training history plots

### Training Commands

```bash
# Basic training
python train.py

# Advanced options
python train.py \
    --model resnet \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --n_samples 5000 \
    --device cuda

# Load existing data
python train.py --load_data
```

## 📈 Performance Metrics

After training, the system provides:

- **Overall Accuracy**: Test set performance
- **Per-Class Metrics**: Precision, recall, F1-score
- **Confusion Matrix**: Visual error analysis
- **Training Curves**: Loss and accuracy progression
- **Uncertainty Calibration**: For Bayesian models

## 🔧 Configuration

### Model Parameters

Edit model configurations in `src/cnn_models.py`:

```python
# Example: Modify DefectCNN architecture
class DefectCNN(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        # Adjust architecture here
        ...
```

### Data Generation Settings

Customize in `src/data_generator.py`:

```python
@dataclass
class ProcessParameters:
    temperature: float = 293.15  # K
    exposure_dose: float = 30.0  # mJ/cm²
    # Add more parameters
```

## 📦 Output Files

After running demos and training:

```
outputs/
├── wafer_defect_patterns.png      # Different defect types
├── synthetic_wafer_images.png     # Realistic wafer images
├── defect_statistics.png          # Statistical analysis
├── cnn_architectures.png          # Model comparisons
├── uncertainty_quantification.png # Bayesian uncertainty
├── active_learning_selection.png  # Sample selection
└── sample_dataset.png             # Training samples

models/
├── best_model.pth                 # Best trained model
├── training_history.png           # Training curves
├── confusion_matrix.png           # Test set confusion matrix
└── training_summary.json          # Training metadata

data/
├── train_images.npy               # Training images
├── train_labels.npy               # Training labels
├── test_images.npy                # Test images
└── test_labels.npy                # Test labels
```

## 🚦 System Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- GPU optional but recommended for training
- ~500MB disk space for models and data

## 📝 API Reference

### Data Generation API

- `DefectDataGenerator`: Main data generation class
- `SyntheticWaferGenerator`: Realistic wafer image generation
- `DefectType`: Enum of defect types

### Model API

- `create_model()`: Factory function for model creation
- `MCDropoutClassifier`: Bayesian CNN with uncertainty
- `ActiveLearningSelector`: Sample selection strategies

### Yield Prediction API

- `YieldPredictor`: Main yield prediction model
- `YieldOptimizer`: Process parameter optimization
- `TimeSeriesYieldPredictor`: Time series forecasting

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the correct directory and have installed requirements
2. **Memory Issues**: Reduce batch size or number of samples
3. **Streamlit Not Loading**: Check port 8501 is not in use
4. **Training Too Slow**: Use GPU or reduce model complexity

### Getting Help

- Check the documentation in the Streamlit app
- Review example code in `demo.py`
- Examine docstrings in source files

## 🎯 Next Steps

1. **Extend Defect Types**: Add new defect patterns in `data_generator.py`
2. **Custom Models**: Implement new architectures in `cnn_models.py`
3. **Real Data Integration**: Replace synthetic data with actual wafer images
4. **Production Deployment**: Export models to ONNX for deployment
5. **Federated Learning**: Implement distributed training across fabs

## 📄 License

MIT License - See repository root for details

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit for the web interface framework
- Scikit-learn for classical ML algorithms

---

**Created by**: Louis Antoine
**Repository**: [github.com/alovladi007/louis-antoine-portfolio](https://github.com/alovladi007/louis-antoine-portfolio)
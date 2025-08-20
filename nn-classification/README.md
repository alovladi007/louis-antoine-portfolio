# Neural Networks Classification Suite

Advanced multi-class deep learning system for semiconductor defect detection and medical imaging classification.

## 🚀 Features

- **State-of-the-art Models**: CNN with attention (CBAM/SE), Vision Transformers, Hybrid architectures
- **Advanced Training**: Mixed precision, gradient accumulation, learning rate scheduling
- **Data Augmentation**: MixUp, CutMix, CutOut, traditional augmentations
- **Production Ready**: Model conversion, quantization, deployment optimization
- **Interactive Demos**: 8+ web-based interactive demonstrations
- **Comprehensive Documentation**: Theory guide, API reference, step-by-step tutorials

## 📁 Project Structure

```
nn-classification/
├── src/
│   ├── models/          # Neural network architectures
│   │   ├── attention.py # CBAM, SE blocks, spatial attention
│   │   ├── cnn.py       # CNN classifiers with attention
│   │   ├── transformer.py # Vision Transformers
│   │   └── ensemble.py  # Ensemble methods
│   ├── data/            # Data loading and preprocessing
│   │   ├── datasets.py  # Dataset classes
│   │   ├── augmentation.py # Augmentation pipelines
│   │   └── dataloader.py # DataLoader utilities
│   ├── training/        # Training pipelines
│   │   ├── trainer.py   # Main training class
│   │   ├── optimizer.py # Optimizers and schedulers
│   │   └── metrics.py   # Evaluation metrics
│   └── utils/           # Utilities
│       ├── visualization.py # Plotting and visualization
│       ├── export.py    # Model export utilities
│       └── deployment.py # Deployment optimization
├── notebooks/           # Jupyter notebooks
├── configs/            # Configuration files
├── tests/              # Unit tests
└── scripts/            # Training and evaluation scripts
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/nn-classification.git
cd nn-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Train a Model

```python
from src.models import CNNWithAttention
from src.data import create_dataloaders
from src.training import Trainer

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_path='data/',
    batch_size=32,
    num_workers=4
)

# Initialize model
model = CNNWithAttention(
    num_classes=8,
    attention_type='cbam'
)

# Configure training
config = {
    'learning_rate': 1e-3,
    'epochs': 50,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    'mixed_precision': True
}

# Create trainer and train
trainer = Trainer(model, train_loader, val_loader, config)
trainer.train()
```

### 2. Evaluate Model

```python
# Load trained model
model.load_state_dict(torch.load('checkpoints/best_model.pth'))

# Evaluate
results = trainer.evaluate(test_loader)
print(f"Test Accuracy: {results['accuracy']:.2%}")
print(f"F1 Score: {results['f1_score']:.3f}")
```

### 3. Visualize Attention

```python
from src.utils.visualization import plot_attention_maps

# Get attention maps
attention_maps = model.get_attention_maps(input_image)

# Visualize
plot_attention_maps(
    image=input_image,
    attention_maps=attention_maps,
    save_path='figures/attention.png'
)
```

## 📊 Performance

| Model | Accuracy | F1 Score | Inference Time |
|-------|----------|----------|----------------|
| CNN-Small | 92.3% | 0.921 | 2.3ms |
| CNN-CBAM | 94.8% | 0.945 | 4.1ms |
| ResNet-18 | 95.6% | 0.954 | 5.8ms |
| ViT-Tiny | 93.7% | 0.935 | 7.2ms |
| Ensemble | **96.2%** | **0.960** | 15.3ms |

## 🎯 Use Cases

### Semiconductor Defect Detection
- Wafer scratch detection
- Particle contamination
- Ring patterns
- Edge location defects

### Medical Imaging
- Tumor detection (small/large)
- Tissue inflammation
- Abnormality classification
- Normal/abnormal distinction

## 🔧 Advanced Features

### Attention Mechanisms
- **CBAM**: Convolutional Block Attention Module
- **SE Blocks**: Squeeze-and-Excitation blocks
- **Spatial Attention**: Focus on critical regions
- **Channel Attention**: Adaptive feature recalibration

### Data Augmentation
- **MixUp**: Linear interpolation of samples
- **CutMix**: Cut and paste augmentation
- **CutOut**: Random masking
- **AutoAugment**: Automated augmentation policies

### Deployment Optimization
- **Quantization**: INT8 conversion for 4x size reduction
- **Pruning**: Remove redundant connections
- **Knowledge Distillation**: Train smaller models
- **ONNX Export**: Cross-platform deployment

## 📚 Documentation

- [Theory Guide](docs/theory.md) - Mathematical foundations and architecture details
- [API Reference](docs/api.md) - Complete API documentation
- [Tutorials](docs/tutorials/) - Step-by-step guides
- [Examples](examples/) - Ready-to-run code examples

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{nn_classification_suite,
  title = {Neural Networks Classification Suite},
  author = {Neural Networks Team},
  year = {2024},
  url = {https://github.com/yourusername/nn-classification}
}
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Timm library for vision transformer implementations
- Albumentations for advanced augmentation techniques
- The open-source community for continuous support

## 📧 Contact

For questions and support, please open an issue on GitHub or contact us at:
- Email: nn-team@example.com
- Discord: [Join our server](https://discord.gg/example)

---

**Made with ❤️ by the Neural Networks Team**
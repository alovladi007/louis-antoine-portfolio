# Model Card - MediMetrics Demo Models

## Model Details

### DenseNet-121 Classifier (Demo)
- **Version**: 1.0.0-demo
- **Architecture**: DenseNet-121 with custom classification head
- **Input**: 224x224 grayscale or RGB images
- **Output**: Multi-label classification (8 classes)
- **Framework**: PyTorch 2.1.0
- **Size**: ~30MB (quantized demo version)

### UNet Segmentation (Demo)
- **Version**: 1.0.0-demo
- **Architecture**: Simplified U-Net
- **Input**: 256x256 grayscale images
- **Output**: Binary segmentation masks
- **Framework**: MONAI 1.3.0
- **Size**: ~15MB (lightweight version)

## Intended Use

### Primary Intended Use
- **Purpose**: Demonstration and testing of the MediMetrics platform
- **Users**: Developers, researchers, and evaluators
- **Context**: Non-clinical environments only

### Out-of-Scope Use Cases
- Clinical diagnosis or treatment decisions
- Production medical imaging workflows
- Patient care or screening programs
- Any use involving real patient data

## Training Data

### Synthetic Data Only
These demo models are trained on:
- Synthetic DICOM images generated programmatically
- No real patient data was used
- Patterns designed to simulate basic anatomical structures
- Limited diversity and complexity

### Data Characteristics
- **Modalities**: Simulated CR/DX
- **Body Parts**: Chest (simulated)
- **Sample Size**: 1,000 synthetic images
- **Labels**: Randomly assigned for demonstration

## Performance

### Demo Metrics (Synthetic Data)
- **Accuracy**: ~85% on synthetic test set
- **F1 Score**: ~0.82
- **AUC-ROC**: ~0.88

**Note**: These metrics are meaningless for clinical use as they're based on synthetic data.

### Limitations
- Not validated on real medical images
- No clinical performance metrics
- High false positive/negative rates expected on real data
- Biased toward synthetic pattern recognition

## Ethical Considerations

### Fairness
- Models not evaluated for demographic bias
- Synthetic data doesn't represent population diversity
- No fairness metrics computed

### Privacy
- Trained only on synthetic data
- No patient privacy concerns for demo models
- Real deployment would require privacy safeguards

### Transparency
- Full model architecture documented
- Training process reproducible
- Limitations clearly stated

## Caveats and Recommendations

### Critical Warnings
1. **DO NOT use for clinical decisions**
2. **DO NOT interpret outputs as medical findings**
3. **DO NOT deploy without proper validation**
4. **DO NOT use with real patient data without approval**

### Technical Limitations
- Limited to specific image sizes
- Requires preprocessing (windowing, normalization)
- May fail on out-of-distribution inputs
- No robustness guarantees

### Recommended Use
1. Platform functionality testing
2. API integration development
3. UI/UX evaluation
4. Workflow demonstration
5. Training and education (with disclaimers)

## Model Updates

### Version History
- v1.0.0-demo (2024): Initial demo release

### Update Policy
- Demo models are static and not updated
- Clinical models would require regulatory approval for updates

## Evaluation

### Test Datasets
- Synthetic test set (200 images)
- No external validation
- No clinical validation

### Metrics Tracked
- Classification: Accuracy, Precision, Recall, F1, AUC-ROC
- Segmentation: Dice coefficient, IoU
- Inference: Latency, throughput, memory usage

## Explainability

### Grad-CAM Integration
- Provides attention heatmaps for classifications
- Highlights regions contributing to predictions
- For demonstration purposes only

### Interpretation Warnings
- Attention maps may be misleading
- Don't indicate clinical significance
- Should not guide medical decisions

## Deployment

### Infrastructure Requirements
- CPU: 4+ cores recommended
- RAM: 8GB minimum
- GPU: Optional (CUDA 11.8+ for acceleration)
- Storage: 1GB for models

### Inference Performance
- CPU: ~200ms per image
- GPU: ~50ms per image
- Batch processing supported

## References

### Architecture Papers
- DenseNet: Huang et al., "Densely Connected Convolutional Networks" (2017)
- U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)

### Frameworks
- PyTorch: https://pytorch.org
- MONAI: https://monai.io

## Contact

- **Model Issues**: models@medimetrics.example
- **Clinical Inquiries**: Not applicable (demo only)
- **Research Collaboration**: research@medimetrics.example

## License

Models are provided under Apache 2.0 License for demonstration purposes only.

## Disclaimer

These models are synthetic demonstrations and have no clinical validity. They must not be used for any medical purpose. See [DISCLAIMER.md](DISCLAIMER.md) for full details.

---

**Model Card Version**: 1.0.0
**Last Updated**: 2024
**Next Review**: Not applicable (demo only)
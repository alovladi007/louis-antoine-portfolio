#!/usr/bin/env python3
"""
Interactive Demo for Defect Detection & Yield Prediction ML System
Run this script to see all features in action with visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_generator import DefectDataGenerator, SyntheticWaferGenerator, DefectType
from src.cnn_models import create_model
import torch
import torch.nn.functional as F

# Set style for better visualizations
plt.style.use('dark_background')
sns.set_palette("husl")

def demo_wafer_generation():
    """Demonstrate wafer map generation with different defect types"""
    print("\n" + "="*60)
    print("WAFER DEFECT GENERATION DEMO")
    print("="*60)
    
    generator = DefectDataGenerator(wafer_size=300, die_size=(10, 10))
    
    # Generate different defect patterns
    defect_types = [
        DefectType.PARTICLE,
        DefectType.SCRATCH,
        DefectType.CLUSTER,
        DefectType.RING,
        DefectType.EDGE,
        DefectType.PATTERN
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, defect_type in enumerate(defect_types):
        wafer_map = generator.generate_wafer_map(
            defect_type=defect_type,
            defect_density=0.05,
            seed=42 + idx
        )
        
        # Create visualization
        ax = axes[idx]
        im = ax.imshow(wafer_map.data, cmap='RdYlGn_r', interpolation='nearest')
        ax.set_title(f'{defect_type.value.capitalize()} Defects\nYield: {wafer_map.yield_rate:.1%}')
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Synthetic Wafer Defect Maps - Different Pattern Types', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('defect-detection-ml/outputs')
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / 'wafer_defect_patterns.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Generated {len(defect_types)} different defect patterns")
    print(f"✓ Visualization saved to {output_dir / 'wafer_defect_patterns.png'}")
    
    return wafer_map

def demo_synthetic_wafer_image():
    """Generate realistic wafer images"""
    print("\n" + "="*60)
    print("SYNTHETIC WAFER IMAGE GENERATION")
    print("="*60)
    
    # Generate wafer map
    generator = DefectDataGenerator(wafer_size=300, die_size=(10, 10))
    wafer_map = generator.generate_wafer_map(
        defect_type=DefectType.CLUSTER,
        defect_density=0.03
    )
    
    # Generate synthetic wafer image
    wafer_gen = SyntheticWaferGenerator(wafer_diameter=300, resolution=1024)
    wafer_image = wafer_gen.generate_wafer_image(wafer_map, add_texture=True)
    wafer_image_with_artifacts = wafer_gen.add_inspection_artifacts(wafer_image)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(wafer_map.data, cmap='RdYlGn_r')
    axes[0].set_title('Wafer Map\n(Die-level defects)')
    axes[0].axis('off')
    
    axes[1].imshow(wafer_image, cmap='gray')
    axes[1].set_title('Synthetic Wafer Image\n(Clean)')
    axes[1].axis('off')
    
    axes[2].imshow(wafer_image_with_artifacts, cmap='gray')
    axes[2].set_title('With Inspection Artifacts\n(Realistic)')
    axes[2].axis('off')
    
    plt.suptitle('Synthetic Wafer Image Generation Pipeline', fontsize=16)
    plt.tight_layout()
    
    output_dir = Path('defect-detection-ml/outputs')
    plt.savefig(output_dir / 'synthetic_wafer_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Generated wafer image with resolution: {wafer_gen.resolution}x{wafer_gen.resolution}")
    print(f"✓ Added realistic inspection artifacts")
    
    return wafer_image_with_artifacts

def demo_defect_statistics():
    """Analyze defect statistics across multiple wafers"""
    print("\n" + "="*60)
    print("DEFECT STATISTICS ANALYSIS")
    print("="*60)
    
    generator = DefectDataGenerator(wafer_size=300, die_size=(10, 10))
    
    # Generate multiple wafers
    n_wafers = 100
    yields = []
    defect_counts = []
    defect_types_used = []
    
    for i in range(n_wafers):
        defect_type = np.random.choice(list(DefectType)[:-2])  # Exclude RANDOM and NONE
        wafer_map = generator.generate_wafer_map(
            defect_type=defect_type,
            defect_density=np.random.uniform(0.01, 0.1),
            seed=i
        )
        yields.append(wafer_map.yield_rate)
        defect_counts.append(wafer_map.defect_count)
        defect_types_used.append(defect_type.value)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Yield distribution
    axes[0, 0].hist(yields, bins=20, edgecolor='white', alpha=0.7, color='green')
    axes[0, 0].axvline(np.mean(yields), color='red', linestyle='--', label=f'Mean: {np.mean(yields):.1%}')
    axes[0, 0].set_xlabel('Yield Rate')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Yield Distribution Across Wafers')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Defect count distribution
    axes[0, 1].hist(defect_counts, bins=20, edgecolor='white', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Defect Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Defect Count Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Yield vs Defect Count
    axes[1, 0].scatter(defect_counts, yields, alpha=0.6, s=30)
    axes[1, 0].set_xlabel('Defect Count')
    axes[1, 0].set_ylabel('Yield Rate')
    axes[1, 0].set_title('Yield vs Defect Count Correlation')
    
    # Add trend line
    z = np.polyfit(defect_counts, yields, 1)
    p = np.poly1d(z)
    axes[1, 0].plot(sorted(defect_counts), p(sorted(defect_counts)), "r--", alpha=0.8, label='Trend')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Defect type distribution
    from collections import Counter
    defect_type_counts = Counter(defect_types_used)
    axes[1, 1].bar(defect_type_counts.keys(), defect_type_counts.values(), color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Defect Type')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Defect Type Distribution')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Statistical Analysis of {n_wafers} Synthetic Wafers', fontsize=16)
    plt.tight_layout()
    
    output_dir = Path('defect-detection-ml/outputs')
    plt.savefig(output_dir / 'defect_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Analyzed {n_wafers} wafers")
    print(f"✓ Average yield: {np.mean(yields):.1%}")
    print(f"✓ Average defect count: {np.mean(defect_counts):.1f}")
    print(f"✓ Yield-defect correlation: {np.corrcoef(defect_counts, yields)[0,1]:.3f}")

def demo_cnn_architectures():
    """Demonstrate different CNN architectures"""
    print("\n" + "="*60)
    print("CNN ARCHITECTURE COMPARISON")
    print("="*60)
    
    # Test input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 224, 224)
    
    models_info = []
    
    model_configs = [
        ('defect_cnn', {}),
        ('resnet', {'resnet_version': 'resnet50', 'pretrained': False}),
        ('attention_cnn', {}),
        ('multiscale_cnn', {})
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (model_name, kwargs) in enumerate(model_configs):
        print(f"\nTesting {model_name}...")
        model = create_model(model_name, num_classes=5, **kwargs)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Forward pass
        with torch.no_grad():
            if model_name == 'attention_cnn':
                output, attention = model(input_tensor)
                # Visualize attention map
                att_map = attention[0, 0].numpy()
                axes[idx].imshow(att_map, cmap='hot')
                axes[idx].set_title(f'{model_name}\nAttention Map')
            else:
                output = model(input_tensor)
                # Visualize output distribution
                probs = F.softmax(output, dim=1)[0].numpy()
                axes[idx].bar(range(5), probs, color='cyan', alpha=0.7)
                axes[idx].set_xlabel('Class')
                axes[idx].set_ylabel('Probability')
                axes[idx].set_title(f'{model_name}\nOutput Distribution')
                axes[idx].set_ylim([0, 1])
        
        models_info.append({
            'name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'output_shape': output.shape
        })
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Output shape: {output.shape}")
    
    plt.suptitle('CNN Architecture Comparison', fontsize=16)
    plt.tight_layout()
    
    output_dir = Path('defect-detection-ml/outputs')
    plt.savefig(output_dir / 'cnn_architectures.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create comparison table
    print("\n" + "-"*60)
    print("Model Comparison Summary:")
    print("-"*60)
    print(f"{'Model':<20} {'Total Params':<15} {'Trainable':<15}")
    print("-"*60)
    for info in models_info:
        print(f"{info['name']:<20} {info['total_params']:>14,} {info['trainable_params']:>14,}")
    
    return models_info

def demo_uncertainty_quantification():
    """Demonstrate Bayesian uncertainty estimation"""
    print("\n" + "="*60)
    print("UNCERTAINTY QUANTIFICATION DEMO")
    print("="*60)
    
    from src.bayesian_models import MCDropoutClassifier
    
    # Create model
    mc_model = MCDropoutClassifier(num_classes=5, dropout_rate=0.5)
    
    # Generate test data
    test_input = torch.randn(10, 1, 224, 224)
    
    # Get predictions with uncertainty
    mean_pred, uncertainty = mc_model.predict_with_uncertainty(test_input, n_samples=30)
    
    # Visualize uncertainty
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Prediction confidence
    pred_probs = mean_pred.max(dim=1)[0].numpy()
    axes[0].bar(range(len(pred_probs)), pred_probs, color='green', alpha=0.7)
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Confidence')
    axes[0].set_title('Prediction Confidence per Sample')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Uncertainty distribution
    axes[1].hist(uncertainty.numpy(), bins=20, color='red', alpha=0.7, edgecolor='white')
    axes[1].set_xlabel('Uncertainty (Mutual Information)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Uncertainty Distribution')
    axes[1].axvline(uncertainty.mean().item(), color='yellow', linestyle='--', 
                   label=f'Mean: {uncertainty.mean().item():.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Bayesian Uncertainty Quantification with MC Dropout', fontsize=16)
    plt.tight_layout()
    
    output_dir = Path('defect-detection-ml/outputs')
    plt.savefig(output_dir / 'uncertainty_quantification.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Performed {30} Monte Carlo forward passes")
    print(f"✓ Mean uncertainty: {uncertainty.mean().item():.3f}")
    print(f"✓ Samples with high uncertainty (>0.5): {(uncertainty > 0.5).sum().item()}")

def demo_active_learning():
    """Demonstrate active learning sample selection"""
    print("\n" + "="*60)
    print("ACTIVE LEARNING DEMONSTRATION")
    print("="*60)
    
    from src.bayesian_models import ActiveLearningSelector, MCDropoutClassifier
    
    # Create model and selector
    model = MCDropoutClassifier(num_classes=5)
    selector = ActiveLearningSelector(strategy='uncertainty')
    
    # Create unlabeled pool
    unlabeled_pool = torch.randn(100, 1, 224, 224)
    
    # Select samples
    n_select = 10
    selected_indices = selector.select_samples(model, unlabeled_pool, n_samples=n_select)
    
    # Visualize selection
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(selected_indices[:10]):
        # Show selected sample (simplified - just show a pattern)
        sample_viz = unlabeled_pool[idx, 0, :50, :50].numpy()
        axes[i].imshow(sample_viz, cmap='viridis')
        axes[i].set_title(f'Sample {idx}')
        axes[i].axis('off')
    
    plt.suptitle(f'Top {n_select} Samples Selected by Active Learning (Uncertainty-based)', fontsize=16)
    plt.tight_layout()
    
    output_dir = Path('defect-detection-ml/outputs')
    plt.savefig(output_dir / 'active_learning_selection.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Selected {n_select} most informative samples from pool of {len(unlabeled_pool)}")
    print(f"✓ Selected indices: {selected_indices}")

def generate_sample_dataset():
    """Generate a sample dataset for training"""
    print("\n" + "="*60)
    print("GENERATING SAMPLE DATASET")
    print("="*60)
    
    generator = DefectDataGenerator()
    
    # Generate dataset
    dataset = generator.generate_dataset(
        n_samples=500,
        defect_classes=["particle", "scratch", "void", "bridge", "none"],
        train_split=0.8
    )
    
    # Save dataset
    output_dir = Path('defect-detection-ml/data')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    np.save(output_dir / 'train_images.npy', dataset['train']['images'])
    np.save(output_dir / 'train_labels.npy', dataset['train']['labels'])
    np.save(output_dir / 'test_images.npy', dataset['test']['images'])
    np.save(output_dir / 'test_labels.npy', dataset['test']['labels'])
    
    print(f"✓ Generated {len(dataset['train']['images'])} training samples")
    print(f"✓ Generated {len(dataset['test']['images'])} test samples")
    print(f"✓ Classes: {dataset['classes']}")
    print(f"✓ Dataset saved to {output_dir}")
    
    # Visualize sample images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(10):
        img = dataset['train']['images'][i]
        label = dataset['train']['labels'][i]
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Class: {dataset['classes'][label]}")
        axes[i].axis('off')
    
    plt.suptitle('Sample Training Images', fontsize=16)
    plt.tight_layout()
    
    output_dir = Path('defect-detection-ml/outputs')
    plt.savefig(output_dir / 'sample_dataset.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return dataset

def main():
    """Run all demonstrations"""
    print("\n" + "="*80)
    print(" DEFECT DETECTION & YIELD PREDICTION ML - COMPREHENSIVE DEMO")
    print("="*80)
    print("\nThis demo will showcase all implemented features with visualizations.")
    print("Output images will be saved to defect-detection-ml/outputs/")
    
    # Create output directory
    output_dir = Path('defect-detection-ml/outputs')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Run demos
        print("\n[1/7] Wafer Defect Generation...")
        demo_wafer_generation()
        
        print("\n[2/7] Synthetic Wafer Images...")
        demo_synthetic_wafer_image()
        
        print("\n[3/7] Statistical Analysis...")
        demo_defect_statistics()
        
        print("\n[4/7] CNN Architectures...")
        demo_cnn_architectures()
        
        print("\n[5/7] Uncertainty Quantification...")
        demo_uncertainty_quantification()
        
        print("\n[6/7] Active Learning...")
        demo_active_learning()
        
        print("\n[7/7] Sample Dataset Generation...")
        generate_sample_dataset()
        
        print("\n" + "="*80)
        print(" DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n✓ All visualizations saved to: {output_dir.absolute()}")
        print("✓ Sample dataset saved to: defect-detection-ml/data/")
        print("\nNext steps:")
        print("  1. Run 'streamlit run streamlit_app.py' for interactive web interface")
        print("  2. Check notebooks/ folder for detailed tutorials")
        print("  3. Use train.py to train models on the generated dataset")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
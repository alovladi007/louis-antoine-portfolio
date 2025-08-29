#!/usr/bin/env python3
"""
Training Pipeline for Defect Detection Models
Complete training script with logging, checkpointing, and evaluation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import argparse

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from src.data_generator import DefectDataGenerator
from src.cnn_models import create_model
from src.bayesian_models import MCDropoutClassifier


class DefectDataset(Dataset):
    """PyTorch Dataset for defect images"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensor and add channel dimension
        image = torch.FloatTensor(image).unsqueeze(0) / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class Trainer:
    """Training pipeline for defect detection models"""
    
    def __init__(self, model, device='cpu', output_dir='models'):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0
        self.best_epoch = 0
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, 
              save_best=True, early_stopping=10):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            save_best: Save best model
            early_stopping: Early stopping patience
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        print(f"\n{'='*60}")
        print(f"Training {self.model.__class__.__name__}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Learning Rate: {lr}")
        print(f"Batch Size: {train_loader.batch_size}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer
            )
            
            # Validation phase
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                if save_best:
                    self._save_checkpoint(epoch, val_acc)
                    print(f"  → Saved best model (Val Acc: {val_acc:.2%})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2%} (Epoch {self.best_epoch})")
        print(f"{'='*60}\n")
        
        return self.history
    
    def _train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            if isinstance(outputs, tuple):  # Handle attention models
                outputs = outputs[0]
            
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, tuple):  # Handle attention models
                    outputs = outputs[0]
                
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _save_checkpoint(self, epoch, val_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"model_epoch{epoch+1}_acc{val_acc:.3f}_{timestamp}.pth"
        torch.save(checkpoint, filename)
        
        # Also save as 'best_model.pth'
        torch.save(checkpoint, self.output_dir / "best_model.pth")
    
    def plot_history(self, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(self.history['val_loss'], label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Acc', color='blue')
        ax2.plot(self.history['val_acc'], label='Val Acc', color='red')
        ax2.axhline(y=self.best_val_acc, color='green', linestyle='--', 
                   label=f'Best: {self.best_val_acc:.2%}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training History - Best Epoch: {self.best_epoch}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def evaluate(self, test_loader, with_uncertainty=False):
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            with_uncertainty: Calculate uncertainty (for MC Dropout models)
        """
        self.model.eval()
        correct = 0
        total = 0
        class_correct = {}
        class_total = {}
        
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        
        print("\nEvaluating on test set...")
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if with_uncertainty and isinstance(self.model, MCDropoutClassifier):
                    # MC Dropout predictions
                    mean_pred, uncertainty = self.model.predict_with_uncertainty(
                        images, n_samples=30
                    )
                    _, predicted = mean_pred.max(1)
                    all_uncertainties.extend(uncertainty.cpu().numpy())
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Per-class accuracy
                for label, pred in zip(labels, predicted):
                    label_item = label.item()
                    if label_item not in class_correct:
                        class_correct[label_item] = 0
                        class_total[label_item] = 0
                    
                    class_total[label_item] += 1
                    if label_item == pred.item():
                        class_correct[label_item] += 1
        
        # Overall accuracy
        overall_acc = correct / total
        
        print(f"\n{'='*50}")
        print(f"Test Results")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {overall_acc:.2%} ({correct}/{total})")
        
        # Per-class accuracy
        print("\nPer-Class Accuracy:")
        classes = ['Particle', 'Scratch', 'Void', 'Bridge', 'None']
        for cls_idx in sorted(class_correct.keys()):
            cls_acc = class_correct[cls_idx] / class_total[cls_idx]
            cls_name = classes[cls_idx] if cls_idx < len(classes) else f"Class {cls_idx}"
            print(f"  {cls_name:12s}: {cls_acc:.2%} ({class_correct[cls_idx]}/{class_total[cls_idx]})")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes[:cm.shape[1]],
               yticklabels=classes[:cm.shape[0]],
               xlabel='Predicted',
               ylabel='True',
               title='Confusion Matrix')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, 
                                   target_names=classes[:len(set(all_labels))]))
        
        if with_uncertainty and all_uncertainties:
            print(f"\nUncertainty Statistics:")
            print(f"  Mean: {np.mean(all_uncertainties):.4f}")
            print(f"  Std:  {np.std(all_uncertainties):.4f}")
            print(f"  Min:  {np.min(all_uncertainties):.4f}")
            print(f"  Max:  {np.max(all_uncertainties):.4f}")
        
        return overall_acc


def main():
    parser = argparse.ArgumentParser(description='Train Defect Detection Model')
    parser.add_argument('--model', type=str, default='defect_cnn',
                       choices=['defect_cnn', 'resnet', 'efficientnet', 
                               'attention_cnn', 'multiscale_cnn', 'mc_dropout'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--n_samples', type=int, default=1000, 
                       help='Number of training samples to generate')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--load_data', action='store_true',
                       help='Load existing data from disk')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(" DEFECT DETECTION MODEL TRAINING")
    print("="*60)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate or load data
    if args.load_data and Path('defect-detection-ml/data/train_images.npy').exists():
        print("\nLoading existing dataset...")
        data_dir = Path('defect-detection-ml/data')
        train_images = np.load(data_dir / 'train_images.npy')
        train_labels = np.load(data_dir / 'train_labels.npy')
        test_images = np.load(data_dir / 'test_images.npy')
        test_labels = np.load(data_dir / 'test_labels.npy')
        print(f"Loaded {len(train_images)} training and {len(test_images)} test samples")
    else:
        print(f"\nGenerating {args.n_samples} synthetic samples...")
        generator = DefectDataGenerator()
        dataset = generator.generate_dataset(
            n_samples=args.n_samples,
            defect_classes=["particle", "scratch", "void", "bridge", "none"],
            train_split=0.8
        )
        
        train_images = dataset['train']['images']
        train_labels = dataset['train']['labels']
        test_images = dataset['test']['images']
        test_labels = dataset['test']['labels']
        
        # Save generated data
        data_dir = Path('defect-detection-ml/data')
        data_dir.mkdir(exist_ok=True, parents=True)
        np.save(data_dir / 'train_images.npy', train_images)
        np.save(data_dir / 'train_labels.npy', train_labels)
        np.save(data_dir / 'test_images.npy', test_images)
        np.save(data_dir / 'test_labels.npy', test_labels)
        print(f"Saved dataset to {data_dir}")
    
    # Create data loaders
    train_dataset = DefectDataset(train_images, train_labels)
    test_dataset = DefectDataset(test_images, test_labels)
    
    # Split training data for validation
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0)
    
    print(f"\nDataset splits:")
    print(f"  Training:   {len(train_subset)} samples")
    print(f"  Validation: {len(val_subset)} samples")
    print(f"  Test:       {len(test_dataset)} samples")
    
    # Create model
    print(f"\nCreating model: {args.model}")
    
    if args.model == 'mc_dropout':
        model = MCDropoutClassifier(num_classes=5, dropout_rate=0.5)
    else:
        kwargs = {}
        if args.model == 'resnet':
            kwargs = {'resnet_version': 'resnet50', 'pretrained': False}
        
        model = create_model(args.model, num_classes=5, **kwargs)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create trainer
    output_dir = Path('defect-detection-ml/models')
    trainer = Trainer(model, device=device, output_dir=output_dir)
    
    # Train model
    history = trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        save_best=True,
        early_stopping=10
    )
    
    # Plot training history
    trainer.plot_history(save_path=output_dir / 'training_history.png')
    
    # Evaluate on test set
    test_acc = trainer.evaluate(
        test_loader, 
        with_uncertainty=(args.model == 'mc_dropout')
    )
    
    # Save training summary
    summary = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'best_val_acc': float(trainer.best_val_acc),
        'best_epoch': trainer.best_epoch,
        'test_acc': float(test_acc),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'history': trainer.history
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Training complete! Results saved to {output_dir}")
    print(f"   - Best model: best_model.pth")
    print(f"   - Training history: training_history.png")
    print(f"   - Confusion matrix: confusion_matrix.png")
    print(f"   - Summary: training_summary.json")


if __name__ == "__main__":
    main()
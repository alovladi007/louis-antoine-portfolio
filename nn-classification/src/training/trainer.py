"""
Training pipeline for neural network models
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, Tuple, List, Callable
import numpy as np
from tqdm import tqdm
import wandb
from ..utils.metrics import calculate_metrics, confusion_matrix
from ..utils.visualization import plot_training_curves


class Trainer:
    """Advanced trainer with mixed precision, gradient accumulation, and monitoring"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Default configuration
        self.config = {
            'epochs': 100,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'warmup_epochs': 5,
            'gradient_clip': 1.0,
            'accumulation_steps': 1,
            'mixed_precision': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'save_dir': 'checkpoints',
            'log_interval': 10,
            'patience': 10,
            'wandb_project': None,
            'label_smoothing': 0.1,
            'mixup_alpha': 0.2,
            'cutmix_alpha': 1.0,
            'augmentation_prob': 0.5
        }
        
        if config:
            self.config.update(config)
        
        self.device = torch.device(self.config['device'])
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config['label_smoothing'])
        
        # Mixed precision training
        self.scaler = GradScaler() if self.config['mixed_precision'] else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Create save directory
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # Initialize wandb if configured
        if self.config['wandb_project']:
            wandb.init(project=self.config['wandb_project'], config=self.config)
            wandb.watch(self.model)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        params = self.model.parameters()
        
        if self.config['optimizer'] == 'adam':
            return optim.Adam(params, lr=self.config['learning_rate'],
                            weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamw':
            return optim.AdamW(params, lr=self.config['learning_rate'],
                             weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'sgd':
            return optim.SGD(params, lr=self.config['learning_rate'],
                           momentum=0.9, weight_decay=self.config['weight_decay'])
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        total_steps = len(self.train_loader) * self.config['epochs']
        warmup_steps = len(self.train_loader) * self.config['warmup_epochs']
        
        if self.config['scheduler'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )
        elif self.config['scheduler'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif self.config['scheduler'] == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5, factor=0.5
            )
        elif self.config['scheduler'] == 'onecycle':
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.config['learning_rate'],
                total_steps=total_steps
            )
        else:
            return optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0)
    
    def mixup_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        y_a, y_b = y, y[index]
        
        return x, y_a, y_b, lam
    
    def _rand_bbox(self, size: Tuple, lam: float) -> Tuple[int, int, int, int]:
        """Generate random bounding box for CutMix"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["epochs"]}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Apply augmentation
            r = np.random.rand(1)
            if r < self.config['augmentation_prob']:
                if r < self.config['augmentation_prob'] / 2:
                    # Apply mixup
                    inputs, targets_a, targets_b, lam = self.mixup_data(
                        inputs, targets, self.config['mixup_alpha']
                    )
                    mixed = True
                else:
                    # Apply cutmix
                    inputs, targets_a, targets_b, lam = self.cutmix_data(
                        inputs, targets, self.config['cutmix_alpha']
                    )
                    mixed = True
            else:
                mixed = False
            
            # Mixed precision training
            if self.config['mixed_precision'] and self.scaler:
                with autocast():
                    outputs = self.model(inputs)
                    if mixed:
                        loss = lam * self.criterion(outputs, targets_a) + \
                               (1 - lam) * self.criterion(outputs, targets_b)
                    else:
                        loss = self.criterion(outputs, targets)
                
                # Gradient accumulation
                loss = loss / self.config['accumulation_steps']
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config['gradient_clip']
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(inputs)
                if mixed:
                    loss = lam * self.criterion(outputs, targets_a) + \
                           (1 - lam) * self.criterion(outputs, targets_b)
                else:
                    loss = self.criterion(outputs, targets)
                
                loss = loss / self.config['accumulation_steps']
                loss.backward()
                
                if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config['gradient_clip']
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            if not mixed:
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            running_loss += loss.item() * self.config['accumulation_steps']
            
            # Update progress bar
            if total > 0:
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        # Update scheduler
        if self.config['scheduler'] != 'plateau':
            self.scheduler.step()
        
        return {
            'loss': running_loss / len(self.train_loader),
            'accuracy': 100. * correct / total if total > 0 else 0
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if self.config['mixed_precision']:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(all_targets, all_preds)
        
        return {
            'loss': running_loss / len(self.val_loader),
            'accuracy': 100. * correct / total,
            **metrics
        }
    
    def train(self) -> Dict:
        """Complete training loop"""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Update scheduler if using plateau
            if self.config['scheduler'] == 'plateau':
                self.scheduler.step(val_metrics['accuracy'])
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log to wandb
            if self.config['wandb_project']:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['accuracy'],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    **val_metrics
                })
            
            # Print progress
            print(f"Epoch {epoch}/{self.config['epochs']}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                print(f"  ✓ New best model saved (Acc: {self.best_val_acc:.2f}%)")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch, val_metrics)
        
        # Final evaluation on test set
        if self.test_loader:
            test_metrics = self.test()
            self.history['test_metrics'] = test_metrics
            print(f"\nTest Results:")
            print(f"  Test Acc: {test_metrics['accuracy']:.2f}%")
            print(f"  Test F1: {test_metrics['f1_weighted']:.4f}")
        
        return self.history
    
    def test(self) -> Dict[str, float]:
        """Test the model"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc='Testing'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if self.config['mixed_precision']:
                    with autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate comprehensive metrics
        metrics = calculate_metrics(all_targets, all_preds)
        metrics['accuracy'] = 100. * correct / total
        
        return metrics
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'history': self.history
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, os.path.join(self.config['save_dir'], filename))
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(os.path.join(self.config['save_dir'], filename))
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        return checkpoint['epoch'], checkpoint.get('metrics', {})
#!/usr/bin/env python3
"""
PyTorch Lightning Callbacks for CsPbBr₃ Digital Twin Training
Advanced callbacks for monitoring, early stopping, and model optimization
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.base import Callback
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
import wandb
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class PhysicsConsistencyCallback(Callback):
    """Monitor physics consistency during training"""
    
    def __init__(self, 
                 check_interval: int = 10,
                 tolerance: float = 0.1,
                 log_plots: bool = True):
        """
        Initialize physics consistency callback
        
        Args:
            check_interval: Check every N epochs
            tolerance: Tolerance for physics violations
            log_plots: Whether to log physics plots
        """
        self.check_interval = check_interval
        self.tolerance = tolerance
        self.log_plots = log_plots
        self.physics_violations = []
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Check physics consistency at end of validation epoch"""
        if trainer.current_epoch % self.check_interval != 0:
            return
        
        # Get validation predictions
        val_dataloader = trainer.val_dataloaders[0]
        pl_module.eval()
        
        physics_violations = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                outputs = pl_module(batch['features'])
                
                # Check physics constraints
                violations = self._check_physics_constraints(outputs, batch)
                physics_violations += violations
                total_samples += batch['features'].shape[0]
        
        violation_rate = physics_violations / total_samples if total_samples > 0 else 0
        self.physics_violations.append(violation_rate)
        
        # Log metrics
        pl_module.log('physics/violation_rate', violation_rate, prog_bar=True)
        pl_module.log('physics/consistency_score', 1.0 - violation_rate)
        
        # Warning if too many violations
        if violation_rate > self.tolerance:
            logger.warning(f"High physics violation rate: {violation_rate:.3f}")
        
        logger.info(f"Epoch {trainer.current_epoch}: Physics violation rate: {violation_rate:.3f}")
    
    def _check_physics_constraints(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> int:
        """Check various physics constraints"""
        violations = 0
        batch_size = outputs['phase']['probabilities'].shape[0]
        
        # 1. Phase probabilities should sum to 1
        phase_sums = outputs['phase']['probabilities'].sum(dim=1)
        prob_violations = torch.abs(phase_sums - 1.0) > 0.01
        violations += prob_violations.sum().item()
        
        # 2. Property values should be physically reasonable
        for prop_name, prop_output in outputs['properties'].items():
            prop_mean = prop_output['mean']
            
            if prop_name == 'bandgap':
                # Bandgap should be 0.5-5.0 eV
                violations += ((prop_mean < 0.5) | (prop_mean > 5.0)).sum().item()
            elif prop_name == 'plqy':
                # PLQY should be 0-1
                violations += ((prop_mean < 0) | (prop_mean > 1)).sum().item()
            elif prop_name == 'particle_size':
                # Size should be positive and reasonable (1-1000 nm)
                violations += ((prop_mean < 1) | (prop_mean > 1000)).sum().item()
        
        # 3. Uncertainty should be positive
        for prop_name, prop_output in outputs['properties'].items():
            prop_std = prop_output['std']
            violations += (prop_std < 0).sum().item()
        
        return violations


class UncertaintyCalibrationCallback(Callback):
    """Monitor and improve uncertainty calibration"""
    
    def __init__(self, 
                 check_interval: int = 5,
                 num_bins: int = 10,
                 save_plots: bool = True):
        """
        Initialize uncertainty calibration callback
        
        Args:
            check_interval: Check every N epochs
            num_bins: Number of bins for calibration plot
            save_plots: Whether to save calibration plots
        """
        self.check_interval = check_interval
        self.num_bins = num_bins
        self.save_plots = save_plots
        self.calibration_history = []
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Check uncertainty calibration"""
        if trainer.current_epoch % self.check_interval != 0:
            return
        
        # Collect predictions and uncertainties
        predictions = []
        uncertainties = []
        targets = []
        
        val_dataloader = trainer.val_dataloaders[0]
        pl_module.eval()
        
        with torch.no_grad():
            for batch in val_dataloader:
                outputs = pl_module(batch['features'])
                
                # Phase predictions
                phase_probs = outputs['phase']['probabilities']
                phase_uncertainty = outputs['phase']['uncertainty']
                phase_targets = batch['phase_labels']
                
                predictions.append(phase_probs.cpu())
                uncertainties.append(phase_uncertainty.cpu())
                targets.append(phase_targets.cpu())
        
        if predictions:
            predictions = torch.cat(predictions, dim=0)
            uncertainties = torch.cat(uncertainties, dim=0)
            targets = torch.cat(targets, dim=0)
            
            # Calculate calibration metrics
            calibration_error = self._calculate_calibration_error(predictions, uncertainties, targets)
            self.calibration_history.append(calibration_error)
            
            # Log metrics
            pl_module.log('uncertainty/calibration_error', calibration_error)
            
            # Save calibration plot
            if self.save_plots:
                self._save_calibration_plot(predictions, uncertainties, targets, trainer.current_epoch)
    
    def _calculate_calibration_error(self, predictions: torch.Tensor, 
                                   uncertainties: torch.Tensor, 
                                   targets: torch.Tensor) -> float:
        """Calculate expected calibration error"""
        # Get predicted class and confidence
        pred_classes = predictions.argmax(dim=1)
        confidences = predictions.max(dim=1)[0]
        
        # Check if predictions are correct
        correct = (pred_classes == targets).float()
        
        # Bin by confidence
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                # Average confidence and accuracy in this bin
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Add to ECE
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()
    
    def _save_calibration_plot(self, predictions: torch.Tensor, 
                             uncertainties: torch.Tensor, 
                             targets: torch.Tensor, 
                             epoch: int):
        """Save calibration plot"""
        plt.figure(figsize=(8, 6))
        
        pred_classes = predictions.argmax(dim=1)
        confidences = predictions.max(dim=1)[0]
        correct = (pred_classes == targets).float()
        
        # Create bins and calculate calibration
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(correct[in_bin].mean().item())
                bin_confidences.append(confidences[in_bin].mean().item())
        
        # Plot calibration curve
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        plt.plot(bin_confidences, bin_accuracies, 'bo-', label='Model calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Calibration Plot - Epoch {epoch}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        save_path = Path(f'calibration_plots/epoch_{epoch}.png')
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class CrossValidationCallback(Callback):
    """Handle cross-validation training and logging"""
    
    def __init__(self, 
                 fold_id: int,
                 total_folds: int,
                 save_fold_results: bool = True):
        """
        Initialize cross-validation callback
        
        Args:
            fold_id: Current fold ID (0-indexed)
            total_folds: Total number of folds
            save_fold_results: Whether to save individual fold results
        """
        self.fold_id = fold_id
        self.total_folds = total_folds
        self.save_fold_results = save_fold_results
        self.fold_metrics = {}
        
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log fold information at start of training"""
        logger.info(f"Starting training for fold {self.fold_id + 1}/{self.total_folds}")
        pl_module.log('fold/id', float(self.fold_id))
        pl_module.log('fold/total', float(self.total_folds))
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Store validation metrics for this fold"""
        current_metrics = trainer.logged_metrics.copy()
        
        # Store best metrics
        if 'val/total_loss' in current_metrics:
            val_loss = current_metrics['val/total_loss'].item()
            if 'best_val_loss' not in self.fold_metrics or val_loss < self.fold_metrics['best_val_loss']:
                self.fold_metrics['best_val_loss'] = val_loss
                self.fold_metrics['best_epoch'] = trainer.current_epoch
        
        if 'val/phase_acc' in current_metrics:
            val_acc = current_metrics['val/phase_acc'].item()
            if 'best_val_acc' not in self.fold_metrics or val_acc > self.fold_metrics['best_val_acc']:
                self.fold_metrics['best_val_acc'] = val_acc
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Save fold results at end of training"""
        logger.info(f"Completed training for fold {self.fold_id + 1}/{self.total_folds}")
        
        # Add final metrics
        self.fold_metrics['fold_id'] = self.fold_id
        self.fold_metrics['total_epochs'] = trainer.current_epoch
        self.fold_metrics['training_time'] = trainer.fit_time if hasattr(trainer, 'fit_time') else 0
        
        # Save fold results
        if self.save_fold_results:
            save_path = Path(f'fold_results/fold_{self.fold_id}_metrics.json')
            save_path.parent.mkdir(exist_ok=True)
            
            with open(save_path, 'w') as f:
                # Convert tensors to floats for JSON serialization
                serializable_metrics = {}
                for key, value in self.fold_metrics.items():
                    if isinstance(value, torch.Tensor):
                        serializable_metrics[key] = value.item()
                    else:
                        serializable_metrics[key] = value
                
                json.dump(serializable_metrics, f, indent=2)
            
            logger.info(f"Saved fold {self.fold_id} results to {save_path}")


class LossComponentCallback(Callback):
    """Monitor individual loss components during training"""
    
    def __init__(self, log_interval: int = 100):
        """
        Initialize loss component callback
        
        Args:
            log_interval: Log every N steps
        """
        self.log_interval = log_interval
        self.loss_history = {
            'phase_loss': [],
            'property_loss': [],
            'physics_loss': [],
            'total_loss': []
        }
    
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, 
                          outputs: Any, batch: Any, batch_idx: int):
        """Log loss components"""
        if batch_idx % self.log_interval == 0:
            # Extract loss components from model if available
            if hasattr(pl_module, 'last_loss_components'):
                loss_components = pl_module.last_loss_components
                
                for component, value in loss_components.items():
                    if isinstance(value, torch.Tensor):
                        self.loss_history[component].append(value.item())
                        pl_module.log(f'train/{component}_detailed', value.item())


class ModelComplexityCallback(Callback):
    """Monitor model complexity and overfitting"""
    
    def __init__(self, check_interval: int = 5):
        """
        Initialize model complexity callback
        
        Args:
            check_interval: Check every N epochs
        """
        self.check_interval = check_interval
        self.complexity_metrics = []
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Calculate and log model complexity metrics"""
        if trainer.current_epoch % self.check_interval != 0:
            return
        
        # Calculate effective number of parameters
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        
        # Calculate weight norms
        l1_norm = sum(p.abs().sum().item() for p in pl_module.parameters())
        l2_norm = sum(p.pow(2).sum().item() for p in pl_module.parameters())
        
        # Overfitting score (train loss - val loss)
        train_loss = trainer.logged_metrics.get('train/total_loss', 0)
        val_loss = trainer.logged_metrics.get('val/total_loss', 0)
        
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.item()
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()
        
        overfitting_score = train_loss - val_loss
        
        # Store metrics
        complexity_data = {
            'epoch': trainer.current_epoch,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'l1_norm': l1_norm,
            'l2_norm': l2_norm,
            'overfitting_score': overfitting_score
        }
        
        self.complexity_metrics.append(complexity_data)
        
        # Log metrics
        pl_module.log('model/total_params', float(total_params))
        pl_module.log('model/trainable_params', float(trainable_params))
        pl_module.log('model/l1_norm', l1_norm)
        pl_module.log('model/l2_norm', l2_norm)
        pl_module.log('model/overfitting_score', overfitting_score)
        
        logger.info(f"Epoch {trainer.current_epoch}: Overfitting score: {overfitting_score:.4f}")


def create_training_callbacks(config: Dict[str, Any]) -> List[Callback]:
    """
    Create standard training callbacks
    
    Args:
        config: Training configuration
        
    Returns:
        List of configured callbacks
    """
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val/total_loss',
        patience=config.get('early_stopping_patience', 20),
        mode='min',
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Model checkpointing
    checkpoint = ModelCheckpoint(
        monitor='val/phase_acc',
        mode='max',
        save_top_k=3,
        filename='{epoch}-{val_phase_acc:.3f}',
        verbose=True
    )
    callbacks.append(checkpoint)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Physics consistency
    if config.get('monitor_physics', True):
        physics_callback = PhysicsConsistencyCallback(
            check_interval=config.get('physics_check_interval', 10),
            tolerance=config.get('physics_tolerance', 0.1)
        )
        callbacks.append(physics_callback)
    
    # Uncertainty calibration
    if config.get('monitor_uncertainty', True):
        uncertainty_callback = UncertaintyCalibrationCallback(
            check_interval=config.get('uncertainty_check_interval', 5)
        )
        callbacks.append(uncertainty_callback)
    
    # Loss components
    loss_callback = LossComponentCallback(
        log_interval=config.get('loss_log_interval', 100)
    )
    callbacks.append(loss_callback)
    
    # Model complexity
    complexity_callback = ModelComplexityCallback(
        check_interval=config.get('complexity_check_interval', 5)
    )
    callbacks.append(complexity_callback)
    
    return callbacks


def create_cross_validation_callbacks(fold_id: int, total_folds: int, 
                                    config: Dict[str, Any]) -> List[Callback]:
    """
    Create callbacks for cross-validation training
    
    Args:
        fold_id: Current fold ID
        total_folds: Total number of folds
        config: Training configuration
        
    Returns:
        List of configured callbacks including CV-specific ones
    """
    callbacks = create_training_callbacks(config)
    
    # Add cross-validation callback
    cv_callback = CrossValidationCallback(
        fold_id=fold_id,
        total_folds=total_folds,
        save_fold_results=config.get('save_fold_results', True)
    )
    callbacks.append(cv_callback)
    
    return callbacks
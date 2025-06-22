#!/usr/bin/env python3
"""
CsPbBr₃ Digital Twin - Main Training Script
End-to-end training pipeline for physics-informed neural networks
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import digital twin components
from synthesis import (
    create_training_pipeline,
    TrainingConfig,
    ExperimentConfig,
    setup_logging,
    create_experiment_directory,
    save_experiment_data
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train CsPbBr₃ Digital Twin Neural Networks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    parser.add_argument("--data-dir", type=str, default="data/",
                       help="Directory containing training data")
    parser.add_argument("--data-file", type=str, default="synthesis_data.csv",
                       help="Training data filename")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.1,
                       help="Test split ratio")
    
    # Model configuration
    parser.add_argument("--input-dim", type=int, default=100,
                       help="Input feature dimension")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64],
                       help="Hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate")
    parser.add_argument("--activation", type=str, default="relu",
                       choices=["relu", "gelu", "swish"],
                       help="Activation function")
    
    # Physics constraints
    parser.add_argument("--physics-weight", type=float, default=0.2,
                       help="Physics constraint weight")
    parser.add_argument("--uncertainty-weight", type=float, default=0.1,
                       help="Uncertainty quantification weight")
    parser.add_argument("--constraint-weight", type=float, default=0.1,
                       help="Domain constraint weight")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                       help="Weight decay (L2 regularization)")
    parser.add_argument("--max-epochs", type=int, default=100,
                       help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=20,
                       help="Early stopping patience")
    parser.add_argument("--lr-scheduler", type=str, default="reduce_on_plateau",
                       choices=["reduce_on_plateau", "cosine", "none"],
                       help="Learning rate scheduler")
    
    # Multi-task weights
    parser.add_argument("--phase-weight", type=float, default=1.0,
                       help="Phase classification weight")
    parser.add_argument("--property-weight", type=float, default=1.0,
                       help="Property regression weight")
    
    # Cross-validation
    parser.add_argument("--n-folds", type=int, default=5,
                       help="Number of cross-validation folds")
    parser.add_argument("--fold", type=int, default=None,
                       help="Specific fold to train (0-indexed, None for all)")
    
    # Experiment configuration
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Experiment name (auto-generated if None)")
    parser.add_argument("--output-dir", type=str, default="experiments/",
                       help="Output directory for experiments")
    parser.add_argument("--description", type=str, default="",
                       help="Experiment description")
    
    # Logging and monitoring
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="cspbbr3-digital-twin",
                       help="W&B project name")
    parser.add_argument("--tags", type=str, nargs="*", default=[],
                       help="Experiment tags")
    
    # Hardware configuration
    parser.add_argument("--accelerator", type=str, default="auto",
                       choices=["auto", "cpu", "gpu", "tpu"],
                       help="Training accelerator")
    parser.add_argument("--devices", type=int, default=1,
                       help="Number of devices to use")
    parser.add_argument("--precision", type=str, default="32",
                       choices=["16", "32", "64"],
                       help="Training precision")
    
    # Resume training
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--pretrained-model", type=str, default=None,
                       help="Path to pretrained model")
    
    # Feature engineering
    parser.add_argument("--normalize-features", action="store_true", default=True,
                       help="Normalize engineered features")
    parser.add_argument("--include-interactions", action="store_true", default=True,
                       help="Include interaction features")
    
    # Data augmentation
    parser.add_argument("--augment-data", action="store_true",
                       help="Enable data augmentation")
    parser.add_argument("--noise-level", type=float, default=0.01,
                       help="Noise level for data augmentation")
    
    return parser.parse_args()


def create_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """Create experiment configuration from arguments"""
    experiment_name = args.experiment_name or f"cspbbr3_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    training_config = TrainingConfig(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout,
        activation=args.activation,
        physics_weight=args.physics_weight,
        uncertainty_weight=args.uncertainty_weight,
        constraint_weight=args.constraint_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
        phase_weight=args.phase_weight,
        property_weight=args.property_weight,
        n_folds=args.n_folds
    )
    
    data_config = {
        "data_dir": args.data_dir,
        "data_file": args.data_file,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "batch_size": args.batch_size,
        "normalize_features": args.normalize_features,
        "include_interactions": args.include_interactions,
        "augment_data": args.augment_data,
        "noise_level": args.noise_level
    }
    
    physics_config = {
        "physics_weight": args.physics_weight,
        "uncertainty_weight": args.uncertainty_weight,
        "constraint_weight": args.constraint_weight
    }
    
    return ExperimentConfig(
        experiment_id=experiment_name,
        description=args.description,
        training_config=training_config,
        data_config=data_config,
        physics_config=physics_config,
        output_dir=os.path.join(args.output_dir, experiment_name)
    )


def setup_trainer(args: argparse.Namespace, 
                 experiment_config: ExperimentConfig,
                 callbacks: List[pl.Callback]) -> pl.Trainer:
    """Setup PyTorch Lightning trainer"""
    
    # Setup loggers
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=experiment_config.output_dir,
        name="tensorboard_logs",
        version="",
        default_hp_metric=False
    )
    loggers.append(tb_logger)
    
    # Weights & Biases logger (optional)
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=experiment_config.experiment_id,
            tags=args.tags,
            save_dir=experiment_config.output_dir
        )
        wandb_logger.experiment.config.update(vars(args))
        loggers.append(wandb_logger)
    
    # Additional callbacks
    additional_callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(experiment_config.output_dir, "checkpoints"),
            filename="best_model_{epoch:02d}_{val_total_loss:.3f}",
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor="val/total_loss",
            patience=args.patience,
            mode="min",
            verbose=True
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]
    
    all_callbacks = callbacks + additional_callbacks
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=all_callbacks,
        logger=loggers,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        check_val_every_n_epoch=1
    )
    
    return trainer


def train_single_fold(args: argparse.Namespace, 
                     experiment_config: ExperimentConfig,
                     fold: Optional[int] = None) -> Dict[str, Any]:
    """Train a single model or fold"""
    
    logger.info(f"Training fold {fold if fold is not None else 'single model'}")
    
    # Create training pipeline
    model, data_module, callbacks = create_training_pipeline(
        experiment_config.training_config,
        experiment_config.data_config
    )
    
    # Setup trainer
    trainer = setup_trainer(args, experiment_config, callbacks)
    
    # Train model
    if args.resume_from:
        logger.info(f"Resuming training from {args.resume_from}")
        trainer.fit(model, data_module, ckpt_path=args.resume_from)
    else:
        trainer.fit(model, data_module)
    
    # Test model
    test_results = trainer.test(model, data_module)
    
    # Save results
    fold_suffix = f"_fold_{fold}" if fold is not None else ""
    results = {
        "test_results": test_results,
        "best_model_path": trainer.checkpoint_callback.best_model_path,
        "fold": fold
    }
    
    results_path = os.path.join(
        experiment_config.output_dir, 
        f"results{fold_suffix}.json"
    )
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def run_cross_validation(args: argparse.Namespace, 
                        experiment_config: ExperimentConfig) -> Dict[str, Any]:
    """Run k-fold cross-validation"""
    
    logger.info(f"Running {args.n_folds}-fold cross-validation")
    
    all_results = []
    
    for fold in range(args.n_folds):
        if args.fold is not None and fold != args.fold:
            continue
            
        logger.info(f"Training fold {fold + 1}/{args.n_folds}")
        
        # Update data config for this fold
        fold_data_config = experiment_config.data_config.copy()
        fold_data_config["current_fold"] = fold
        fold_data_config["n_folds"] = args.n_folds
        
        # Create fold-specific experiment config
        fold_experiment_config = ExperimentConfig(
            experiment_id=f"{experiment_config.experiment_id}_fold_{fold}",
            description=f"Fold {fold} - {experiment_config.description}",
            training_config=experiment_config.training_config,
            data_config=fold_data_config,
            physics_config=experiment_config.physics_config,
            output_dir=os.path.join(experiment_config.output_dir, f"fold_{fold}")
        )
        
        # Train fold
        fold_results = train_single_fold(args, fold_experiment_config, fold)
        all_results.append(fold_results)
    
    # Aggregate results
    aggregated_results = {
        "cross_validation_results": all_results,
        "n_folds": args.n_folds,
        "experiment_config": experiment_config.__dict__
    }
    
    # Save aggregated results
    cv_results_path = os.path.join(experiment_config.output_dir, "cv_results.json")
    with open(cv_results_path, 'w') as f:
        json.dump(aggregated_results, f, indent=2, default=str)
    
    return aggregated_results


def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger.info("Starting CsPbBr₃ Digital Twin training")
    
    # Create experiment configuration
    experiment_config = create_experiment_config(args)
    
    # Create experiment directory
    create_experiment_directory(experiment_config.output_dir)
    
    # Save experiment configuration
    config_path = os.path.join(experiment_config.output_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(experiment_config.__dict__, f, indent=2, default=str)
    
    # Save command line arguments
    args_path = os.path.join(experiment_config.output_dir, "args.json")
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"Experiment: {experiment_config.experiment_id}")
    logger.info(f"Output directory: {experiment_config.output_dir}")
    
    try:
        # Run training
        if args.n_folds > 1 and args.fold is None:
            # Cross-validation
            results = run_cross_validation(args, experiment_config)
            logger.info("Cross-validation completed successfully")
        else:
            # Single model training
            results = train_single_fold(args, experiment_config, args.fold)
            logger.info("Training completed successfully")
        
        # Save final results
        save_experiment_data(experiment_config.output_dir, "final_results.json", results)
        
        logger.info(f"All results saved to: {experiment_config.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
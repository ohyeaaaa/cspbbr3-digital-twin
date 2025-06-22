#!/usr/bin/env python3
"""
Enhanced Training Script with Uncertainty Quantification
Train Bayesian and MC Dropout models for better prediction confidence
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as sklearn_split
import os
import json
import argparse
import logging
from pathlib import Path
import pickle
from datetime import datetime

from uncertainty_models import (
    BayesianNeuralNetwork, MCDropoutNeuralNetwork, EnsembleNeuralNetwork,
    train_bayesian_model, calibrate_uncertainty
)
from simple_train import SynthesisDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UncertaintyTrainingPipeline:
    """Training pipeline for uncertainty-aware models"""
    
    def __init__(self, output_dir: str = "uncertainty_training_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.scaler = None
        self.feature_columns = None
        self.models = {}
        self.training_histories = {}
        self.calibration_results = {}
    
    def load_and_prepare_data(self, data_file: str):
        """Load and prepare training data"""
        logger.info(f"Loading data from {data_file}")
        
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        
        # Feature columns (exclude target variables)
        self.feature_columns = [col for col in df.columns if col not in 
                               ['phase_label', 'bandgap', 'plqy', 'particle_size', 'emission_peak']]
        
        # Prepare features
        X = df[self.feature_columns].values
        
        # Prepare targets
        y_phase = df['phase_label'].values
        y_properties = {
            'bandgap': df['bandgap'].values,
            'plqy': df['plqy'].values,
            'particle_size': df['particle_size'].values,
            'emission_peak': df['emission_peak'].values
        }
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Features shape: {X_scaled.shape}")
        logger.info(f"Phase distribution: {np.bincount(y_phase)}")
        
        return X_scaled, y_phase, y_properties
    
    def create_data_loaders(self, X, y_phase, y_properties, batch_size=32, test_size=0.2):
        """Create train/validation data loaders"""
        
        # Split data
        X_train, X_val, y_phase_train, y_phase_val = sklearn_split(
            X, y_phase, test_size=test_size, random_state=42, stratify=y_phase
        )
        
        # Split properties using the same indices
        _, _, *property_splits = sklearn_split(
            X, y_phase, *y_properties.values(), 
            test_size=test_size, random_state=42, stratify=y_phase
        )
        
        # Reconstruct property dictionaries
        prop_names = list(y_properties.keys())
        y_props_train = {prop_names[i]: property_splits[i*2] for i in range(len(prop_names))}
        y_props_val = {prop_names[i]: property_splits[i*2+1] for i in range(len(prop_names))}
        
        # Create datasets
        train_dataset = SynthesisDataset(X_train, y_phase_train, y_props_train)
        val_dataset = SynthesisDataset(X_val, y_phase_val, y_props_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_bayesian_model(self, train_loader, val_loader, input_dim, epochs=50):
        """Train Bayesian Neural Network"""
        logger.info("🔮 Training Bayesian Neural Network")
        
        model = BayesianNeuralNetwork(input_dim)
        
        # Train with ELBO loss
        history = train_bayesian_model(model, train_loader, val_loader, epochs=epochs)
        
        # Save model
        model_path = self.output_dir / "bayesian_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Calibrate uncertainty
        calibration = calibrate_uncertainty(model, val_loader)
        
        self.models['bayesian'] = model
        self.training_histories['bayesian'] = history
        self.calibration_results['bayesian'] = calibration
        
        logger.info(f"✅ Bayesian model trained. Uncertainty correlation: {calibration['uncertainty_accuracy_correlation']:.3f}")
        
        return model, history, calibration
    
    def train_mcdropout_model(self, train_loader, val_loader, input_dim, epochs=30):
        """Train MC Dropout Neural Network"""
        logger.info("🎲 Training MC Dropout Neural Network")
        
        model = MCDropoutNeuralNetwork(input_dim, dropout_rate=0.3)
        
        # Standard training with dropout
        history = self._train_standard_model(model, train_loader, val_loader, epochs)
        
        # Save model
        model_path = self.output_dir / "mcdropout_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Calibrate uncertainty
        calibration = calibrate_uncertainty(model, val_loader)
        
        self.models['mcdropout'] = model
        self.training_histories['mcdropout'] = history
        self.calibration_results['mcdropout'] = calibration
        
        logger.info(f"✅ MC Dropout model trained. Uncertainty correlation: {calibration['uncertainty_accuracy_correlation']:.3f}")
        
        return model, history, calibration
    
    def train_ensemble_models(self, train_loader, val_loader, input_dim, n_models=5, epochs=20):
        """Train ensemble of models"""
        logger.info(f"👥 Training Ensemble of {n_models} models")
        
        ensemble = EnsembleNeuralNetwork(input_dim, n_models=n_models)
        
        # Train each model in ensemble separately
        for i, model in enumerate(ensemble.models):
            logger.info(f"Training ensemble model {i+1}/{n_models}")
            history = self._train_standard_model(model, train_loader, val_loader, epochs)
        
        # Save ensemble
        model_path = self.output_dir / "ensemble_model.pth"
        torch.save(ensemble.state_dict(), model_path)
        
        # Calibrate uncertainty
        calibration = calibrate_uncertainty(ensemble, val_loader)
        
        self.models['ensemble'] = ensemble
        self.training_histories['ensemble'] = {'train_loss': [], 'val_loss': []}  # Simplified
        self.calibration_results['ensemble'] = calibration
        
        logger.info(f"✅ Ensemble trained. Uncertainty correlation: {calibration['uncertainty_accuracy_correlation']:.3f}")
        
        return ensemble, {}, calibration
    
    def _train_standard_model(self, model, train_loader, val_loader, epochs):
        """Standard training loop for non-Bayesian models"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Loss functions and optimizer
        phase_criterion = nn.CrossEntropyLoss()
        property_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                features = batch['features'].to(device)
                phase_labels = batch['phase_labels'].to(device)
                properties = {k: v.to(device) for k, v in batch['properties'].items()}
                
                outputs = model(features)
                
                # Calculate losses
                phase_loss = phase_criterion(outputs['phase_logits'], phase_labels)
                
                property_loss = 0
                for prop_name, pred in outputs['properties'].items():
                    target = properties[prop_name]
                    property_loss += property_criterion(pred, target)
                
                total_loss = phase_loss + 0.5 * property_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct_phases = 0
            val_samples = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(device)
                    phase_labels = batch['phase_labels'].to(device)
                    properties = {k: v.to(device) for k, v in batch['properties'].items()}
                    
                    outputs = model(features)
                    
                    phase_loss = phase_criterion(outputs['phase_logits'], phase_labels)
                    
                    property_loss = 0
                    for prop_name, pred in outputs['properties'].items():
                        target = properties[prop_name]
                        property_loss += property_criterion(pred, target)
                    
                    total_loss = phase_loss + 0.5 * property_loss
                    val_loss += total_loss.item()
                    
                    # Accuracy
                    _, predicted = torch.max(outputs['phase_logits'].data, 1)
                    correct_phases += (predicted == phase_labels).sum().item()
                    val_samples += features.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct_phases / val_samples
            
            train_history['train_loss'].append(avg_train_loss)
            train_history['val_loss'].append(avg_val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if hasattr(self, 'output_dir'):
                    torch.save(model.state_dict(), self.output_dir / 'best_standard_model.pth')
        
        return train_history
    
    def compare_models(self, val_loader):
        """Compare different uncertainty quantification approaches"""
        logger.info("📊 Comparing model uncertainties")
        
        comparison_results = {}
        
        for model_name, model in self.models.items():
            calibration = self.calibration_results[model_name]
            
            comparison_results[model_name] = {
                'uncertainty_accuracy_correlation': float(calibration['uncertainty_accuracy_correlation']),
                'mean_uncertainty': float(calibration['mean_uncertainty']),
                'uncertainty_std': float(calibration['uncertainty_std']),
                'calibration_quality': len(calibration['calibration_curve'])
            }
        
        # Save comparison
        comparison_file = self.output_dir / "model_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        logger.info("Model Comparison:")
        for model_name, results in comparison_results.items():
            logger.info(f"  {model_name}: "
                       f"Uncertainty-Accuracy Correlation = {results['uncertainty_accuracy_correlation']:.3f}, "
                       f"Mean Uncertainty = {results['mean_uncertainty']:.3f}")
        
        return comparison_results
    
    def save_training_results(self):
        """Save all training results"""
        
        # Training configuration
        config = {
            'feature_columns': self.feature_columns,
            'training_date': datetime.now().isoformat(),
            'models_trained': list(self.models.keys()),
            'model_configs': {
                'bayesian': {
                    'type': 'BayesianNeuralNetwork',
                    'input_dim': len(self.feature_columns),
                    'hidden_dims': [128, 64],
                    'uncertainty_method': 'Bayesian'
                },
                'mcdropout': {
                    'type': 'MCDropoutNeuralNetwork', 
                    'input_dim': len(self.feature_columns),
                    'hidden_dims': [128, 64],
                    'uncertainty_method': 'MC Dropout'
                },
                'ensemble': {
                    'type': 'EnsembleNeuralNetwork',
                    'input_dim': len(self.feature_columns),
                    'n_models': 5,
                    'uncertainty_method': 'Ensemble'
                }
            }
        }
        
        # Save configuration
        config_file = self.output_dir / "uncertainty_training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save scaler
        scaler_file = self.output_dir / "uncertainty_scaler.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save training histories
        history_file = self.output_dir / "training_histories.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_histories, f, indent=2)
        
        # Save calibration results
        calibration_file = self.output_dir / "calibration_results.json"
        with open(calibration_file, 'w') as f:
            json.dump(self.calibration_results, f, indent=2, default=str)
        
        logger.info(f"✅ All results saved to {self.output_dir}")
    
    def create_best_model_selection(self):
        """Select and save the best uncertainty model"""
        
        if not self.calibration_results:
            logger.warning("No calibration results available for model selection")
            return
        
        # Rank models by uncertainty-accuracy correlation
        model_scores = {}
        for model_name, calibration in self.calibration_results.items():
            correlation = calibration['uncertainty_accuracy_correlation']
            model_scores[model_name] = correlation
        
        # Select best model
        best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k])
        best_model = self.models[best_model_name]
        
        # Save best model
        best_model_path = self.output_dir / "best_uncertainty_model.pth"
        torch.save(best_model.state_dict(), best_model_path)
        
        # Save model info
        best_model_info = {
            'model_type': best_model_name,
            'uncertainty_accuracy_correlation': model_scores[best_model_name],
            'model_path': str(best_model_path),
            'scaler_path': str(self.output_dir / "uncertainty_scaler.pkl"),
            'feature_columns': self.feature_columns,
            'selection_date': datetime.now().isoformat()
        }
        
        info_file = self.output_dir / "best_model_info.json"
        with open(info_file, 'w') as f:
            json.dump(best_model_info, f, indent=2)
        
        logger.info(f"🏆 Best model selected: {best_model_name} "
                   f"(correlation: {model_scores[best_model_name]:.3f})")
        
        return best_model_name, best_model

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train uncertainty-aware models")
    parser.add_argument('--data-file', type=str, default='training_data.csv',
                       help='Training data CSV file')
    parser.add_argument('--epochs-bayesian', type=int, default=50,
                       help='Epochs for Bayesian training')
    parser.add_argument('--epochs-mcdropout', type=int, default=30,
                       help='Epochs for MC Dropout training')
    parser.add_argument('--epochs-ensemble', type=int, default=20,
                       help='Epochs for ensemble training')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--output-dir', type=str, default='uncertainty_training_output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("🔮 Training Uncertainty-Aware CsPbBr₃ Digital Twin")
    print("=" * 60)
    
    # Initialize training pipeline
    pipeline = UncertaintyTrainingPipeline(args.output_dir)
    
    # Load data
    X, y_phase, y_properties = pipeline.load_and_prepare_data(args.data_file)
    train_loader, val_loader = pipeline.create_data_loaders(
        X, y_phase, y_properties, batch_size=args.batch_size
    )
    
    input_dim = X.shape[1]
    
    # Train all models
    print("\n🚀 Training all uncertainty models...")
    
    # Bayesian Neural Network
    pipeline.train_bayesian_model(train_loader, val_loader, input_dim, args.epochs_bayesian)
    
    # MC Dropout Neural Network  
    pipeline.train_mcdropout_model(train_loader, val_loader, input_dim, args.epochs_mcdropout)
    
    # Ensemble Neural Network
    pipeline.train_ensemble_models(train_loader, val_loader, input_dim, epochs=args.epochs_ensemble)
    
    # Compare models
    print("\n📊 Comparing model performance...")
    comparison = pipeline.compare_models(val_loader)
    
    # Select best model
    print("\n🏆 Selecting best model...")
    best_name, best_model = pipeline.create_best_model_selection()
    
    # Save all results
    print("\n💾 Saving training results...")
    pipeline.save_training_results()
    
    print(f"\n✅ Uncertainty training completed!")
    print(f"📂 Results saved to: {args.output_dir}")
    print(f"🏆 Best model: {best_name}")
    print(f"\n💡 Next steps:")
    print(f"   1. Use best model for improved predictions")
    print(f"   2. Integrate with active learning system")
    print(f"   3. Update validation pipeline with uncertainty")

if __name__ == "__main__":
    main()
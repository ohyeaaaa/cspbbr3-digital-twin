#!/usr/bin/env python3
"""
Uncertainty Quantification Models for CsPbBr₃ Digital Twin
Bayesian Neural Networks and Monte Carlo Dropout for prediction confidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty"""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_log_sigma = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 2)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_log_sigma = nn.Parameter(torch.randn(out_features) * 0.1 - 2)
        
        # Prior
        self.prior_std = prior_std
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample weights from posterior
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        
        bias_sigma = torch.exp(self.bias_log_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior"""
        # Weight KL
        weight_var = torch.exp(2 * self.weight_log_sigma)
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu**2 + weight_var) / (self.prior_std**2) - 
            1 - 2 * self.weight_log_sigma + 2 * torch.log(torch.tensor(self.prior_std))
        )
        
        # Bias KL
        bias_var = torch.exp(2 * self.bias_log_sigma)
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu**2 + bias_var) / (self.prior_std**2) - 
            1 - 2 * self.bias_log_sigma + 2 * torch.log(torch.tensor(self.prior_std))
        )
        
        return weight_kl + bias_kl

class MCDropoutLinear(nn.Module):
    """Linear layer with Monte Carlo Dropout for uncertainty"""
    
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(x))

class BayesianNeuralNetwork(nn.Module):
    """Bayesian Neural Network for uncertainty quantification"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], 
                 n_phases: int = 5, n_properties: int = 4, prior_std: float = 1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_phases = n_phases
        self.n_properties = n_properties
        
        # Build Bayesian layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim, prior_std))
            prev_dim = hidden_dim
        
        self.feature_layers = nn.ModuleList(layers)
        
        # Output heads
        self.phase_classifier = BayesianLinear(prev_dim, n_phases, prior_std)
        self.property_regressors = nn.ModuleDict({
            'bandgap': BayesianLinear(prev_dim, 1, prior_std),
            'plqy': BayesianLinear(prev_dim, 1, prior_std),
            'particle_size': BayesianLinear(prev_dim, 1, prior_std),
            'emission_peak': BayesianLinear(prev_dim, 1, prior_std)
        })
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Forward through feature layers
        features = x
        for layer in self.feature_layers:
            features = F.relu(layer(features))
        
        # Phase classification
        phase_logits = self.phase_classifier(features)
        
        # Property regression
        properties = {}
        for prop_name, regressor in self.property_regressors.items():
            properties[prop_name] = regressor(features).squeeze()
        
        return {
            'phase_logits': phase_logits,
            'properties': properties
        }
    
    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence for all layers"""
        kl_total = torch.tensor(0.0)
        
        for layer in self.feature_layers:
            kl_total += layer.kl_divergence()
        
        kl_total += self.phase_classifier.kl_divergence()
        
        for regressor in self.property_regressors.values():
            kl_total += regressor.kl_divergence()
        
        return kl_total
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """Make predictions with uncertainty quantification"""
        self.train()  # Enable sampling
        
        phase_samples = []
        property_samples = {prop: [] for prop in self.property_regressors.keys()}
        
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.forward(x)
                
                # Collect phase predictions
                phase_probs = F.softmax(outputs['phase_logits'], dim=-1)
                phase_samples.append(phase_probs)
                
                # Collect property predictions
                for prop_name, pred in outputs['properties'].items():
                    property_samples[prop_name].append(pred)
        
        # Compute statistics
        phase_samples = torch.stack(phase_samples, dim=0)  # [num_samples, batch_size, n_phases]
        phase_mean = phase_samples.mean(dim=0)
        phase_std = phase_samples.std(dim=0)
        
        property_stats = {}
        for prop_name, samples in property_samples.items():
            samples = torch.stack(samples, dim=0)
            property_stats[prop_name] = {
                'mean': samples.mean(dim=0),
                'std': samples.std(dim=0),
                'quantile_025': torch.quantile(samples, 0.025, dim=0),
                'quantile_975': torch.quantile(samples, 0.975, dim=0)
            }
        
        return {
            'phase_probabilities': {
                'mean': phase_mean,
                'std': phase_std,
                'samples': phase_samples
            },
            'properties': property_stats
        }

class MCDropoutNeuralNetwork(nn.Module):
    """Monte Carlo Dropout Neural Network for uncertainty"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], 
                 n_phases: int = 5, n_properties: int = 4, dropout_rate: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_phases = n_phases
        self.n_properties = n_properties
        self.dropout_rate = dropout_rate
        
        # Build network with dropout
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads
        self.phase_classifier = nn.Linear(prev_dim, n_phases)
        self.property_regressors = nn.ModuleDict({
            'bandgap': nn.Linear(prev_dim, 1),
            'plqy': nn.Linear(prev_dim, 1),
            'particle_size': nn.Linear(prev_dim, 1),
            'emission_peak': nn.Linear(prev_dim, 1)
        })
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        
        phase_logits = self.phase_classifier(features)
        
        properties = {}
        for prop_name, regressor in self.property_regressors.items():
            properties[prop_name] = regressor(features).squeeze()
        
        return {
            'phase_logits': phase_logits,
            'properties': properties
        }
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """Make predictions with MC Dropout uncertainty"""
        self.train()  # Keep dropout active
        
        phase_samples = []
        property_samples = {prop: [] for prop in self.property_regressors.keys()}
        
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.forward(x)
                
                # Collect phase predictions
                phase_probs = F.softmax(outputs['phase_logits'], dim=-1)
                phase_samples.append(phase_probs)
                
                # Collect property predictions
                for prop_name, pred in outputs['properties'].items():
                    property_samples[prop_name].append(pred)
        
        # Compute statistics (same as Bayesian)
        phase_samples = torch.stack(phase_samples, dim=0)
        phase_mean = phase_samples.mean(dim=0)
        phase_std = phase_samples.std(dim=0)
        
        property_stats = {}
        for prop_name, samples in property_samples.items():
            samples = torch.stack(samples, dim=0)
            property_stats[prop_name] = {
                'mean': samples.mean(dim=0),
                'std': samples.std(dim=0),
                'quantile_025': torch.quantile(samples, 0.025, dim=0),
                'quantile_975': torch.quantile(samples, 0.975, dim=0)
            }
        
        return {
            'phase_probabilities': {
                'mean': phase_mean,
                'std': phase_std,
                'samples': phase_samples
            },
            'properties': property_stats
        }

class EnsembleNeuralNetwork(nn.Module):
    """Ensemble of models for uncertainty quantification"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], 
                 n_phases: int = 5, n_properties: int = 4, n_models: int = 5):
        super().__init__()
        
        from simple_train import SimpleNeuralNetwork
        
        self.models = nn.ModuleList([
            SimpleNeuralNetwork(input_dim, hidden_dims, n_phases, n_properties)
            for _ in range(n_models)
        ])
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Just use first model for training
        return self.models[0](x)
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = None) -> Dict[str, torch.Tensor]:
        """Ensemble predictions for uncertainty"""
        
        phase_predictions = []
        property_predictions = {prop: [] for prop in ['bandgap', 'plqy', 'particle_size', 'emission_peak']}
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                outputs = model(x)
                
                # Phase predictions
                phase_probs = F.softmax(outputs['phase_logits'], dim=-1)
                phase_predictions.append(phase_probs)
                
                # Property predictions
                for prop_name, pred in outputs['properties'].items():
                    property_predictions[prop_name].append(pred)
        
        # Compute ensemble statistics
        phase_samples = torch.stack(phase_predictions, dim=0)
        phase_mean = phase_samples.mean(dim=0)
        phase_std = phase_samples.std(dim=0)
        
        property_stats = {}
        for prop_name, predictions in property_predictions.items():
            samples = torch.stack(predictions, dim=0)
            property_stats[prop_name] = {
                'mean': samples.mean(dim=0),
                'std': samples.std(dim=0),
                'quantile_025': torch.quantile(samples, 0.025, dim=0),
                'quantile_975': torch.quantile(samples, 0.975, dim=0)
            }
        
        return {
            'phase_probabilities': {
                'mean': phase_mean,
                'std': phase_std,
                'samples': phase_samples
            },
            'properties': property_stats
        }

def train_bayesian_model(model: BayesianNeuralNetwork, train_loader, val_loader, 
                        epochs: int = 50, lr: float = 0.001, kl_weight: float = 1e-3):
    """Train Bayesian neural network with ELBO loss"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss functions
    phase_criterion = nn.CrossEntropyLoss()
    property_criterion = nn.MSELoss()
    
    train_history = {'train_loss': [], 'val_loss': [], 'kl_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        kl_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            features = batch['features'].to(device)
            phase_labels = batch['phase_labels'].to(device)
            properties = {k: v.to(device) for k, v in batch['properties'].items()}
            
            # Forward pass
            outputs = model(features)
            
            # Compute likelihood losses
            phase_loss = phase_criterion(outputs['phase_logits'], phase_labels)
            
            property_loss = 0
            for prop_name, pred in outputs['properties'].items():
                target = properties[prop_name]
                property_loss += property_criterion(pred, target)
            
            # KL divergence
            kl_div = model.kl_divergence()
            
            # ELBO loss
            total_loss = phase_loss + 0.5 * property_loss + kl_weight * kl_div
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            kl_loss += kl_div.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
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
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_kl_loss = kl_loss / len(train_loader)
        
        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        train_history['kl_loss'].append(avg_kl_loss)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
    
    return train_history

def calibrate_uncertainty(model, val_loader, num_samples: int = 100):
    """Calibrate uncertainty estimates using validation data"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            phase_labels = batch['phase_labels'].to(device)
            
            # Get uncertainty predictions
            uncertainty_outputs = model.predict_with_uncertainty(features, num_samples)
            
            # Extract predictions and uncertainties
            phase_probs = uncertainty_outputs['phase_probabilities']['mean']
            phase_std = uncertainty_outputs['phase_probabilities']['std']
            
            predicted_phase = torch.argmax(phase_probs, dim=-1)
            max_prob = torch.max(phase_probs, dim=-1)[0]
            uncertainty = torch.max(phase_std, dim=-1)[0]
            
            all_predictions.extend(predicted_phase.cpu().numpy())
            all_targets.extend(phase_labels.cpu().numpy())
            all_uncertainties.extend(uncertainty.cpu().numpy())
    
    # Compute calibration metrics
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    uncertainties = np.array(all_uncertainties)
    
    # Accuracy vs uncertainty correlation
    correct = (predictions == targets).astype(float)
    uncertainty_accuracy_corr = np.corrcoef(uncertainties, correct)[0, 1]
    
    # Calibration curve
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    calibration_data = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_uncertainty_in_bin = uncertainties[in_bin].mean()
            
            calibration_data.append({
                'uncertainty_range': (bin_lower, bin_upper),
                'avg_uncertainty': avg_uncertainty_in_bin,
                'accuracy': accuracy_in_bin,
                'count': in_bin.sum()
            })
    
    return {
        'uncertainty_accuracy_correlation': uncertainty_accuracy_corr,
        'calibration_curve': calibration_data,
        'mean_uncertainty': uncertainties.mean(),
        'uncertainty_std': uncertainties.std()
    }

def main():
    """Test uncertainty quantification models"""
    
    # Create synthetic data for testing
    torch.manual_seed(42)
    batch_size = 32
    input_dim = 12
    n_samples = 1000
    
    # Synthetic features and targets
    features = torch.randn(n_samples, input_dim)
    phase_labels = torch.randint(0, 5, (n_samples,))
    properties = {
        'bandgap': torch.randn(n_samples),
        'plqy': torch.randn(n_samples),
        'particle_size': torch.randn(n_samples),
        'emission_peak': torch.randn(n_samples)
    }
    
    # Create datasets
    from simple_train import SynthesisDataset
    dataset = SynthesisDataset(features, phase_labels, properties)
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("🔮 Testing Uncertainty Quantification Models")
    print("=" * 50)
    
    # Test Bayesian Neural Network
    print("\n1. Bayesian Neural Network")
    bayesian_model = BayesianNeuralNetwork(input_dim)
    
    # Test forward pass
    test_input = torch.randn(5, input_dim)
    uncertainty_output = bayesian_model.predict_with_uncertainty(test_input, num_samples=50)
    
    print(f"   Phase uncertainty shape: {uncertainty_output['phase_probabilities']['std'].shape}")
    print(f"   Property uncertainty available: {list(uncertainty_output['properties'].keys())}")
    
    # Test MC Dropout
    print("\n2. MC Dropout Neural Network")
    mcdropout_model = MCDropoutNeuralNetwork(input_dim)
    uncertainty_output = mcdropout_model.predict_with_uncertainty(test_input, num_samples=50)
    
    print(f"   Phase uncertainty shape: {uncertainty_output['phase_probabilities']['std'].shape}")
    print(f"   Mean phase uncertainty: {uncertainty_output['phase_probabilities']['std'].mean():.4f}")
    
    # Test Ensemble
    print("\n3. Ensemble Neural Network")
    ensemble_model = EnsembleNeuralNetwork(input_dim, n_models=3)
    uncertainty_output = ensemble_model.predict_with_uncertainty(test_input)
    
    print(f"   Ensemble uncertainty shape: {uncertainty_output['phase_probabilities']['std'].shape}")
    print(f"   Mean ensemble uncertainty: {uncertainty_output['phase_probabilities']['std'].mean():.4f}")
    
    print("\n✅ All uncertainty models tested successfully!")

if __name__ == "__main__":
    main()
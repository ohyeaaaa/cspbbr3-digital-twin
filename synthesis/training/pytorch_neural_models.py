#!/usr/bin/env python3
"""
PyTorch Neural Network Models for CsPbBr₃ Digital Twin
Physics-Informed Multi-Task Neural Networks with Uncertainty Quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError, MeanSquaredError, F1Score
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for neural network models"""
    # Network architecture
    input_dim: int = 100
    hidden_dims: List[int] = None
    dropout_rate: float = 0.3
    activation: str = "relu"
    
    # Physics constraints
    physics_weight: float = 0.2
    uncertainty_weight: float = 0.1
    constraint_weight: float = 0.1
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    lr_scheduler: str = "reduce_on_plateau"
    
    # Multi-task weights
    phase_weight: float = 1.0
    property_weight: float = 1.0
    uncertainty_weight_final: float = 0.1
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


class PhysicsConstraintLayer(nn.Module):
    """Physics-informed constraint layer enforcing domain knowledge"""
    
    def __init__(self, input_dim: int, constraint_type: str = "positive"):
        super().__init__()
        self.constraint_type = constraint_type
        self.constraint_weights = nn.Parameter(torch.ones(input_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply physics constraints to features"""
        if self.constraint_type == "positive":
            # Ensure positive values for physical quantities like rates, concentrations
            return F.relu(x) * self.constraint_weights
        elif self.constraint_type == "bounded":
            # Bound values between 0 and 1 for probabilities, fractions
            return torch.sigmoid(x) * self.constraint_weights
        elif self.constraint_type == "normalized":
            # L2 normalization for direction vectors
            return F.normalize(x, p=2, dim=-1) * self.constraint_weights
        else:
            return x * self.constraint_weights


class UncertaintyLayer(nn.Module):
    """Variational layer for uncertainty quantification"""
    
    def __init__(self, input_dim: int, output_dim: int, uncertainty_type: str = "aleatoric"):
        super().__init__()
        self.uncertainty_type = uncertainty_type
        
        if uncertainty_type == "aleatoric":
            # Data-dependent uncertainty
            self.mean_layer = nn.Linear(input_dim, output_dim)
            self.logvar_layer = nn.Linear(input_dim, output_dim)
        elif uncertainty_type == "epistemic":
            # Model uncertainty via Monte Carlo dropout
            self.layer = nn.Linear(input_dim, output_dim)
            self.dropout = nn.Dropout(0.5)
        else:
            # Combined uncertainty
            self.mean_layer = nn.Linear(input_dim, output_dim)
            self.logvar_layer = nn.Linear(input_dim, output_dim)
            self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty quantification"""
        if self.uncertainty_type == "aleatoric":
            mean = self.mean_layer(x)
            logvar = self.logvar_layer(x)
            std = torch.exp(0.5 * logvar)
            return mean, std
            
        elif self.uncertainty_type == "epistemic":
            if training:
                x = self.dropout(x)
            output = self.layer(x)
            # Uncertainty estimated via multiple forward passes during inference
            return output, torch.ones_like(output) * 0.1
            
        else:  # combined
            if training:
                x = self.dropout(x)
            mean = self.mean_layer(x)
            logvar = self.logvar_layer(x)
            std = torch.exp(0.5 * logvar)
            return mean, std


class PhaseClassificationHead(nn.Module):
    """Multi-class phase classification with uncertainty"""
    
    def __init__(self, input_dim: int, num_phases: int = 5, config: ModelConfig = None):
        super().__init__()
        self.num_phases = num_phases
        self.config = config or ModelConfig()
        
        # Classification layers
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate)
        )
        
        # Uncertainty-aware classification
        self.uncertainty_layer = UncertaintyLayer(64, num_phases, "combined")
        
        # Physics constraint (probabilities must sum to 1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor, training: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass for phase classification"""
        features = self.feature_layers(x)
        logits, uncertainty = self.uncertainty_layer(features, training)
        
        # Apply softmax for probabilities
        probabilities = self.softmax(logits)
        
        return {
            "logits": logits,
            "probabilities": probabilities,
            "uncertainty": uncertainty,
            "features": features
        }


class PropertyRegressionHead(nn.Module):
    """Property regression for each phase with physics constraints"""
    
    def __init__(self, input_dim: int, num_properties: int = 8, config: ModelConfig = None):
        super().__init__()
        self.num_properties = num_properties
        self.config = config or ModelConfig()
        
        # Property-specific networks
        self.property_networks = nn.ModuleDict({
            "bandgap": self._create_property_network(input_dim, constraint_type="positive"),
            "plqy": self._create_property_network(input_dim, constraint_type="bounded"),
            "emission_peak": self._create_property_network(input_dim, constraint_type="positive"),
            "emission_fwhm": self._create_property_network(input_dim, constraint_type="positive"),
            "particle_size": self._create_property_network(input_dim, constraint_type="positive"),
            "size_distribution": self._create_property_network(input_dim, constraint_type="positive"),
            "lifetime": self._create_property_network(input_dim, constraint_type="positive"),
            "stability_score": self._create_property_network(input_dim, constraint_type="bounded")
        })
        
    def _create_property_network(self, input_dim: int, constraint_type: str) -> nn.Module:
        """Create network for individual property with appropriate constraints"""
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            PhysicsConstraintLayer(32, constraint_type),
            UncertaintyLayer(32, 1, "aleatoric")
        )
    
    def forward(self, x: torch.Tensor, training: bool = True) -> Dict[str, Dict[str, torch.Tensor]]:
        """Forward pass for property regression"""
        outputs = {}
        
        for property_name, network in self.property_networks.items():
            # Get layers except the last uncertainty layer
            features = x
            for layer in network[:-1]:
                features = layer(features)
            
            # Apply uncertainty layer
            mean, std = network[-1](features, training)
            
            outputs[property_name] = {
                "mean": mean.squeeze(-1),
                "std": std.squeeze(-1),
                "features": features
            }
        
        return outputs


class PhysicsInformedNeuralNetwork(pl.LightningModule):
    """
    Main Physics-Informed Neural Network for CsPbBr₃ synthesis prediction
    Combines multi-task learning with uncertainty quantification
    """
    
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.save_hyperparameters()
        
        # Feature extraction backbone
        self.backbone = self._create_backbone()
        
        # Task-specific heads
        self.phase_head = PhaseClassificationHead(
            self.config.hidden_dims[-1], 
            num_phases=5, 
            config=self.config
        )
        
        self.property_head = PropertyRegressionHead(
            self.config.hidden_dims[-1],
            num_properties=8,
            config=self.config
        )
        
        # Physics constraint validator
        self.physics_validator = nn.ModuleDict({
            "nucleation_constraints": PhysicsConstraintLayer(64, "positive"),
            "thermodynamic_constraints": PhysicsConstraintLayer(64, "bounded")
        })
        
        # Metrics
        self._setup_metrics()
        
        # Loss weights
        self.phase_weight = config.phase_weight if config else 1.0
        self.property_weight = config.property_weight if config else 1.0
        self.physics_weight = config.physics_weight if config else 0.2
        
    def _create_backbone(self) -> nn.Module:
        """Create feature extraction backbone"""
        layers = []
        input_dim = self.config.input_dim
        
        for hidden_dim in self.config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU() if self.config.activation == "relu" else nn.GELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(self.config.dropout_rate)
            ])
            input_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _setup_metrics(self):
        """Setup training and validation metrics"""
        # Phase classification metrics
        self.train_phase_acc = Accuracy(task="multiclass", num_classes=5)
        self.val_phase_acc = Accuracy(task="multiclass", num_classes=5)
        self.train_phase_f1 = F1Score(task="multiclass", num_classes=5, average="macro")
        self.val_phase_f1 = F1Score(task="multiclass", num_classes=5, average="macro")
        
        # Property regression metrics
        self.train_prop_mae = MeanAbsoluteError()
        self.val_prop_mae = MeanAbsoluteError()
        self.train_prop_mse = MeanSquaredError()
        self.val_prop_mse = MeanSquaredError()
    
    def forward(self, x: torch.Tensor, training: bool = True) -> Dict[str, Any]:
        """Forward pass through the complete network"""
        # Feature extraction
        backbone_features = self.backbone(x)
        
        # Phase prediction
        phase_output = self.phase_head(backbone_features, training)
        
        # Property prediction
        property_output = self.property_head(backbone_features, training)
        
        # Physics constraints validation
        physics_features = {
            name: constraint(backbone_features) 
            for name, constraint in self.physics_validator.items()
        }
        
        return {
            "phase": phase_output,
            "properties": property_output,
            "physics": physics_features,
            "backbone_features": backbone_features
        }
    
    def compute_loss(self, outputs: Dict[str, Any], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss with uncertainty weighting"""
        losses = {}
        
        # Phase classification loss
        phase_logits = outputs["phase"]["logits"]
        phase_targets = batch["phase_labels"]
        phase_uncertainty = outputs["phase"]["uncertainty"]
        
        # Cross-entropy loss with uncertainty weighting
        ce_loss = F.cross_entropy(phase_logits, phase_targets, reduction="none")
        phase_loss = torch.mean(ce_loss / (2 * phase_uncertainty.var(dim=-1)) + 0.5 * torch.log(phase_uncertainty.var(dim=-1)))
        losses["phase_loss"] = phase_loss
        
        # Property regression losses
        property_losses = []
        for prop_name, prop_output in outputs["properties"].items():
            if prop_name in batch["properties"]:
                prop_targets = batch["properties"][prop_name]
                prop_mean = prop_output["mean"]
                prop_std = prop_output["std"]
                
                # Gaussian negative log-likelihood with uncertainty
                mse_loss = F.mse_loss(prop_mean, prop_targets, reduction="none")
                prop_loss = torch.mean(mse_loss / (2 * prop_std**2) + torch.log(prop_std))
                property_losses.append(prop_loss)
                losses[f"{prop_name}_loss"] = prop_loss
        
        property_loss = torch.stack(property_losses).mean() if property_losses else torch.tensor(0.0, device=self.device)
        losses["property_loss"] = property_loss
        
        # Physics constraint loss
        physics_loss = torch.tensor(0.0, device=self.device)
        if "physics_constraints" in batch:
            for constraint_name, constraint_values in batch["physics_constraints"].items():
                if constraint_name in outputs["physics"]:
                    physics_pred = outputs["physics"][constraint_name]
                    physics_loss += F.mse_loss(physics_pred.mean(dim=-1), constraint_values)
        losses["physics_loss"] = physics_loss
        
        # Total weighted loss
        total_loss = (
            self.phase_weight * phase_loss + 
            self.property_weight * property_loss + 
            self.physics_weight * physics_loss
        )
        losses["total_loss"] = total_loss
        
        return losses
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        outputs = self(batch["features"], training=True)
        losses = self.compute_loss(outputs, batch)
        
        # Log training metrics
        self.log("train/total_loss", losses["total_loss"], prog_bar=True)
        self.log("train/phase_loss", losses["phase_loss"])
        self.log("train/property_loss", losses["property_loss"])
        self.log("train/physics_loss", losses["physics_loss"])
        
        # Phase classification metrics
        phase_preds = outputs["phase"]["probabilities"].argmax(dim=-1)
        self.train_phase_acc(phase_preds, batch["phase_labels"])
        self.train_phase_f1(phase_preds, batch["phase_labels"])
        self.log("train/phase_acc", self.train_phase_acc, prog_bar=True)
        self.log("train/phase_f1", self.train_phase_f1)
        
        return losses["total_loss"]
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step"""
        outputs = self(batch["features"], training=False)
        losses = self.compute_loss(outputs, batch)
        
        # Log validation metrics
        self.log("val/total_loss", losses["total_loss"], prog_bar=True)
        self.log("val/phase_loss", losses["phase_loss"])
        self.log("val/property_loss", losses["property_loss"])
        self.log("val/physics_loss", losses["physics_loss"])
        
        # Phase classification metrics
        phase_preds = outputs["phase"]["probabilities"].argmax(dim=-1)
        self.val_phase_acc(phase_preds, batch["phase_labels"])
        self.val_phase_f1(phase_preds, batch["phase_labels"])
        self.log("val/phase_acc", self.val_phase_acc, prog_bar=True)
        self.log("val/phase_f1", self.val_phase_f1)
        
        return losses
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Dict[str, Any]:
        """Prediction with uncertainty quantification via Monte Carlo sampling"""
        self.eval()
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                output = self(x, training=True)  # Keep dropout active for MC sampling
                predictions.append(output)
        
        # Aggregate predictions
        phase_probs = torch.stack([p["phase"]["probabilities"] for p in predictions])
        phase_mean = phase_probs.mean(dim=0)
        phase_std = phase_probs.std(dim=0)
        
        # Property aggregation
        property_means = {}
        property_stds = {}
        for prop_name in predictions[0]["properties"].keys():
            prop_values = torch.stack([p["properties"][prop_name]["mean"] for p in predictions])
            property_means[prop_name] = prop_values.mean(dim=0)
            property_stds[prop_name] = prop_values.std(dim=0)
        
        return {
            "phase": {
                "probabilities": phase_mean,
                "uncertainty": phase_std
            },
            "properties": {
                prop_name: {
                    "mean": property_means[prop_name],
                    "uncertainty": property_stds[prop_name]
                }
                for prop_name in property_means.keys()
            }
        }
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        if self.config.lr_scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode="min", 
                factor=0.5, 
                patience=10,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/total_loss"
                }
            }
        elif self.config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=100,
                eta_min=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        else:
            return optimizer


class EnsembleModel(nn.Module):
    """Ensemble of multiple neural networks for improved predictions"""
    
    def __init__(self, models: List[PhysicsInformedNeuralNetwork]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward pass through ensemble"""
        outputs = [model(x, training=False) for model in self.models]
        
        # Aggregate phase predictions
        phase_probs = torch.stack([out["phase"]["probabilities"] for out in outputs])
        ensemble_phase_probs = phase_probs.mean(dim=0)
        phase_uncertainty = phase_probs.std(dim=0)
        
        # Aggregate property predictions
        ensemble_properties = {}
        for prop_name in outputs[0]["properties"].keys():
            prop_means = torch.stack([out["properties"][prop_name]["mean"] for out in outputs])
            prop_stds = torch.stack([out["properties"][prop_name]["std"] for out in outputs])
            
            ensemble_properties[prop_name] = {
                "mean": prop_means.mean(dim=0),
                "aleatoric_uncertainty": prop_stds.mean(dim=0),
                "epistemic_uncertainty": prop_means.std(dim=0),
                "total_uncertainty": torch.sqrt(prop_stds.mean(dim=0)**2 + prop_means.std(dim=0)**2)
            }
        
        return {
            "phase": {
                "probabilities": ensemble_phase_probs,
                "uncertainty": phase_uncertainty
            },
            "properties": ensemble_properties
        }


def create_model(config: Dict[str, Any] = None) -> PhysicsInformedNeuralNetwork:
    """Factory function to create a configured model"""
    model_config = ModelConfig(**config) if config else ModelConfig()
    return PhysicsInformedNeuralNetwork(model_config)


def load_pretrained_model(checkpoint_path: str) -> PhysicsInformedNeuralNetwork:
    """Load a pretrained model from checkpoint"""
    model = PhysicsInformedNeuralNetwork.load_from_checkpoint(checkpoint_path)
    return model
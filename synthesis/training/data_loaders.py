#!/usr/bin/env python3
"""
PyTorch Data Loaders for CsPbBr₃ Digital Twin
Advanced data handling with stratified sampling and physics-informed augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
import pickle
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SynthesisParameters:
    """Synthesis parameters for a single experiment"""
    cs_br_concentration: float
    pb_br2_concentration: float
    temperature: float
    solvent_type: int  # Encoded as integer
    oa_concentration: float = 0.0
    oam_concentration: float = 0.0
    reaction_time: float = 10.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for tensor creation"""
        return {
            'cs_br_concentration': self.cs_br_concentration,
            'pb_br2_concentration': self.pb_br2_concentration,
            'temperature': self.temperature,
            'solvent_type': float(self.solvent_type),
            'oa_concentration': self.oa_concentration,
            'oam_concentration': self.oam_concentration,
            'reaction_time': self.reaction_time
        }


@dataclass
class MaterialProperties:
    """Material properties for a synthesis outcome"""
    bandgap: float
    plqy: float
    emission_peak: float
    emission_fwhm: float
    particle_size: float
    size_distribution_width: float
    lifetime: float
    stability_score: float
    phase_purity: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for tensor creation"""
        return {
            'bandgap': self.bandgap,
            'plqy': self.plqy,
            'emission_peak': self.emission_peak,
            'emission_fwhm': self.emission_fwhm,
            'particle_size': self.particle_size,
            'size_distribution_width': self.size_distribution_width,
            'lifetime': self.lifetime,
            'stability_score': self.stability_score,
            'phase_purity': self.phase_purity
        }


class SynthesisDataset(Dataset):
    """PyTorch Dataset for CsPbBr₃ synthesis data"""
    
    def __init__(self, 
                 synthesis_params: List[SynthesisParameters],
                 phase_labels: List[int],
                 properties: List[MaterialProperties],
                 feature_engineer: Optional[Callable] = None,
                 augment_data: bool = False,
                 cache_features: bool = True):
        """
        Initialize synthesis dataset
        
        Args:
            synthesis_params: List of synthesis parameters
            phase_labels: List of phase labels (0-4)
            properties: List of material properties
            feature_engineer: Feature engineering function
            augment_data: Whether to apply data augmentation
            cache_features: Whether to cache engineered features
        """
        self.synthesis_params = synthesis_params
        self.phase_labels = phase_labels
        self.properties = properties
        self.feature_engineer = feature_engineer
        self.augment_data = augment_data
        self.cache_features = cache_features
        
        # Validation
        assert len(synthesis_params) == len(phase_labels) == len(properties)
        
        # Feature cache
        self._feature_cache = {} if cache_features else None
        
        # Data augmentation parameters
        self.augment_noise_std = 0.05  # 5% noise
        self.augment_prob = 0.3  # 30% chance of augmentation
        
        logger.info(f"Initialized dataset with {len(self)} samples")
        self._log_dataset_statistics()
    
    def __len__(self) -> int:
        return len(self.synthesis_params)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        # Check cache first
        if self.cache_features and idx in self._feature_cache:
            return self._feature_cache[idx]
        
        # Get raw data
        params = self.synthesis_params[idx]
        phase_label = self.phase_labels[idx]
        props = self.properties[idx]
        
        # Apply data augmentation if enabled
        if self.augment_data and np.random.random() < self.augment_prob:
            params = self._augment_parameters(params)
            props = self._augment_properties(props)
        
        # Convert to tensors
        param_dict = params.to_dict()
        param_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in param_dict.items()}
        
        # Feature engineering
        if self.feature_engineer is not None:
            features = self.feature_engineer(param_tensors)
        else:
            features = torch.tensor(list(param_dict.values()), dtype=torch.float32)
        
        # Prepare sample
        sample = {
            'features': features,
            'phase_labels': torch.tensor(phase_label, dtype=torch.long),
            'properties': {k: torch.tensor(v, dtype=torch.float32) for k, v in props.to_dict().items()},
            'raw_params': param_tensors
        }
        
        # Cache if enabled
        if self.cache_features:
            self._feature_cache[idx] = sample
        
        return sample
    
    def _augment_parameters(self, params: SynthesisParameters) -> SynthesisParameters:
        """Apply data augmentation to synthesis parameters"""
        # Add small amount of noise to continuous parameters
        augmented = SynthesisParameters(
            cs_br_concentration=max(0.1, params.cs_br_concentration * (1 + np.random.normal(0, self.augment_noise_std))),
            pb_br2_concentration=max(0.1, params.pb_br2_concentration * (1 + np.random.normal(0, self.augment_noise_std))),
            temperature=max(60, min(300, params.temperature + np.random.normal(0, 5))),  # ±5°C
            solvent_type=params.solvent_type,  # Don't augment categorical
            oa_concentration=max(0, params.oa_concentration * (1 + np.random.normal(0, self.augment_noise_std))),
            oam_concentration=max(0, params.oam_concentration * (1 + np.random.normal(0, self.augment_noise_std))),
            reaction_time=max(1, params.reaction_time * (1 + np.random.normal(0, self.augment_noise_std)))
        )
        return augmented
    
    def _augment_properties(self, props: MaterialProperties) -> MaterialProperties:
        """Apply data augmentation to material properties"""
        # Add noise to properties (smaller for more sensitive properties)
        augmented = MaterialProperties(
            bandgap=max(0.5, props.bandgap * (1 + np.random.normal(0, 0.02))),  # ±2%
            plqy=max(0, min(1, props.plqy * (1 + np.random.normal(0, 0.05)))),  # ±5%
            emission_peak=max(300, props.emission_peak * (1 + np.random.normal(0, 0.01))),  # ±1%
            emission_fwhm=max(5, props.emission_fwhm * (1 + np.random.normal(0, 0.1))),  # ±10%
            particle_size=max(1, props.particle_size * (1 + np.random.normal(0, 0.1))),  # ±10%
            size_distribution_width=max(0.1, props.size_distribution_width * (1 + np.random.normal(0, 0.1))),
            lifetime=max(0.1, props.lifetime * (1 + np.random.normal(0, 0.1))),
            stability_score=max(0, min(1, props.stability_score * (1 + np.random.normal(0, 0.05)))),
            phase_purity=max(0, min(1, props.phase_purity * (1 + np.random.normal(0, 0.02))))
        )
        return augmented
    
    def _log_dataset_statistics(self):
        """Log dataset statistics"""
        phase_counts = Counter(self.phase_labels)
        logger.info(f"Phase distribution: {dict(phase_counts)}")
        
        # Property statistics
        bandgaps = [p.bandgap for p in self.properties]
        plqys = [p.plqy for p in self.properties]
        sizes = [p.particle_size for p in self.properties]
        
        logger.info(f"Bandgap range: {min(bandgaps):.2f} - {max(bandgaps):.2f} eV")
        logger.info(f"PLQY range: {min(plqys):.2f} - {max(plqys):.2f}")
        logger.info(f"Size range: {min(sizes):.1f} - {max(sizes):.1f} nm")
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        phase_counts = Counter(self.phase_labels)
        total_samples = len(self.phase_labels)
        num_classes = len(phase_counts)
        
        weights = torch.zeros(5)  # 5 possible phases
        for phase, count in phase_counts.items():
            weights[phase] = total_samples / (num_classes * count)
        
        return weights
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for WeightedRandomSampler"""
        class_weights = self.get_class_weights()
        sample_weights = torch.zeros(len(self))
        
        for i, phase in enumerate(self.phase_labels):
            sample_weights[i] = class_weights[phase]
        
        return sample_weights
    
    def clear_cache(self):
        """Clear feature cache to free memory"""
        if self._feature_cache is not None:
            self._feature_cache.clear()
            logger.info("Feature cache cleared")


class StratifiedSynthesisDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule with stratified sampling"""
    
    def __init__(self,
                 data_path: Optional[str] = None,
                 synthesis_params: Optional[List[SynthesisParameters]] = None,
                 phase_labels: Optional[List[int]] = None,
                 properties: Optional[List[MaterialProperties]] = None,
                 feature_engineer: Optional[Callable] = None,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 stratify: bool = True,
                 augment_train: bool = True,
                 cache_features: bool = True,
                 random_state: int = 42):
        """
        Initialize data module
        
        Args:
            data_path: Path to saved dataset
            synthesis_params: List of synthesis parameters (if not loading from file)
            phase_labels: List of phase labels
            properties: List of material properties
            feature_engineer: Feature engineering function
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            train_val_test_split: Split ratios for train/val/test
            stratify: Whether to use stratified sampling
            augment_train: Whether to augment training data
            cache_features: Whether to cache engineered features
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.data_path = data_path
        self.synthesis_params = synthesis_params
        self.phase_labels = phase_labels
        self.properties = properties
        self.feature_engineer = feature_engineer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.stratify = stratify
        self.augment_train = augment_train
        self.cache_features = cache_features
        self.random_state = random_state
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.full_dataset = None
        
    def prepare_data(self):
        """Download/prepare data (called once)"""
        if self.data_path and Path(self.data_path).exists():
            logger.info(f"Data found at {self.data_path}")
        elif self.synthesis_params is None:
            logger.warning("No data path provided and no data in memory. "
                          "Will need to generate synthetic data.")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage"""
        # Load or use provided data
        if self.data_path and Path(self.data_path).exists():
            self._load_data_from_file()
        elif self.synthesis_params is not None:
            # Data already provided
            pass
        else:
            # Generate synthetic data as fallback
            self._generate_synthetic_data()
        
        # Create full dataset
        self.full_dataset = SynthesisDataset(
            self.synthesis_params,
            self.phase_labels,
            self.properties,
            feature_engineer=self.feature_engineer,
            augment_data=False,  # No augmentation for splits
            cache_features=self.cache_features
        )
        
        # Create train/val/test splits
        if stage == "fit" or stage is None:
            train_idx, val_idx, test_idx = self._create_stratified_splits()
            
            # Training dataset with augmentation
            train_params = [self.synthesis_params[i] for i in train_idx]
            train_labels = [self.phase_labels[i] for i in train_idx]
            train_props = [self.properties[i] for i in train_idx]
            
            self.train_dataset = SynthesisDataset(
                train_params, train_labels, train_props,
                feature_engineer=self.feature_engineer,
                augment_data=self.augment_train,
                cache_features=self.cache_features
            )
            
            # Validation dataset without augmentation
            val_params = [self.synthesis_params[i] for i in val_idx]
            val_labels = [self.phase_labels[i] for i in val_idx]
            val_props = [self.properties[i] for i in val_idx]
            
            self.val_dataset = SynthesisDataset(
                val_params, val_labels, val_props,
                feature_engineer=self.feature_engineer,
                augment_data=False,
                cache_features=self.cache_features
            )
        
        if stage == "test" or stage is None:
            if not hasattr(self, 'test_idx'):
                _, _, test_idx = self._create_stratified_splits()
            
            test_params = [self.synthesis_params[i] for i in test_idx]
            test_labels = [self.phase_labels[i] for i in test_idx]
            test_props = [self.properties[i] for i in test_idx]
            
            self.test_dataset = SynthesisDataset(
                test_params, test_labels, test_props,
                feature_engineer=self.feature_engineer,
                augment_data=False,
                cache_features=self.cache_features
            )
    
    def _create_stratified_splits(self) -> Tuple[List[int], List[int], List[int]]:
        """Create stratified train/val/test splits"""
        indices = np.arange(len(self.synthesis_params))
        
        if self.stratify:
            # First split: train+val vs test
            train_val_idx, test_idx = train_test_split(
                indices, 
                test_size=self.train_val_test_split[2],
                stratify=self.phase_labels,
                random_state=self.random_state
            )
            
            # Second split: train vs val
            train_labels = [self.phase_labels[i] for i in train_val_idx]
            val_size = self.train_val_test_split[1] / (self.train_val_test_split[0] + self.train_val_test_split[1])
            
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size,
                stratify=train_labels,
                random_state=self.random_state
            )
        else:
            # Simple random split
            train_size = int(len(indices) * self.train_val_test_split[0])
            val_size = int(len(indices) * self.train_val_test_split[1])
            
            np.random.seed(self.random_state)
            shuffled_idx = np.random.permutation(indices)
            
            train_idx = shuffled_idx[:train_size]
            val_idx = shuffled_idx[train_size:train_size + val_size]
            test_idx = shuffled_idx[train_size + val_size:]
        
        logger.info(f"Data splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader"""
        # Use weighted sampling for imbalanced classes
        sample_weights = self.train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction data loader"""
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching"""
        collated = {
            'features': torch.stack([item['features'] for item in batch]),
            'phase_labels': torch.stack([item['phase_labels'] for item in batch]),
            'properties': {},
            'raw_params': {}
        }
        
        # Collate properties
        for prop_name in batch[0]['properties'].keys():
            collated['properties'][prop_name] = torch.stack([item['properties'][prop_name] for item in batch])
        
        # Collate raw parameters
        for param_name in batch[0]['raw_params'].keys():
            collated['raw_params'][param_name] = torch.stack([item['raw_params'][param_name] for item in batch])
        
        return collated
    
    def _load_data_from_file(self):
        """Load data from pickle file"""
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.synthesis_params = data['synthesis_params']
        self.phase_labels = data['phase_labels']
        self.properties = data['properties']
        
        logger.info(f"Loaded {len(self.synthesis_params)} samples from {self.data_path}")
    
    def _generate_synthetic_data(self, num_samples: int = 1000):
        """Generate synthetic data for testing"""
        logger.warning("Generating synthetic data for testing purposes")
        
        np.random.seed(self.random_state)
        
        self.synthesis_params = []
        self.phase_labels = []
        self.properties = []
        
        for _ in range(num_samples):
            # Random synthesis parameters
            params = SynthesisParameters(
                cs_br_concentration=np.random.uniform(0.5, 3.0),
                pb_br2_concentration=np.random.uniform(0.5, 2.0),
                temperature=np.random.uniform(80, 200),
                solvent_type=np.random.randint(0, 5),
                oa_concentration=np.random.uniform(0, 1.0),
                oam_concentration=np.random.uniform(0, 1.0),
                reaction_time=np.random.uniform(1, 60)
            )
            
            # Random phase (biased towards CsPbBr3)
            phase_probs = [0.6, 0.15, 0.15, 0.08, 0.02]  # Realistic distribution
            phase = np.random.choice(5, p=phase_probs)
            
            # Random properties (phase-dependent)
            if phase == 0:  # CsPbBr3
                bandgap = np.random.normal(2.36, 0.1)
                plqy = np.random.beta(8, 2)  # Skewed toward high PLQY
                emission = np.random.normal(520, 10)
            elif phase == 1:  # Cs4PbBr6
                bandgap = np.random.normal(3.0, 0.15)
                plqy = np.random.beta(2, 8)  # Lower PLQY
                emission = np.random.normal(410, 15)
            else:  # Other phases
                bandgap = np.random.uniform(2.0, 3.5)
                plqy = np.random.uniform(0.1, 0.8)
                emission = np.random.uniform(400, 600)
            
            props = MaterialProperties(
                bandgap=max(0.5, bandgap),
                plqy=max(0, min(1, plqy)),
                emission_peak=max(300, emission),
                emission_fwhm=np.random.uniform(15, 50),
                particle_size=np.random.lognormal(2.5, 0.5),  # Log-normal distribution
                size_distribution_width=np.random.uniform(0.2, 0.8),
                lifetime=np.random.uniform(1, 50),
                stability_score=np.random.beta(3, 2),
                phase_purity=np.random.beta(5, 2)
            )
            
            self.synthesis_params.append(params)
            self.phase_labels.append(phase)
            self.properties.append(props)
        
        logger.info(f"Generated {num_samples} synthetic samples")
    
    def save_data(self, path: str):
        """Save current data to file"""
        data = {
            'synthesis_params': self.synthesis_params,
            'phase_labels': self.phase_labels,
            'properties': self.properties
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved dataset to {path}")


def create_datamodule(config: Dict[str, Any]) -> StratifiedSynthesisDataModule:
    """Factory function to create configured data module"""
    return StratifiedSynthesisDataModule(**config)
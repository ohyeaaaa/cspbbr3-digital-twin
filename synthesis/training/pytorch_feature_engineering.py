#!/usr/bin/env python3
"""
Physics-Informed Feature Engineering for CsPbBr₃ Digital Twin
Advanced feature extraction combining synthesis parameters with physics calculations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class SolventType(str, Enum):
    """Solvent types with physical properties"""
    DMSO = "DMSO"
    DMF = "DMF"
    WATER = "water"
    TOLUENE = "toluene"
    OCTADECENE = "octadecene"


@dataclass
class SolventProperties:
    """Physical properties of solvents"""
    dielectric_constant: float
    viscosity: float  # mPa·s
    boiling_point: float  # °C
    dipole_moment: float  # Debye
    donor_number: float  # Lewis basicity
    acceptor_number: float  # Lewis acidity
    
    @classmethod
    def get_properties(cls, solvent: SolventType) -> 'SolventProperties':
        """Get physical properties for a given solvent"""
        properties = {
            SolventType.DMSO: cls(47.2, 1.99, 189.0, 3.96, 29.8, 19.3),
            SolventType.DMF: cls(38.3, 0.80, 153.0, 3.82, 26.6, 16.0),
            SolventType.WATER: cls(80.1, 1.00, 100.0, 1.85, 18.0, 54.8),
            SolventType.TOLUENE: cls(2.38, 0.56, 110.6, 0.36, 0.1, 8.2),
            SolventType.OCTADECENE: cls(2.05, 2.84, 315.0, 0.0, 0.0, 0.0)
        }
        return properties[solvent]


@dataclass
class PhysicsConstants:
    """Physical constants for calculations"""
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    N_A = 6.02214076e23  # Avogadro number
    R = 8.314462618  # Gas constant (J/mol·K)
    h = 6.62607015e-34  # Planck constant (J·s)
    c = 299792458  # Speed of light (m/s)
    
    # Material-specific constants
    surface_energy_cspbbr3 = 0.134  # J/m² (literature value)
    surface_energy_cs4pbbr6 = 0.145  # J/m²
    surface_energy_cspb2br5 = 0.128  # J/m²
    
    # Diffusion coefficients in different solvents (m²/s)
    diffusion_coeff = {
        SolventType.DMSO: 1.2e-10,
        SolventType.DMF: 0.8e-10,
        SolventType.WATER: 2.1e-9,
        SolventType.TOLUENE: 0.5e-10,
        SolventType.OCTADECENE: 0.3e-10
    }


class PhysicsCalculator:
    """Core physics calculations for feature engineering"""
    
    def __init__(self):
        self.constants = PhysicsConstants()
    
    def supersaturation_ratio(self, cs_conc: float, pb_conc: float, br_conc: float, 
                             temperature: float, solvent: SolventType) -> float:
        """Calculate supersaturation ratio for CsPbBr₃ formation"""
        # Solubility product approximation (temperature and solvent dependent)
        solvent_props = SolventProperties.get_properties(solvent)
        
        # Temperature dependence (van't Hoff equation approximation)
        T_ref = 298.15  # Reference temperature (K)
        T = temperature + 273.15  # Convert to Kelvin
        
        # Base Ksp values (estimated from literature)
        ksp_base = {
            SolventType.DMSO: 1e-15,
            SolventType.DMF: 5e-16,
            SolventType.WATER: 1e-18,
            SolventType.TOLUENE: 1e-20,
            SolventType.OCTADECENE: 1e-22
        }
        
        # Temperature correction
        delta_h_sol = -50000  # Enthalpy of dissolution (J/mol, estimated)
        ksp = ksp_base[solvent] * np.exp(delta_h_sol / self.constants.R * (1/T_ref - 1/T))
        
        # Activity coefficients (simplified Debye-Hückel)
        ionic_strength = 0.5 * (cs_conc + pb_conc + 3 * br_conc)
        gamma = np.exp(-0.5 * np.sqrt(ionic_strength) / (1 + np.sqrt(ionic_strength)))
        
        # Ion activity product
        iap = (cs_conc * gamma) * (pb_conc * gamma) * (br_conc * gamma)**3
        
        return iap / ksp if ksp > 0 else 0.0
    
    def nucleation_rate(self, supersaturation: float, surface_energy: float, 
                       temperature: float, molecular_volume: float) -> float:
        """Calculate nucleation rate using classical nucleation theory"""
        if supersaturation <= 1.0:
            return 0.0
        
        T = temperature + 273.15  # Convert to Kelvin
        
        # Critical nucleus size
        r_critical = 2 * surface_energy * molecular_volume / (self.constants.k_B * T * np.log(supersaturation))
        
        # Critical nucleation energy
        delta_g_critical = (16 * np.pi * surface_energy**3 * molecular_volume**2) / (3 * (self.constants.k_B * T * np.log(supersaturation))**2)
        
        # Pre-exponential factor (simplified)
        A = 1e20  # molecules/(m³·s), typical order of magnitude
        
        # Nucleation rate
        J = A * np.exp(-delta_g_critical / (self.constants.k_B * T))
        
        return float(J)
    
    def growth_rate(self, supersaturation: float, temperature: float, 
                   solvent: SolventType, ligand_coverage: float = 0.0) -> float:
        """Calculate crystal growth rate"""
        if supersaturation <= 1.0:
            return 0.0
        
        T = temperature + 273.15
        
        # Diffusion coefficient
        D = self.constants.diffusion_coeff[solvent]
        
        # Temperature dependence (Arrhenius)
        Ea = 45000  # Activation energy (J/mol, estimated)
        D_temp = D * np.exp(-Ea / (self.constants.R * T))
        
        # Boundary layer thickness (estimated)
        delta = 1e-6  # m
        
        # Surface attachment coefficient (reduced by ligand coverage)
        beta = (1 - ligand_coverage) * D_temp / delta
        
        # Growth rate (Burton-Cabrera-Frank model)
        driving_force = supersaturation - 1.0
        R_growth = beta * driving_force
        
        return float(R_growth)
    
    def ligand_binding_energy(self, ligand_type: str, surface_type: str = "100") -> float:
        """Calculate ligand binding energy (simplified)"""
        # Binding energies in eV (estimated from DFT studies)
        binding_energies = {
            ("oleic_acid", "100"): 0.8,
            ("oleic_acid", "110"): 0.9,
            ("oleylamine", "100"): 0.7,
            ("oleylamine", "110"): 0.8,
            ("none", "100"): 0.0,
            ("none", "110"): 0.0
        }
        
        return binding_energies.get((ligand_type, surface_type), 0.5)
    
    def thermal_energy(self, temperature: float) -> float:
        """Calculate thermal energy at given temperature"""
        T = temperature + 273.15
        return self.constants.k_B * T  # Joules
    
    def diffusion_length(self, temperature: float, time: float, solvent: SolventType) -> float:
        """Calculate characteristic diffusion length"""
        T = temperature + 273.15
        D = self.constants.diffusion_coeff[solvent]
        
        # Temperature dependence
        Ea = 20000  # Lower activation energy for diffusion (J/mol)
        D_temp = D * np.exp(-Ea / (self.constants.R * T))
        
        return np.sqrt(2 * D_temp * time * 60)  # time in minutes -> seconds


class PhysicsInformedFeatureEngineer(nn.Module):
    """
    Physics-informed feature engineering for CsPbBr₃ synthesis prediction
    Combines experimental parameters with physics-based calculations
    """
    
    def __init__(self, normalize_features: bool = True, include_interactions: bool = True):
        super().__init__()
        self.normalize_features = normalize_features
        self.include_interactions = include_interactions
        self.physics_calc = PhysicsCalculator()
        
        # Feature normalization parameters (learnable)
        if normalize_features:
            self.register_buffer('feature_means', torch.zeros(100))  # Will be updated
            self.register_buffer('feature_stds', torch.ones(100))
            self.normalization_initialized = False
        
        # Learnable physics weights
        self.physics_weights = nn.Parameter(torch.ones(30))  # 30 physics features
        self.interaction_weights = nn.Parameter(torch.ones(25)) if include_interactions else None
    
    def extract_base_features(self, synthesis_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract base synthesis parameters as features"""
        features = []
        
        # Concentration features
        features.append(synthesis_params['cs_br_concentration'])
        features.append(synthesis_params['pb_br2_concentration'])
        features.append(synthesis_params['oa_concentration'])
        features.append(synthesis_params['oam_concentration'])
        
        # Derived concentration features
        total_halide = synthesis_params['cs_br_concentration'] + 2 * synthesis_params['pb_br2_concentration']
        features.append(total_halide)
        
        cs_pb_ratio = synthesis_params['cs_br_concentration'] / (synthesis_params['pb_br2_concentration'] + 1e-8)
        features.append(cs_pb_ratio)
        
        ligand_total = synthesis_params['oa_concentration'] + synthesis_params['oam_concentration']
        features.append(ligand_total)
        
        # Temperature features
        features.append(synthesis_params['temperature'])
        features.append(synthesis_params['temperature'] ** 2)  # Non-linear temperature effects
        
        # Time features
        features.append(synthesis_params['reaction_time'])
        features.append(torch.log(synthesis_params['reaction_time'] + 1))  # Log time
        
        # Solvent features (one-hot encoded)
        solvent_features = self._encode_solvent(synthesis_params['solvent_type'])
        features.extend(solvent_features)
        
        return torch.stack(features, dim=-1)
    
    def _encode_solvent(self, solvent_type: torch.Tensor) -> List[torch.Tensor]:
        """One-hot encode solvent type with physical properties"""
        batch_size = solvent_type.shape[0]
        device = solvent_type.device
        
        # One-hot encoding for 5 solvents
        solvent_onehot = torch.zeros(batch_size, 5, device=device)
        solvent_onehot.scatter_(1, solvent_type.long().unsqueeze(1), 1)
        
        # Physical properties as features
        properties = torch.zeros(batch_size, 6, device=device)  # 6 physical properties
        
        for i, solvent_idx in enumerate(solvent_type):
            solvent = list(SolventType)[solvent_idx.int()]
            props = SolventProperties.get_properties(solvent)
            properties[i] = torch.tensor([
                props.dielectric_constant / 100,  # Normalized
                props.viscosity / 5,
                props.boiling_point / 400,
                props.dipole_moment / 5,
                props.donor_number / 50,
                props.acceptor_number / 60
            ], device=device)
        
        # Combine one-hot and properties
        combined = torch.cat([solvent_onehot, properties], dim=-1)
        return [combined[:, i] for i in range(combined.shape[1])]
    
    def calculate_physics_features(self, synthesis_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate physics-informed features"""
        batch_size = synthesis_params['cs_br_concentration'].shape[0]
        device = synthesis_params['cs_br_concentration'].device
        
        physics_features = []
        
        for i in range(batch_size):
            # Extract parameters for this sample
            cs_conc = synthesis_params['cs_br_concentration'][i].item()
            pb_conc = synthesis_params['pb_br2_concentration'][i].item()
            br_conc = cs_conc + 2 * pb_conc  # Total bromide
            temp = synthesis_params['temperature'][i].item()
            time = synthesis_params['reaction_time'][i].item()
            solvent_idx = synthesis_params['solvent_type'][i].item()
            solvent = list(SolventType)[int(solvent_idx)]
            oa_conc = synthesis_params['oa_concentration'][i].item()
            oam_conc = synthesis_params['oam_concentration'][i].item()
            
            # Calculate physics features
            sample_features = []
            
            # 1. Supersaturation
            supersaturation = self.physics_calc.supersaturation_ratio(cs_conc, pb_conc, br_conc, temp, solvent)
            sample_features.append(supersaturation)
            sample_features.append(np.log(supersaturation + 1))  # Log supersaturation
            
            # 2. Nucleation rates for different phases
            molecular_volume = 1e-28  # m³, estimated
            for surface_energy in [0.134, 0.145, 0.128]:  # Different phases
                nuc_rate = self.physics_calc.nucleation_rate(supersaturation, surface_energy, temp, molecular_volume)
                sample_features.append(np.log(nuc_rate + 1e-10))  # Log nucleation rate
            
            # 3. Growth rates
            ligand_coverage = (oa_conc + oam_conc) / (cs_conc + pb_conc + 1e-8)  # Simplified
            growth_rate = self.physics_calc.growth_rate(supersaturation, temp, solvent, ligand_coverage)
            sample_features.append(np.log(growth_rate + 1e-10))
            
            # 4. Thermal energy
            thermal_E = self.physics_calc.thermal_energy(temp)
            sample_features.append(thermal_E * 6.242e18)  # Convert to eV
            
            # 5. Diffusion length
            diff_length = self.physics_calc.diffusion_length(temp, time, solvent)
            sample_features.append(np.log(diff_length + 1e-12))
            
            # 6. Ligand binding energies
            oa_binding = self.physics_calc.ligand_binding_energy("oleic_acid") * oa_conc
            oam_binding = self.physics_calc.ligand_binding_energy("oleylamine") * oam_conc
            sample_features.extend([oa_binding, oam_binding])
            
            # 7. Thermodynamic ratios
            cs_pb_ratio = cs_conc / (pb_conc + 1e-8)
            sample_features.append(cs_pb_ratio)
            sample_features.append(1 / (cs_pb_ratio + 1e-8))  # Pb/Cs ratio
            
            # 8. Solvent effects
            solvent_props = SolventProperties.get_properties(solvent)
            sample_features.append(solvent_props.dielectric_constant / temp)  # Dielectric/T
            sample_features.append(solvent_props.viscosity * temp)  # Viscosity*T
            
            # 9. Kinetic factors
            arrhenius_factor = np.exp(-45000 / (8.314 * (temp + 273.15)))  # Activation energy
            sample_features.append(arrhenius_factor)
            
            # 10. Competition factors
            total_ionic_strength = cs_conc + pb_conc + br_conc
            sample_features.append(total_ionic_strength)
            
            # 11. Time-dependent features
            characteristic_time = 10.0  # minutes, estimated
            time_ratio = time / characteristic_time
            sample_features.append(time_ratio)
            sample_features.append(np.exp(-time_ratio))  # Exponential decay
            
            # 12. Size predictions (simplified)
            predicted_size = diff_length * 1e9  # Convert to nm
            sample_features.append(np.log(predicted_size + 1))
            
            # 13. Phase stability indicators
            gibbs_factor = supersaturation * thermal_E * 6.242e18  # eV
            sample_features.append(gibbs_factor)
            
            # 14. Additional ratios
            halide_metal_ratio = br_conc / (cs_conc + pb_conc + 1e-8)
            sample_features.append(halide_metal_ratio)
            
            # 15. Surface area effects
            estimated_surface_area = predicted_size ** 2  # nm²
            sample_features.append(np.log(estimated_surface_area + 1))
            
            physics_features.append(sample_features)
        
        # Convert to tensor (30 physics features)
        physics_tensor = torch.tensor(physics_features, dtype=torch.float32, device=device)
        return physics_tensor
    
    def calculate_interaction_features(self, base_features: torch.Tensor, 
                                     physics_features: torch.Tensor) -> torch.Tensor:
        """Calculate interaction features between base and physics features"""
        if not self.include_interactions:
            return torch.empty(base_features.shape[0], 0, device=base_features.device)
        
        interactions = []
        
        # Temperature interactions with concentrations
        temp_idx = 7  # Temperature is at index 7 in base features
        temp = base_features[:, temp_idx:temp_idx+1]
        
        interactions.append(base_features[:, 0:1] * temp)  # Cs*T
        interactions.append(base_features[:, 1:1] * temp)  # Pb*T
        interactions.append(base_features[:, 2:1] * temp)  # OA*T
        interactions.append(base_features[:, 3:1] * temp)  # OAm*T
        
        # Concentration cross-interactions
        interactions.append(base_features[:, 0:1] * base_features[:, 1:2])  # Cs*Pb
        interactions.append(base_features[:, 0:1] * base_features[:, 2:3])  # Cs*OA
        interactions.append(base_features[:, 1:2] * base_features[:, 2:3])  # Pb*OA
        
        # Physics-synthesis interactions
        supersaturation = physics_features[:, 0:1]
        nucleation_rate = physics_features[:, 2:3]  # CsPbBr3 nucleation
        
        interactions.append(supersaturation * temp)
        interactions.append(nucleation_rate * base_features[:, 9:10])  # Nucleation * time
        interactions.append(physics_features[:, 5:6] * temp)  # Growth rate * temp
        
        # Solvent-concentration interactions
        for i in range(5):  # 5 solvents in one-hot
            solvent_feature = base_features[:, 11+i:12+i]
            interactions.append(solvent_feature * base_features[:, 0:1])  # Solvent * Cs
            interactions.append(solvent_feature * base_features[:, 1:2])  # Solvent * Pb
        
        # Time interactions
        time_feature = base_features[:, 9:10]
        interactions.append(time_feature * supersaturation)
        interactions.append(time_feature * temp)
        
        # Additional physics interactions
        interactions.append(physics_features[:, 8:9] * physics_features[:, 9:10])  # OA * OAm binding
        interactions.append(physics_features[:, 10:11] * physics_features[:, 11:12])  # Ratios
        
        return torch.cat(interactions, dim=-1)
    
    def forward(self, synthesis_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass: convert synthesis parameters to physics-informed features
        
        Args:
            synthesis_params: Dictionary with synthesis parameters
            
        Returns:
            Tensor of shape (batch_size, num_features) with engineered features
        """
        # Extract base features (20 features)
        base_features = self.extract_base_features(synthesis_params)
        
        # Calculate physics features (30 features)
        physics_features = self.calculate_physics_features(synthesis_params)
        
        # Apply learnable weights to physics features
        weighted_physics = physics_features * self.physics_weights
        
        # Calculate interaction features (25 features)
        interaction_features = self.calculate_interaction_features(base_features, weighted_physics)
        
        # Apply weights to interactions if included
        if self.include_interactions and self.interaction_weights is not None:
            weighted_interactions = interaction_features * self.interaction_weights
        else:
            weighted_interactions = interaction_features
        
        # Combine all features
        all_features = torch.cat([base_features, weighted_physics, weighted_interactions], dim=-1)
        
        # Normalize features if enabled
        if self.normalize_features:
            if not self.normalization_initialized:
                self._initialize_normalization(all_features)
            
            normalized_features = (all_features - self.feature_means) / (self.feature_stds + 1e-8)
            return normalized_features
        
        return all_features
    
    def _initialize_normalization(self, features: torch.Tensor):
        """Initialize normalization parameters"""
        with torch.no_grad():
            self.feature_means.data = features.mean(dim=0)
            self.feature_stds.data = features.std(dim=0)
            self.normalization_initialized = True
    
    def update_normalization(self, features: torch.Tensor):
        """Update normalization parameters with new data (exponential moving average)"""
        if not self.normalization_initialized:
            self._initialize_normalization(features)
            return
        
        alpha = 0.01  # Learning rate for moving average
        with torch.no_grad():
            batch_mean = features.mean(dim=0)
            batch_std = features.std(dim=0)
            
            self.feature_means.data = (1 - alpha) * self.feature_means.data + alpha * batch_mean
            self.feature_stds.data = (1 - alpha) * self.feature_stds.data + alpha * batch_std
    
    def get_feature_names(self) -> List[str]:
        """Get names of all engineered features"""
        names = []
        
        # Base features
        names.extend([
            'cs_br_conc', 'pb_br2_conc', 'oa_conc', 'oam_conc',
            'total_halide', 'cs_pb_ratio', 'total_ligand',
            'temperature', 'temperature_sq', 'reaction_time', 'log_time'
        ])
        
        # Solvent one-hot
        names.extend([f'solvent_{s.value}' for s in SolventType])
        
        # Solvent properties
        names.extend(['dielectric', 'viscosity', 'boiling_point', 'dipole', 'donor_num', 'acceptor_num'])
        
        # Physics features
        names.extend([
            'supersaturation', 'log_supersaturation',
            'nucleation_cspbbr3', 'nucleation_cs4pbbr6', 'nucleation_cspb2br5',
            'growth_rate', 'thermal_energy', 'diffusion_length',
            'oa_binding', 'oam_binding', 'cs_pb_ratio_phys', 'pb_cs_ratio',
            'dielectric_temp', 'viscosity_temp', 'arrhenius', 'ionic_strength',
            'time_ratio', 'time_decay', 'predicted_size', 'gibbs_factor',
            'halide_metal_ratio', 'surface_area'
        ])
        
        # Interaction features (if included)
        if self.include_interactions:
            names.extend([
                'cs_temp', 'pb_temp', 'oa_temp', 'oam_temp',
                'cs_pb', 'cs_oa', 'pb_oa',
                'supersaturation_temp', 'nucleation_time', 'growth_temp'
            ])
            
            # Solvent interactions
            for solvent in SolventType:
                names.extend([f'{solvent.value}_cs', f'{solvent.value}_pb'])
            
            names.extend(['time_supersaturation', 'time_temp', 'binding_interaction', 'ratio_interaction'])
        
        return names


def create_feature_engineer(config: Dict[str, Any] = None) -> PhysicsInformedFeatureEngineer:
    """Factory function to create configured feature engineer"""
    config = config or {}
    return PhysicsInformedFeatureEngineer(
        normalize_features=config.get('normalize_features', True),
        include_interactions=config.get('include_interactions', True)
    )
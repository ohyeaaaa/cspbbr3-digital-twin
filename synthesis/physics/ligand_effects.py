222111111111111111111112#!/usr/bin/env python3
"""
Ligand Effects and Surface Chemistry Models for CsPbBr₃ Digital Twin
Competitive binding, surface passivation, and morphology control
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LigandProperties:
    """Properties of organic ligands"""
    binding_energy: float      # eV
    steric_parameter: float    # Å²
    chain_length: int          # Number of carbons
    polarity: float           # Debye
    pka: Optional[float]      # Acid dissociation constant


class LigandDatabase:
    """Database of ligand properties"""
    
    def __init__(self):
        self.ligands = {
            'oleic_acid': LigandProperties(
                binding_energy=0.85, steric_parameter=25.0, 
                chain_length=18, polarity=1.8, pka=9.85
            ),
            'oleylamine': LigandProperties(
                binding_energy=0.75, steric_parameter=22.0,
                chain_length=18, polarity=1.2, pka=10.65
            ),
            'octylamine': LigandProperties(
                binding_energy=0.65, steric_parameter=15.0,
                chain_length=8, polarity=1.0, pka=10.65
            ),
            'trioctylphosphine': LigandProperties(
                binding_energy=0.95, steric_parameter=35.0,
                chain_length=24, polarity=0.8, pka=None
            ),
            'didodecyldimethylammonium': LigandProperties(
                binding_energy=0.88, steric_parameter=40.0,
                chain_length=24, polarity=3.5, pka=None
            )
        }


class CompetitiveLigandBinding:
    """Model for competitive ligand binding on crystal surfaces"""
    
    def __init__(self):
        self.ligand_db = LigandDatabase()
        self.k_B = 1.380649e-23  # Boltzmann constant
        self.surface_site_density = 1.5e19  # sites/m² for perovskite (100) surface
        
        # Surface-specific binding energies (relative to (100) face)
        self.surface_factors = {
            '100': 1.0,    # Reference
            '110': 1.15,   # Higher energy surface
            '111': 0.85    # Lower energy surface
        }
    
    def calculate_binding_constant(self, 
                                 ligand: str, 
                                 temperature: float,
                                 surface: str = '100',
                                 solvent: str = 'octadecene') -> float:
        """
        Calculate temperature-dependent binding constant
        
        Args:
            ligand: Ligand name
            temperature: Temperature (°C)
            surface: Crystal surface orientation
            solvent: Solvent type
            
        Returns:
            Binding constant (M⁻¹)
        """
        if ligand not in self.ligand_db.ligands:
            return 1e3  # Default moderate binding
        
        T = temperature + 273.15
        ligand_props = self.ligand_db.ligands[ligand]
        
        # Base binding energy corrected for surface
        E_bind = ligand_props.binding_energy * self.surface_factors[surface]  # eV
        
        # Solvent effects (simplified)
        solvent_corrections = {
            'octadecene': 1.0,    # Reference nonpolar
            'toluene': 0.9,       # Slightly polar
            'DMSO': 0.6,          # Polar, competes for binding
            'DMF': 0.65,          # Polar, competes for binding
            'water': 0.3          # Highly polar, strong competition
        }
        
        E_bind *= solvent_corrections.get(solvent, 1.0)
        
        # Convert to Joules and calculate binding constant
        E_bind_J = E_bind * 1.602e-19
        
        # K = exp(E_bind / kT) with reference state correction
        K = 1e6 * np.exp(E_bind_J / (self.k_B * T))  # M⁻¹
        
        return K
    
    def calculate_surface_coverage(self,
                                 ligand_concentrations: Dict[str, float],
                                 temperature: float,
                                 surface: str = '100',
                                 solvent: str = 'octadecene') -> Dict[str, float]:
        """
        Calculate surface coverage using competitive Langmuir adsorption
        
        Args:
            ligand_concentrations: Dict of ligand concentrations (mol/L)
            temperature: Temperature (°C)
            surface: Crystal surface orientation
            solvent: Solvent type
            
        Returns:
            Dictionary of surface coverages (θ, fraction 0-1)
        """
        # Calculate binding constants for all ligands
        binding_constants = {}
        for ligand, conc in ligand_concentrations.items():
            if conc > 0:
                binding_constants[ligand] = self.calculate_binding_constant(
                    ligand, temperature, surface, solvent
                )
        
        # Competitive Langmuir adsorption
        # θᵢ = Kᵢ*Cᵢ / (1 + Σⱼ(Kⱼ*Cⱼ))
        
        denominator = 1.0
        for ligand, conc in ligand_concentrations.items():
            if ligand in binding_constants:
                denominator += binding_constants[ligand] * conc
        
        coverages = {}
        total_coverage = 0.0
        
        for ligand, conc in ligand_concentrations.items():
            if ligand in binding_constants and conc > 0:
                theta = binding_constants[ligand] * conc / denominator
                coverages[ligand] = theta
                total_coverage += theta
            else:
                coverages[ligand] = 0.0
        
        coverages['total'] = min(1.0, total_coverage)  # Physical maximum
        
        return coverages
    
    def calculate_steric_effects(self,
                               coverages: Dict[str, float],
                               crystal_size: float) -> Dict[str, float]:
        """
        Calculate steric effects of ligands on crystal growth
        
        Args:
            coverages: Surface coverages from calculate_surface_coverage
            crystal_size: Crystal size (nm)
            
        Returns:
            Dictionary of steric effects
        """
        # Steric hindrance increases with coverage and ligand size
        total_steric_parameter = 0.0
        
        for ligand, coverage in coverages.items():
            if ligand in self.ligand_db.ligands and coverage > 0:
                props = self.ligand_db.ligands[ligand]
                total_steric_parameter += coverage * props.steric_parameter
        
        # Size-dependent effects (smaller crystals more affected)
        size_factor = 10.0 / (crystal_size + 5.0)  # More hindrance for small crystals
        
        steric_hindrance = min(0.95, total_steric_parameter * size_factor / 100.0)
        
        # Growth rate reduction factor
        growth_reduction = 1.0 - steric_hindrance
        
        # Anisotropy enhancement (ligands prefer certain faces)
        anisotropy_factor = 1.0 + 2.0 * steric_hindrance
        
        return {
            'steric_hindrance': steric_hindrance,
            'growth_reduction': growth_reduction,
            'anisotropy_factor': anisotropy_factor,
            'size_focusing_effect': 1.0 + steric_hindrance  # Better monodispersity
        }


class LigandExchangeKinetics:
    """Model for dynamic ligand exchange during synthesis"""
    
    def __init__(self):
        self.ligand_db = LigandDatabase()
        self.binding_model = CompetitiveLigandBinding()
    
    def calculate_exchange_rate(self,
                              ligand_from: str,
                              ligand_to: str,
                              temperature: float,
                              concentration_ratio: float) -> float:
        """
        Calculate ligand exchange rate constant
        
        Args:
            ligand_from: Initial ligand
            ligand_to: Incoming ligand
            temperature: Temperature (°C)
            concentration_ratio: [ligand_to] / [ligand_from]
            
        Returns:
            Exchange rate constant (s⁻¹)
        """
        T = temperature + 273.15
        
        # Energy barrier for exchange (difference in binding energies)
        if ligand_from in self.ligand_db.ligands and ligand_to in self.ligand_db.ligands:
            E_from = self.ligand_db.ligands[ligand_from].binding_energy
            E_to = self.ligand_db.ligands[ligand_to].binding_energy
            
            # Barrier height (simplified)
            E_barrier = max(E_from, E_to) + 0.2  # eV
            
            # Arrhenius rate
            attempt_freq = 1e12  # s⁻¹, typical molecular vibration
            rate = attempt_freq * np.exp(-E_barrier * 1.602e-19 / (1.380649e-23 * T))
            
            # Concentration dependence
            rate *= concentration_ratio
            
            return rate
        
        return 1e-6  # Default slow exchange


class MorphologyControl:
    """Model for ligand-controlled crystal morphology"""
    
    def __init__(self):
        self.ligand_db = LigandDatabase()
        self.binding_model = CompetitiveLigandBinding()
    
    def predict_aspect_ratio(self,
                           ligand_concentrations: Dict[str, float],
                           temperature: float,
                           crystal_size: float = 10.0) -> Dict[str, Any]:
        """
        Predict crystal aspect ratio based on ligand binding
        
        Args:
            ligand_concentrations: Ligand concentrations (mol/L)
            temperature: Temperature (°C)
            crystal_size: Current crystal size (nm)
            
        Returns:
            Dictionary with morphology predictions
        """
        # Calculate coverages on different crystal faces
        faces = ['100', '110', '111']
        face_coverages = {}
        
        for face in faces:
            coverage = self.binding_model.calculate_surface_coverage(
                ligand_concentrations, temperature, face
            )
            face_coverages[face] = coverage['total']
        
        # Relative growth rates (inversely related to coverage)
        growth_rates = {}
        for face in faces:
            # Higher coverage = slower growth
            growth_rates[face] = 1.0 / (1.0 + 5.0 * face_coverages[face])
        
        # Aspect ratio calculation
        # Assuming cubic → rod transition with preferential (100) capping
        if face_coverages['100'] > face_coverages['110']:
            # (100) faces capped → growth along [001]
            aspect_ratio = 1.0 + 3.0 * (face_coverages['100'] - face_coverages['110'])
            morphology = 'rod'
        elif face_coverages['111'] > 0.7:
            # (111) faces exposed → truncated cube
            aspect_ratio = 0.8
            morphology = 'truncated_cube'
        else:
            # Isotropic growth
            aspect_ratio = 1.0
            morphology = 'cube'
        
        # Size distribution (monodispersity)
        total_coverage = sum(face_coverages.values()) / len(faces)
        polydispersity = max(0.05, 0.3 - 0.25 * total_coverage)  # Lower with more ligands
        
        return {
            'aspect_ratio': max(0.5, min(5.0, aspect_ratio)),
            'morphology': morphology,
            'polydispersity': polydispersity,
            'face_coverages': face_coverages,
            'relative_growth_rates': growth_rates
        }


class PyTorchLigandModel(nn.Module):
    """PyTorch wrapper for ligand effects calculations"""
    
    def __init__(self):
        super().__init__()
        self.binding_model = CompetitiveLigandBinding()
        self.morphology_model = MorphologyControl()
        
        # Learnable parameters
        self.binding_corrections = nn.Parameter(torch.ones(5))  # Per ligand type
        self.morphology_weights = nn.Parameter(torch.ones(3))   # Per crystal face
    
    def forward(self, synthesis_params: torch.Tensor) -> torch.Tensor:
        """
        Calculate ligand effects features
        
        Args:
            synthesis_params: Tensor of shape (batch_size, 7)
                [cs_conc, pb_conc, br_conc, temperature, solvent_idx, oa_conc, oam_conc]
        
        Returns:
            Tensor of shape (batch_size, 8) with ligand features:
                [total_coverage, oa_coverage, oam_coverage, growth_reduction,
                 anisotropy_factor, aspect_ratio, polydispersity, steric_hindrance]
        """
        batch_size = synthesis_params.shape[0]
        device = synthesis_params.device
        
        temperature = synthesis_params[:, 3]
        solvent_idx = synthesis_params[:, 4].long()
        oa_conc = synthesis_params[:, 5]
        oam_conc = synthesis_params[:, 6]
        
        solvent_names = ['DMSO', 'DMF', 'water', 'toluene', 'octadecene']
        
        ligand_features = torch.zeros(batch_size, 8, device=device)
        
        for i in range(batch_size):
            temp_val = temperature[i].item()
            solvent = solvent_names[solvent_idx[i].item()]
            oa_val = oa_conc[i].item()
            oam_val = oam_conc[i].item()
            
            # Ligand concentrations
            ligand_concs = {
                'oleic_acid': oa_val,
                'oleylamine': oam_val
            }
            
            # Calculate surface coverage
            coverages = self.binding_model.calculate_surface_coverage(
                ligand_concs, temp_val, solvent=solvent
            )
            
            # Calculate steric effects
            crystal_size = 10.0  # nm, typical size
            steric_effects = self.binding_model.calculate_steric_effects(
                coverages, crystal_size
            )
            
            # Calculate morphology
            morphology = self.morphology_model.predict_aspect_ratio(
                ligand_concs, temp_val, crystal_size
            )
            
            # Compile features
            features = [
                coverages['total'],
                coverages.get('oleic_acid', 0.0),
                coverages.get('oleylamine', 0.0),
                steric_effects['growth_reduction'],
                steric_effects['anisotropy_factor'],
                morphology['aspect_ratio'],
                morphology['polydispersity'],
                steric_effects['steric_hindrance']
            ]
            
            ligand_features[i] = torch.tensor(features, device=device)
        
        return ligand_features


def create_ligand_model() -> CompetitiveLigandBinding:
    """Factory function"""
    return CompetitiveLigandBinding()


def create_pytorch_ligand_model() -> PyTorchLigandModel:
    """Factory function"""
    return PyTorchLigandModel()
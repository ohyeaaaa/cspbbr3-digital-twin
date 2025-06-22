#!/usr/bin/env python3
"""
Phase Selection and Thermodynamic Stability Models for CsPbBr₃ Digital Twin
Gibbs energy calculations and competitive phase formation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PhaseType(Enum):
    """Perovskite phase types"""
    CSPBBR3_3D = 0
    CS4PBBR6_0D = 1
    CSPB2BR5_2D = 2
    MIXED_PHASES = 3
    FAILED_SYNTHESIS = 4


@dataclass
class ThermodynamicData:
    """Thermodynamic data for perovskite phases"""
    # Formation enthalpies (kJ/mol) - from DFT calculations and experiments
    formation_enthalpies = {
        PhaseType.CSPBBR3_3D: -125.4,
        PhaseType.CS4PBBR6_0D: -132.1,
        PhaseType.CSPB2BR5_2D: -118.7
    }
    
    # Entropies (J/mol·K) - estimated from lattice dynamics
    entropies = {
        PhaseType.CSPBBR3_3D: 184.2,
        PhaseType.CS4PBBR6_0D: 298.5,
        PhaseType.CSPB2BR5_2D: 245.8
    }
    
    # Heat capacities (J/mol·K)
    heat_capacities = {
        PhaseType.CSPBBR3_3D: 142.1,
        PhaseType.CS4PBBR6_0D: 235.7,
        PhaseType.CSPB2BR5_2D: 189.3
    }


class GibbsEnergyCalculator:
    """Calculate Gibbs free energies for phase stability analysis"""
    
    def __init__(self):
        self.thermo_data = ThermodynamicData()
        self.R = 8.314462618  # Gas constant
        self.reference_temp = 298.15  # K
    
    def calculate_gibbs_energy(self, 
                             phase: PhaseType, 
                             temperature: float,
                             pressure: float = 1.0) -> float:
        """
        Calculate Gibbs free energy for a given phase
        
        Args:
            phase: Phase type
            temperature: Temperature (°C)
            pressure: Pressure (bar)
            
        Returns:
            Gibbs free energy (kJ/mol)
        """
        T = temperature + 273.15  # Convert to Kelvin
        
        # Get thermodynamic data
        H_form = self.thermo_data.formation_enthalpies[phase]  # kJ/mol
        S_ref = self.thermo_data.entropies[phase]  # J/mol·K
        Cp = self.thermo_data.heat_capacities[phase]  # J/mol·K
        
        # Temperature correction for enthalpy (assuming constant Cp)
        H_T = H_form + Cp * (T - self.reference_temp) / 1000  # kJ/mol
        
        # Temperature correction for entropy
        S_T = S_ref + Cp * np.log(T / self.reference_temp)  # J/mol·K
        
        # Gibbs free energy
        G = H_T - T * S_T / 1000  # kJ/mol
        
        # Pressure correction (minimal for solids)
        if pressure != 1.0:
            # Assume molar volume of ~0.1 L/mol
            V_molar = 0.1  # L/mol
            G += V_molar * (pressure - 1.0) / 1000  # kJ/mol
        
        return G
    
    def calculate_relative_stability(self, 
                                   temperature: float,
                                   reference_phase: PhaseType = PhaseType.CSPBBR3_3D) -> Dict[PhaseType, float]:
        """Calculate relative Gibbs energies with respect to reference phase"""
        G_ref = self.calculate_gibbs_energy(reference_phase, temperature)
        
        relative_energies = {}
        for phase in [PhaseType.CSPBBR3_3D, PhaseType.CS4PBBR6_0D, PhaseType.CSPB2BR5_2D]:
            G_phase = self.calculate_gibbs_energy(phase, temperature)
            relative_energies[phase] = G_phase - G_ref
        
        return relative_energies


class PhaseSelectionModel:
    """Model for predicting dominant phase based on thermodynamics and kinetics"""
    
    def __init__(self):
        self.gibbs_calc = GibbsEnergyCalculator()
        self.k_B = 1.380649e-23  # Boltzmann constant
        self.R = 8.314462618     # Gas constant
    
    def calculate_phase_probabilities(self,
                                    cs_concentration: float,
                                    pb_concentration: float,
                                    temperature: float,
                                    kinetic_factors: Optional[Dict[PhaseType, float]] = None) -> Dict[PhaseType, float]:
        """
        Calculate phase formation probabilities using thermodynamic + kinetic model
        
        Args:
            cs_concentration: Cs⁺ concentration (mol/L)
            pb_concentration: Pb²⁺ concentration (mol/L)
            temperature: Temperature (°C)
            kinetic_factors: Optional kinetic barriers (kJ/mol)
            
        Returns:
            Dictionary of phase probabilities
        """
        T = temperature + 273.15
        
        # Calculate relative Gibbs energies
        relative_G = self.gibbs_calc.calculate_relative_stability(temperature)
        
        # Add composition-dependent terms
        composition_effects = self._calculate_composition_effects(cs_concentration, pb_concentration, T)
        
        # Add kinetic barriers if provided
        if kinetic_factors is None:
            kinetic_factors = {
                PhaseType.CSPBBR3_3D: 0.0,    # Reference
                PhaseType.CS4PBBR6_0D: 15.0,  # Higher barrier
                PhaseType.CSPB2BR5_2D: 8.0     # Moderate barrier
            }
        
        # Combined energy (thermodynamic + kinetic + composition)
        combined_energies = {}
        for phase in [PhaseType.CSPBBR3_3D, PhaseType.CS4PBBR6_0D, PhaseType.CSPB2BR5_2D]:
            E_total = (relative_G[phase] + 
                      composition_effects[phase] + 
                      kinetic_factors[phase])  # kJ/mol
            combined_energies[phase] = E_total
        
        # Boltzmann distribution
        probabilities = {}
        total_weight = 0.0
        
        for phase, energy in combined_energies.items():
            weight = np.exp(-energy * 1000 / (self.R * T))  # Convert kJ to J
            probabilities[phase] = weight
            total_weight += weight
        
        # Normalize probabilities
        for phase in probabilities:
            probabilities[phase] /= total_weight
        
        # Add mixed phases and failed synthesis possibilities
        max_prob = max(probabilities.values())
        
        # Mixed phases more likely when energies are close
        energy_spread = max(combined_energies.values()) - min(combined_energies.values())
        mixed_prob = 0.1 * np.exp(-energy_spread / 5.0)  # Higher prob for close energies
        
        # Failed synthesis at extreme conditions
        if temperature < 60 or temperature > 250:
            failed_prob = 0.2
        elif cs_concentration < 0.1 or pb_concentration < 0.1:
            failed_prob = 0.15
        else:
            failed_prob = 0.05
        
        # Renormalize
        scaling = (1 - mixed_prob - failed_prob) / sum(probabilities.values())
        for phase in probabilities:
            probabilities[phase] *= scaling
        
        probabilities[PhaseType.MIXED_PHASES] = mixed_prob
        probabilities[PhaseType.FAILED_SYNTHESIS] = failed_prob
        
        return probabilities
    
    def _calculate_composition_effects(self, 
                                     cs_conc: float, 
                                     pb_conc: float, 
                                     temperature: float) -> Dict[PhaseType, float]:
        """Calculate composition-dependent energy contributions"""
        cs_pb_ratio = cs_conc / (pb_conc + 1e-8)
        
        # Ideal stoichiometric ratios for each phase
        ideal_ratios = {
            PhaseType.CSPBBR3_3D: 1.0,    # 1:1 Cs:Pb
            PhaseType.CS4PBBR6_0D: 4.0,   # 4:1 Cs:Pb
            PhaseType.CSPB2BR5_2D: 0.5    # 1:2 Cs:Pb
        }
        
        composition_effects = {}
        
        for phase, ideal_ratio in ideal_ratios.items():
            # Penalty for deviation from ideal stoichiometry
            deviation = abs(np.log(cs_pb_ratio / ideal_ratio))
            penalty = 2.0 * deviation  # kJ/mol per ln unit deviation
            composition_effects[phase] = penalty
        
        return composition_effects
    
    def predict_dominant_phase(self, 
                             cs_concentration: float,
                             pb_concentration: float,
                             temperature: float) -> Tuple[PhaseType, float]:
        """Predict most likely phase and confidence"""
        probabilities = self.calculate_phase_probabilities(cs_concentration, pb_concentration, temperature)
        
        # Find dominant phase
        dominant_phase = max(probabilities, key=probabilities.get)
        confidence = probabilities[dominant_phase]
        
        return dominant_phase, confidence


class PyTorchPhaseSelectionModel(nn.Module):
    """PyTorch wrapper for phase selection calculations"""
    
    def __init__(self):
        super().__init__()
        self.phase_model = PhaseSelectionModel()
        
        # Learnable parameters
        self.thermodynamic_corrections = nn.Parameter(torch.zeros(3))  # Per phase
        self.composition_weights = nn.Parameter(torch.ones(3))
        
    def forward(self, synthesis_params: torch.Tensor) -> torch.Tensor:
        """
        Calculate phase selection features
        
        Args:
            synthesis_params: Tensor of shape (batch_size, 7)
            
        Returns:
            Tensor of shape (batch_size, 5) with phase probabilities
        """
        batch_size = synthesis_params.shape[0]
        device = synthesis_params.device
        
        cs_conc = synthesis_params[:, 0]
        pb_conc = synthesis_params[:, 1]
        temperature = synthesis_params[:, 3]
        
        phase_features = torch.zeros(batch_size, 5, device=device)
        
        for i in range(batch_size):
            cs_val = cs_conc[i].item()
            pb_val = pb_conc[i].item()
            temp_val = temperature[i].item()
            
            # Calculate probabilities
            probabilities = self.phase_model.calculate_phase_probabilities(cs_val, pb_val, temp_val)
            
            # Convert to tensor
            prob_values = [
                probabilities[PhaseType.CSPBBR3_3D],
                probabilities[PhaseType.CS4PBBR6_0D],
                probabilities[PhaseType.CSPB2BR5_2D],
                probabilities[PhaseType.MIXED_PHASES],
                probabilities[PhaseType.FAILED_SYNTHESIS]
            ]
            
            phase_features[i] = torch.tensor(prob_values, device=device)
        
        return phase_features


def create_phase_selection_model() -> PhaseSelectionModel:
    """Factory function"""
    return PhaseSelectionModel()


def create_pytorch_phase_model() -> PyTorchPhaseSelectionModel:
    """Factory function"""
    return PyTorchPhaseSelectionModel()
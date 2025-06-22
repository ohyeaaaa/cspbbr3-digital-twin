#!/usr/bin/env python3
"""
Classical Nucleation Theory Models for CsPbBr₃ Digital Twin
Physics-based nucleation rate calculations and critical size predictions
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class PhaseType(Enum):
    """Perovskite phase types"""
    CSPBBR3_3D = 0      # 3D Perovskite
    CS4PBBR6_0D = 1     # 0D Zero-dimensional
    CSPB2BR5_2D = 2     # 2D Quasi-layered
    MIXED_PHASES = 3     # Multiple phases
    FAILED_SYNTHESIS = 4 # No crystalline product


@dataclass
class PhysicalConstants:
    """Physical constants for nucleation calculations"""
    k_B = 1.380649e-23    # Boltzmann constant (J/K)
    N_A = 6.02214076e23   # Avogadro number
    R = 8.314462618       # Gas constant (J/mol·K)
    h = 6.62607015e-34    # Planck constant (J·s)
    
    # Material-specific constants from literature
    cspbbr3_lattice_param = 5.87e-10      # m, cubic lattice parameter
    cspbbr3_molecular_volume = 2.02e-28    # m³, volume per formula unit
    cspbbr3_molar_volume = 1.22e-4        # m³/mol
    
    # Surface energies from DFT calculations (J/m²)
    surface_energies = {
        PhaseType.CSPBBR3_3D: 0.134,
        PhaseType.CS4PBBR6_0D: 0.145,
        PhaseType.CSPB2BR5_2D: 0.128
    }
    
    # Solubility products (temperature dependent)
    # Base values at 298K in different solvents
    ksp_base = {
        'DMSO': {'CsPbBr3': 1.2e-15, 'Cs4PbBr6': 3.5e-18, 'CsPb2Br5': 8.7e-17},
        'DMF': {'CsPbBr3': 5.8e-16, 'Cs4PbBr6': 1.2e-18, 'CsPb2Br5': 4.1e-17},
        'water': {'CsPbBr3': 2.1e-18, 'Cs4PbBr6': 1.8e-20, 'CsPb2Br5': 1.5e-19},
        'toluene': {'CsPbBr3': 1.5e-20, 'Cs4PbBr6': 3.2e-23, 'CsPb2Br5': 8.9e-22},
        'octadecene': {'CsPbBr3': 8.7e-22, 'Cs4PbBr6': 1.1e-24, 'CsPb2Br5': 3.4e-23}
    }
    
    # Enthalpy of dissolution (J/mol) - estimated from temperature dependence
    delta_h_dissolution = {
        'CsPbBr3': -48000,
        'Cs4PbBr6': -52000,
        'CsPb2Br5': -45000
    }


@dataclass
class NucleationResult:
    """Results from nucleation calculations"""
    nucleation_rate: float              # nuclei/(m³·s)
    critical_radius: float              # m
    critical_free_energy: float         # J
    supersaturation: float              # dimensionless
    thermodynamic_driving_force: float  # J/mol
    kinetic_prefactor: float            # s⁻¹
    temperature: float                  # K
    phase_type: PhaseType
    confidence: float                   # 0-1, model confidence


class ClassicalNucleationTheory:
    """
    Classical nucleation theory implementation for perovskite formation
    Based on Volmer-Weber and Becker-Döring theories
    """
    
    def __init__(self):
        self.constants = PhysicalConstants()
        self.solvent_properties = self._initialize_solvent_properties()
    
    def _initialize_solvent_properties(self) -> Dict[str, Dict[str, float]]:
        """Initialize solvent-dependent properties"""
        return {
            'DMSO': {
                'dielectric_constant': 47.2,
                'viscosity': 1.99e-3,  # Pa·s
                'density': 1100,       # kg/m³
                'dipole_moment': 3.96  # Debye
            },
            'DMF': {
                'dielectric_constant': 38.3,
                'viscosity': 0.80e-3,
                'density': 944,
                'dipole_moment': 3.82
            },
            'water': {
                'dielectric_constant': 80.1,
                'viscosity': 1.00e-3,
                'density': 1000,
                'dipole_moment': 1.85
            },
            'toluene': {
                'dielectric_constant': 2.38,
                'viscosity': 0.56e-3,
                'density': 867,
                'dipole_moment': 0.36
            },
            'octadecene': {
                'dielectric_constant': 2.05,
                'viscosity': 2.84e-3,
                'density': 789,
                'dipole_moment': 0.0
            }
        }
    
    def calculate_supersaturation(self, 
                                cs_concentration: float,
                                pb_concentration: float,
                                br_concentration: float,
                                temperature: float,
                                solvent: str,
                                phase_type: PhaseType) -> float:
        """
        Calculate supersaturation ratio for specific phase
        
        Args:
            cs_concentration: Cs⁺ concentration (mol/L)
            pb_concentration: Pb²⁺ concentration (mol/L)
            br_concentration: Br⁻ concentration (mol/L)
            temperature: Temperature (°C)
            solvent: Solvent type
            phase_type: Target phase type
            
        Returns:
            Supersaturation ratio (S = IAP/Ksp)
        """
        T = temperature + 273.15  # Convert to Kelvin
        T_ref = 298.15  # Reference temperature
        
        # Get phase-specific stoichiometry
        stoichiometry = self._get_phase_stoichiometry(phase_type)
        
        # Calculate ion activity product
        ionic_strength = 0.5 * (cs_concentration + 4 * pb_concentration + br_concentration)
        activity_coefficients = self._calculate_activity_coefficients(ionic_strength, T, solvent)
        
        cs_activity = cs_concentration * activity_coefficients['cs']
        pb_activity = pb_concentration * activity_coefficients['pb']
        br_activity = br_concentration * activity_coefficients['br']
        
        # Phase-specific IAP calculation
        if phase_type == PhaseType.CSPBBR3_3D:
            iap = cs_activity * pb_activity * (br_activity ** 3)
            phase_name = 'CsPbBr3'
        elif phase_type == PhaseType.CS4PBBR6_0D:
            iap = (cs_activity ** 4) * pb_activity * (br_activity ** 6)
            phase_name = 'Cs4PbBr6'
        elif phase_type == PhaseType.CSPB2BR5_2D:
            iap = cs_activity * (pb_activity ** 2) * (br_activity ** 5)
            phase_name = 'CsPb2Br5'
        else:
            return 1.0  # No supersaturation for mixed/failed phases
        
        # Temperature-dependent solubility product
        ksp_ref = self.constants.ksp_base[solvent][phase_name]
        delta_h = self.constants.delta_h_dissolution[phase_name]
        
        # van't Hoff equation
        ksp = ksp_ref * np.exp(delta_h / self.constants.R * (1/T_ref - 1/T))
        
        supersaturation = iap / ksp if ksp > 0 else 1.0
        
        return max(1.0, supersaturation)  # Minimum supersaturation is 1
    
    def _get_phase_stoichiometry(self, phase_type: PhaseType) -> Dict[str, int]:
        """Get stoichiometric coefficients for each phase"""
        stoichiometry = {
            PhaseType.CSPBBR3_3D: {'cs': 1, 'pb': 1, 'br': 3},
            PhaseType.CS4PBBR6_0D: {'cs': 4, 'pb': 1, 'br': 6},
            PhaseType.CSPB2BR5_2D: {'cs': 1, 'pb': 2, 'br': 5}
        }
        return stoichiometry.get(phase_type, {'cs': 1, 'pb': 1, 'br': 3})
    
    def _calculate_activity_coefficients(self, 
                                       ionic_strength: float,
                                       temperature: float,
                                       solvent: str) -> Dict[str, float]:
        """Calculate activity coefficients using extended Debye-Hückel equation"""
        # Extended Debye-Hückel parameters
        A = 0.509  # kg^0.5/mol^0.5 at 25°C in water
        B = 0.328  # kg^0.5·m^-1·mol^-0.5
        
        # Temperature correction
        A *= (temperature / 298.15) ** 0.5
        
        # Solvent correction based on dielectric constant
        epsilon_r = self.solvent_properties[solvent]['dielectric_constant']
        A *= (78.5 / epsilon_r) ** 1.5  # Water reference dielectric constant
        
        # Ion-specific parameters (Å)
        ion_sizes = {'cs': 3.0, 'pb': 4.0, 'br': 3.5}
        
        activity_coeffs = {}
        for ion, charge in [('cs', 1), ('pb', 2), ('br', 1)]:
            a_ion = ion_sizes[ion]
            
            # Extended Debye-Hückel equation
            numerator = -A * charge**2 * np.sqrt(ionic_strength)
            denominator = 1 + B * a_ion * np.sqrt(ionic_strength)
            
            log_gamma = numerator / denominator
            activity_coeffs[ion] = np.exp(log_gamma)
        
        return activity_coeffs
    
    def calculate_critical_radius(self,
                                supersaturation: float,
                                temperature: float,
                                phase_type: PhaseType) -> float:
        """
        Calculate critical nucleus radius using Gibbs-Thomson equation
        
        Args:
            supersaturation: Supersaturation ratio
            temperature: Temperature (°C)
            phase_type: Phase type for surface energy
            
        Returns:
            Critical radius (m)
        """
        if supersaturation <= 1.0:
            return float('inf')  # No nucleation below saturation
        
        T = temperature + 273.15
        surface_energy = self.constants.surface_energies[phase_type]
        molecular_volume = self.constants.cspbbr3_molecular_volume
        
        # Gibbs-Thomson equation
        r_critical = (2 * surface_energy * molecular_volume) / (self.constants.k_B * T * np.log(supersaturation))
        
        return r_critical
    
    def calculate_critical_free_energy(self,
                                     critical_radius: float,
                                     phase_type: PhaseType) -> float:
        """Calculate critical nucleation free energy"""
        surface_energy = self.constants.surface_energies[phase_type]
        
        # For spherical nucleus
        delta_g_critical = (16 * np.pi * surface_energy**3) / (3 * (4 * np.pi * surface_energy / critical_radius)**2)
        
        # Simplified: ΔG* = (4π/3) * σ * r*²
        delta_g_critical = (4 * np.pi / 3) * surface_energy * critical_radius**2
        
        return delta_g_critical
    
    def calculate_kinetic_prefactor(self,
                                  cs_concentration: float,
                                  pb_concentration: float,
                                  temperature: float,
                                  solvent: str) -> float:
        """
        Calculate kinetic prefactor for nucleation rate
        
        Based on attachment frequency of growth units
        """
        T = temperature + 273.15
        
        # Concentration of growth units (formula units per m³)
        total_concentration = (cs_concentration + pb_concentration) * 1000 * self.constants.N_A  # m⁻³
        
        # Attachment frequency (simplified kinetic theory)
        # ν = (kT/h) * exp(-Ea/kT)
        activation_energy = 40000  # J/mol, estimated attachment barrier
        
        frequency = (self.constants.k_B * T / self.constants.h) * np.exp(-activation_energy / (self.constants.R * T))
        
        # Kinetic prefactor includes concentration and frequency
        kinetic_prefactor = total_concentration * frequency  # nuclei/(m³·s)
        
        return kinetic_prefactor
    
    def calculate_nucleation_rate(self,
                                cs_concentration: float,
                                pb_concentration: float,
                                br_concentration: float,
                                temperature: float,
                                solvent: str,
                                phase_type: PhaseType) -> NucleationResult:
        """
        Calculate complete nucleation rate using classical nucleation theory
        
        Returns:
            NucleationResult with all calculated parameters
        """
        # Calculate supersaturation
        supersaturation = self.calculate_supersaturation(
            cs_concentration, pb_concentration, br_concentration,
            temperature, solvent, phase_type
        )
        
        if supersaturation <= 1.0:
            return NucleationResult(
                nucleation_rate=0.0,
                critical_radius=float('inf'),
                critical_free_energy=float('inf'),
                supersaturation=supersaturation,
                thermodynamic_driving_force=0.0,
                kinetic_prefactor=0.0,
                temperature=temperature + 273.15,
                phase_type=phase_type,
                confidence=1.0  # High confidence for no nucleation
            )
        
        T = temperature + 273.15
        
        # Calculate critical radius
        critical_radius = self.calculate_critical_radius(supersaturation, temperature, phase_type)
        
        # Calculate critical free energy
        critical_free_energy = self.calculate_critical_free_energy(critical_radius, phase_type)
        
        # Calculate kinetic prefactor
        kinetic_prefactor = self.calculate_kinetic_prefactor(
            cs_concentration, pb_concentration, temperature, solvent
        )
        
        # Thermodynamic driving force
        driving_force = self.constants.R * T * np.log(supersaturation)  # J/mol
        
        # Classical nucleation rate: J = A * exp(-ΔG*/kT)
        boltzmann_factor = np.exp(-critical_free_energy / (self.constants.k_B * T))
        nucleation_rate = kinetic_prefactor * boltzmann_factor
        
        # Estimate model confidence based on supersaturation and temperature
        confidence = self._estimate_confidence(supersaturation, temperature, phase_type)
        
        return NucleationResult(
            nucleation_rate=nucleation_rate,
            critical_radius=critical_radius,
            critical_free_energy=critical_free_energy,
            supersaturation=supersaturation,
            thermodynamic_driving_force=driving_force,
            kinetic_prefactor=kinetic_prefactor,
            temperature=T,
            phase_type=phase_type,
            confidence=confidence
        )
    
    def _estimate_confidence(self, supersaturation: float, temperature: float, phase_type: PhaseType) -> float:
        """Estimate model confidence based on parameter ranges"""
        confidence = 1.0
        
        # Reduce confidence for extreme supersaturations
        if supersaturation > 100:
            confidence *= 0.7  # High supersaturation may not follow classical theory
        elif supersaturation < 1.1:
            confidence *= 0.8  # Near-saturation conditions are sensitive
        
        # Reduce confidence for extreme temperatures
        T = temperature + 273.15
        if T > 373:  # Above 100°C
            confidence *= 0.8
        elif T < 273:  # Below 0°C
            confidence *= 0.6
        
        # Phase-specific confidence
        if phase_type == PhaseType.CSPBBR3_3D:
            confidence *= 0.95  # Well-studied phase
        elif phase_type in [PhaseType.CS4PBBR6_0D, PhaseType.CSPB2BR5_2D]:
            confidence *= 0.85  # Less data available
        else:
            confidence *= 0.5   # Mixed/failed phases are complex
        
        return max(0.1, min(1.0, confidence))
    
    def calculate_competitive_nucleation(self,
                                       cs_concentration: float,
                                       pb_concentration: float,
                                       br_concentration: float,
                                       temperature: float,
                                       solvent: str) -> Dict[PhaseType, NucleationResult]:
        """
        Calculate nucleation rates for all competing phases
        
        Returns:
            Dictionary mapping phase types to nucleation results
        """
        results = {}
        
        for phase_type in [PhaseType.CSPBBR3_3D, PhaseType.CS4PBBR6_0D, PhaseType.CSPB2BR5_2D]:
            result = self.calculate_nucleation_rate(
                cs_concentration, pb_concentration, br_concentration,
                temperature, solvent, phase_type
            )
            results[phase_type] = result
        
        return results
    
    def predict_dominant_phase(self,
                             cs_concentration: float,
                             pb_concentration: float,
                             br_concentration: float,
                             temperature: float,
                             solvent: str) -> Tuple[PhaseType, float]:
        """
        Predict the dominant phase based on nucleation rates
        
        Returns:
            Tuple of (dominant_phase, confidence)
        """
        competitive_results = self.calculate_competitive_nucleation(
            cs_concentration, pb_concentration, br_concentration,
            temperature, solvent
        )
        
        # Find phase with highest nucleation rate
        max_rate = 0.0
        dominant_phase = PhaseType.FAILED_SYNTHESIS
        
        for phase_type, result in competitive_results.items():
            if result.nucleation_rate > max_rate:
                max_rate = result.nucleation_rate
                dominant_phase = phase_type
        
        # Calculate relative confidence
        total_rate = sum(result.nucleation_rate for result in competitive_results.values())
        relative_confidence = max_rate / total_rate if total_rate > 0 else 0.0
        
        # If all rates are very low, predict failed synthesis
        if max_rate < 1e10:  # nuclei/(m³·s), threshold for viable nucleation
            dominant_phase = PhaseType.FAILED_SYNTHESIS
            relative_confidence = 0.8
        
        return dominant_phase, relative_confidence


class PyTorchNucleationModel(nn.Module):
    """
    PyTorch wrapper for nucleation theory calculations
    Allows integration with neural networks and gradient-based optimization
    """
    
    def __init__(self):
        super().__init__()
        self.nucleation_theory = ClassicalNucleationTheory()
        
        # Learnable correction factors
        self.surface_energy_corrections = nn.Parameter(torch.ones(3))  # For 3 phases
        self.kinetic_corrections = nn.Parameter(torch.ones(5))  # For 5 solvents
        
    def forward(self, synthesis_params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for nucleation calculations
        
        Args:
            synthesis_params: Tensor of shape (batch_size, 7) containing:
                [cs_conc, pb_conc, br_conc, temperature, solvent_idx, oa_conc, oam_conc]
        
        Returns:
            Tensor of shape (batch_size, 9) containing nucleation features:
                [supersaturation_3d, supersaturation_0d, supersaturation_2d,
                 nucleation_rate_3d, nucleation_rate_0d, nucleation_rate_2d,
                 critical_radius_3d, critical_radius_0d, critical_radius_2d]
        """
        batch_size = synthesis_params.shape[0]
        device = synthesis_params.device
        
        # Extract parameters
        cs_conc = synthesis_params[:, 0]
        pb_conc = synthesis_params[:, 1]
        br_conc = synthesis_params[:, 2]
        temperature = synthesis_params[:, 3]
        solvent_idx = synthesis_params[:, 4].long()
        
        solvent_names = ['DMSO', 'DMF', 'water', 'toluene', 'octadecene']
        
        nucleation_features = torch.zeros(batch_size, 9, device=device)
        
        for i in range(batch_size):
            cs_val = cs_conc[i].item()
            pb_val = pb_conc[i].item()
            br_val = br_conc[i].item()
            temp_val = temperature[i].item()
            solvent = solvent_names[solvent_idx[i].item()]
            
            # Calculate for each phase
            phase_features = []
            for j, phase_type in enumerate([PhaseType.CSPBBR3_3D, PhaseType.CS4PBBR6_0D, PhaseType.CSPB2BR5_2D]):
                result = self.nucleation_theory.calculate_nucleation_rate(
                    cs_val, pb_val, br_val, temp_val, solvent, phase_type
                )
                
                # Apply learnable corrections
                corrected_rate = result.nucleation_rate * self.kinetic_corrections[solvent_idx[i]].item()
                
                phase_features.extend([
                    result.supersaturation,
                    np.log(corrected_rate + 1e-10),  # Log nucleation rate
                    np.log(result.critical_radius + 1e-12)  # Log critical radius
                ])
            
            nucleation_features[i] = torch.tensor(phase_features, device=device)
        
        return nucleation_features


def create_nucleation_model() -> ClassicalNucleationTheory:
    """Factory function to create nucleation model"""
    return ClassicalNucleationTheory()


def create_pytorch_nucleation_model() -> PyTorchNucleationModel:
    """Factory function to create PyTorch nucleation model"""
    return PyTorchNucleationModel()
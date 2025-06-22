#!/usr/bin/env python3
"""
Crystal Growth Kinetics Models for CsPbBr₃ Digital Twin
Burton-Cabrera-Frank growth model and size distribution predictions
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


@dataclass
class GrowthResult:
    """Results from crystal growth calculations"""
    growth_rate: float                    # m/s
    final_size: float                     # m
    growth_time: float                    # s
    size_distribution_width: float       # relative standard deviation
    aspect_ratio: float                   # length/width for anisotropic growth
    surface_coverage: float               # ligand coverage fraction
    diffusion_limited: bool               # True if diffusion-limited
    surface_limited: bool                 # True if surface-limited
    temperature: float                    # K
    confidence: float                     # 0-1, model confidence


@dataclass
class DiffusionConstants:
    """Diffusion constants for different species and solvents"""
    # Base diffusion coefficients at 298K (m²/s)
    base_coefficients = {
        'DMSO': {'Cs': 1.2e-10, 'Pb': 0.8e-10, 'Br': 1.8e-10, 'complex': 0.5e-10},
        'DMF': {'Cs': 0.8e-10, 'Pb': 0.6e-10, 'Br': 1.2e-10, 'complex': 0.4e-10},
        'water': {'Cs': 2.1e-9, 'Pb': 0.9e-9, 'Br': 2.0e-9, 'complex': 0.7e-9},
        'toluene': {'Cs': 0.5e-10, 'Pb': 0.3e-10, 'Br': 0.8e-10, 'complex': 0.2e-10},
        'octadecene': {'Cs': 0.3e-10, 'Pb': 0.2e-10, 'Br': 0.5e-10, 'complex': 0.15e-10}
    }
    
    # Activation energies for diffusion (J/mol)
    activation_energies = {
        'DMSO': 15000,
        'DMF': 12000,
        'water': 18000,
        'toluene': 8000,
        'octadecene': 20000
    }


class BurtonCabreraFrankModel:
    """
    Burton-Cabrera-Frank (BCF) crystal growth model
    Accounts for surface kinetics, diffusion, and ligand effects
    """
    
    def __init__(self):
        self.diffusion_constants = DiffusionConstants()
        self.k_B = 1.380649e-23  # Boltzmann constant
        self.R = 8.314462618     # Gas constant
        
        # Surface properties
        self.step_energy = 0.02  # eV, typical step energy
        self.kink_energy = 0.01  # eV, kink site energy
        self.adatom_energy = 0.05  # eV, adatom binding energy
        
        # Ligand binding energies (eV)
        self.ligand_binding = {
            'oleic_acid': {'100': 0.8, '110': 0.9, '111': 0.7},
            'oleylamine': {'100': 0.7, '110': 0.8, '111': 0.6}
        }
    
    def calculate_diffusion_coefficient(self, 
                                      species: str, 
                                      temperature: float, 
                                      solvent: str,
                                      viscosity_correction: bool = True) -> float:
        """
        Calculate temperature-dependent diffusion coefficient
        
        Args:
            species: Ion species ('Cs', 'Pb', 'Br', 'complex')
            temperature: Temperature in °C
            solvent: Solvent type
            viscosity_correction: Apply Stokes-Einstein viscosity correction
            
        Returns:
            Diffusion coefficient in m²/s
        """
        T = temperature + 273.15  # Convert to Kelvin
        T_ref = 298.15
        
        # Base diffusion coefficient
        D_ref = self.diffusion_constants.base_coefficients[solvent][species]
        
        # Arrhenius temperature dependence
        Ea = self.diffusion_constants.activation_energies[solvent]
        D_temp = D_ref * np.exp(-Ea / self.R * (1/T - 1/T_ref))
        
        if viscosity_correction:
            # Stokes-Einstein relation: D ∝ T/η
            # Simplified viscosity temperature dependence
            viscosity_ratio = np.exp(Ea / (2 * self.R) * (1/T - 1/T_ref))
            D_temp *= (T / T_ref) / viscosity_ratio
        
        return D_temp
    
    def calculate_boundary_layer_thickness(self,
                                         growth_rate: float,
                                         diffusion_coeff: float,
                                         concentration: float) -> float:
        """Calculate diffusion boundary layer thickness"""
        if growth_rate <= 0:
            return 1e-6  # Default 1 μm
        
        # Simplified boundary layer model
        # δ = sqrt(D / v) where v is growth velocity
        growth_velocity = growth_rate  # m/s
        
        boundary_thickness = np.sqrt(diffusion_coeff / growth_velocity)
        
        # Reasonable bounds
        return max(1e-9, min(1e-4, boundary_thickness))  # 1 nm to 100 μm
    
    def calculate_surface_attachment_rate(self,
                                        temperature: float,
                                        supersaturation: float,
                                        ligand_coverage: float = 0.0,
                                        surface_orientation: str = "100") -> float:
        """
        Calculate surface attachment rate coefficient
        
        Args:
            temperature: Temperature in °C
            supersaturation: Supersaturation ratio
            ligand_coverage: Fraction of surface covered by ligands (0-1)
            surface_orientation: Crystal face orientation
            
        Returns:
            Attachment rate coefficient (m/s)
        """
        T = temperature + 273.15
        
        # Available surface sites (reduced by ligand coverage)
        available_sites = 1.0 - ligand_coverage
        
        # Thermal velocity (kinetic theory)
        mass = 400 * 1.66e-27  # kg, approximate molecular mass of CsPbBr3
        thermal_velocity = np.sqrt(3 * self.k_B * T / mass)
        
        # Attachment probability (depends on supersaturation and surface energy)
        attachment_energy = self.adatom_energy * 1.602e-19  # Convert eV to J
        
        # Transition state theory
        attachment_prob = np.exp(-attachment_energy / (self.k_B * T))
        
        # Supersaturation enhancement
        driving_force = max(0, supersaturation - 1.0)
        enhanced_prob = attachment_prob * (1 + driving_force)
        
        # Surface kinetic coefficient
        kinetic_coeff = 0.25 * thermal_velocity * enhanced_prob * available_sites
        
        return kinetic_coeff
    
    def calculate_step_velocity(self,
                              supersaturation: float,
                              temperature: float,
                              step_density: float = 1e6) -> float:
        """Calculate step propagation velocity"""
        if supersaturation <= 1.0:
            return 0.0
        
        T = temperature + 273.15
        
        # Net attachment frequency at steps
        attachment_freq = 1e13 * np.exp(-self.step_energy * 1.602e-19 / (self.k_B * T))  # Hz
        
        # Driving force
        driving_force = supersaturation - 1.0
        
        # Step velocity
        lattice_param = 5.87e-10  # m
        step_velocity = attachment_freq * lattice_param * driving_force
        
        return step_velocity
    
    def calculate_growth_rate(self,
                            cs_concentration: float,
                            pb_concentration: float,
                            temperature: float,
                            supersaturation: float,
                            solvent: str,
                            ligand_coverage: float = 0.0,
                            crystal_size: float = 1e-8) -> GrowthResult:
        """
        Calculate crystal growth rate using BCF model
        
        Args:
            cs_concentration: Cs⁺ concentration (mol/L)
            pb_concentration: Pb²⁺ concentration (mol/L)
            temperature: Temperature (°C)
            supersaturation: Supersaturation ratio
            solvent: Solvent type
            ligand_coverage: Ligand surface coverage fraction
            crystal_size: Current crystal size (m)
            
        Returns:
            GrowthResult with calculated parameters
        """
        if supersaturation <= 1.0:
            return GrowthResult(
                growth_rate=0.0, final_size=crystal_size, growth_time=0.0,
                size_distribution_width=0.0, aspect_ratio=1.0,
                surface_coverage=ligand_coverage, diffusion_limited=False,
                surface_limited=True, temperature=temperature + 273.15,
                confidence=1.0
            )
        
        T = temperature + 273.15
        
        # Calculate diffusion coefficients
        D_cs = self.calculate_diffusion_coefficient('Cs', temperature, solvent)
        D_pb = self.calculate_diffusion_coefficient('Pb', temperature, solvent)
        D_complex = self.calculate_diffusion_coefficient('complex', temperature, solvent)
        
        # Effective diffusion coefficient (harmonic mean)
        D_eff = 3 / (1/D_cs + 1/D_pb + 1/D_complex)
        
        # Surface attachment rate
        attachment_rate = self.calculate_surface_attachment_rate(
            temperature, supersaturation, ligand_coverage
        )
        
        # Mass transport rate (diffusion-limited)
        total_concentration = (cs_concentration + pb_concentration) * 1000 * 6.022e23  # molecules/m³
        boundary_thickness = 1e-6  # Initial guess, will be refined
        
        # Iterative solution for growth rate
        growth_rate = 0.0
        for iteration in range(10):  # Convergence iterations
            # Update boundary layer thickness
            boundary_thickness = self.calculate_boundary_layer_thickness(
                growth_rate, D_eff, total_concentration
            )
            
            # Diffusion flux
            diffusion_flux = D_eff * total_concentration * (supersaturation - 1) / boundary_thickness
            
            # Surface kinetics flux
            surface_flux = attachment_rate * total_concentration * (supersaturation - 1)
            
            # Combined rate (series resistances)
            combined_rate = 1 / (1/diffusion_flux + 1/surface_flux)
            
            # Convert to linear growth rate
            molecular_volume = 2.02e-28  # m³ per formula unit
            new_growth_rate = combined_rate * molecular_volume
            
            # Check convergence
            if abs(new_growth_rate - growth_rate) / (growth_rate + 1e-15) < 0.01:
                break
            
            growth_rate = new_growth_rate
        
        # Determine limiting mechanism
        diffusion_limited = diffusion_flux < surface_flux
        surface_limited = not diffusion_limited
        
        # Calculate step velocity for anisotropic growth
        step_velocity = self.calculate_step_velocity(supersaturation, temperature)
        
        # Aspect ratio (simplified model)
        # Higher ligand coverage leads to more anisotropic growth
        base_aspect_ratio = 1.0 + 2.0 * ligand_coverage
        aspect_ratio = base_aspect_ratio * (1 + 0.5 * np.log(supersaturation))
        
        # Size distribution width (relative standard deviation)
        # Broader distribution at higher supersaturation and temperature
        relative_width = 0.1 + 0.2 * np.log(supersaturation) + 0.001 * (T - 298)
        size_distribution_width = max(0.05, min(0.8, relative_width))
        
        # Estimate confidence
        confidence = self._estimate_growth_confidence(
            supersaturation, temperature, ligand_coverage, solvent
        )
        
        return GrowthResult(
            growth_rate=growth_rate,
            final_size=crystal_size,  # Will be updated by time integration
            growth_time=0.0,  # Will be calculated externally
            size_distribution_width=size_distribution_width,
            aspect_ratio=aspect_ratio,
            surface_coverage=ligand_coverage,
            diffusion_limited=diffusion_limited,
            surface_limited=surface_limited,
            temperature=T,
            confidence=confidence
        )
    
    def integrate_growth_over_time(self,
                                 initial_size: float,
                                 growth_rate: float,
                                 reaction_time: float,
                                 supersaturation_decay: bool = True) -> Tuple[float, float]:
        """
        Integrate growth over time accounting for supersaturation decay
        
        Args:
            initial_size: Initial nucleus size (m)
            growth_rate: Initial growth rate (m/s)
            reaction_time: Total reaction time (minutes)
            supersaturation_decay: Account for supersaturation decay
            
        Returns:
            Tuple of (final_size, average_growth_rate)
        """
        time_seconds = reaction_time * 60  # Convert to seconds
        
        if not supersaturation_decay:
            # Simple linear growth
            final_size = initial_size + growth_rate * time_seconds
            return final_size, growth_rate
        
        # Account for supersaturation decay
        # Simplified exponential decay model
        decay_constant = 1.0 / (reaction_time * 60 / 3)  # 1/3 of reaction time
        
        # Integrate: size(t) = initial + ∫[0,t] rate(τ) dτ
        # where rate(τ) = rate_0 * exp(-decay_constant * τ)
        
        if decay_constant > 0:
            size_increase = growth_rate / decay_constant * (1 - np.exp(-decay_constant * time_seconds))
        else:
            size_increase = growth_rate * time_seconds
        
        final_size = initial_size + size_increase
        average_rate = size_increase / time_seconds if time_seconds > 0 else 0
        
        return final_size, average_rate
    
    def calculate_ligand_effects(self,
                               oa_concentration: float,
                               oam_concentration: float,
                               temperature: float,
                               crystal_surface_area: float) -> Dict[str, float]:
        """
        Calculate ligand effects on growth kinetics
        
        Args:
            oa_concentration: Oleic acid concentration (mol/L)
            oam_concentration: Oleylamine concentration (mol/L)
            temperature: Temperature (°C)
            crystal_surface_area: Crystal surface area (m²)
            
        Returns:
            Dictionary with ligand effects
        """
        T = temperature + 273.15
        
        # Surface site density (sites/m²)
        site_density = 1e19  # Typical for perovskite surfaces
        total_sites = crystal_surface_area * site_density
        
        # Langmuir adsorption isotherms
        # θ = K*C / (1 + K*C) where K = exp(E_bind / kT)
        
        # Oleic acid binding
        E_oa = self.ligand_binding['oleic_acid']['100'] * 1.602e-19  # J
        K_oa = np.exp(E_oa / (self.k_B * T))
        oa_conc_sites = oa_concentration * 1000 * 6.022e23  # molecules/m³
        theta_oa = K_oa * oa_conc_sites / (1 + K_oa * oa_conc_sites)
        
        # Oleylamine binding
        E_oam = self.ligand_binding['oleylamine']['100'] * 1.602e-19  # J
        K_oam = np.exp(E_oam / (self.k_B * T))
        oam_conc_sites = oam_concentration * 1000 * 6.022e23  # molecules/m³
        theta_oam = K_oam * oam_conc_sites / (1 + K_oam * oam_conc_sites)
        
        # Total coverage (assuming competitive binding)
        total_coverage = min(1.0, theta_oa + theta_oam)
        
        # Growth rate modification factors
        rate_reduction = 1.0 - 0.8 * total_coverage  # Strong inhibition
        anisotropy_enhancement = 1.0 + 2.0 * total_coverage
        
        return {
            'oa_coverage': theta_oa,
            'oam_coverage': theta_oam,
            'total_coverage': total_coverage,
            'rate_reduction_factor': rate_reduction,
            'anisotropy_factor': anisotropy_enhancement,
            'size_focusing_factor': 1.0 + total_coverage  # Ligands improve monodispersity
        }
    
    def _estimate_growth_confidence(self,
                                  supersaturation: float,
                                  temperature: float,
                                  ligand_coverage: float,
                                  solvent: str) -> float:
        """Estimate model confidence for growth calculations"""
        confidence = 1.0
        
        # Supersaturation effects
        if supersaturation > 50:
            confidence *= 0.7  # Very high supersaturation deviates from BCF
        elif supersaturation < 1.1:
            confidence *= 0.8  # Near-saturation is sensitive
        
        # Temperature effects
        T = temperature + 273.15
        if T > 373 or T < 273:
            confidence *= 0.7  # Extreme temperatures
        
        # Ligand coverage effects
        if ligand_coverage > 0.8:
            confidence *= 0.8  # High coverage complicates kinetics
        
        # Solvent effects
        if solvent in ['water', 'toluene']:
            confidence *= 0.8  # Less studied solvents
        
        return max(0.1, min(1.0, confidence))


class PyTorchGrowthModel(nn.Module):
    """
    PyTorch wrapper for growth kinetics calculations
    Enables integration with neural networks
    """
    
    def __init__(self):
        super().__init__()
        self.growth_model = BurtonCabreraFrankModel()
        
        # Learnable correction factors
        self.diffusion_corrections = nn.Parameter(torch.ones(5))  # Per solvent
        self.kinetic_corrections = nn.Parameter(torch.ones(3))    # Per mechanism
        
    def forward(self, synthesis_params: torch.Tensor, nucleation_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for growth calculations
        
        Args:
            synthesis_params: Tensor of shape (batch_size, 7)
            nucleation_features: Tensor from nucleation model
            
        Returns:
            Tensor of shape (batch_size, 6) with growth features:
                [growth_rate, final_size, aspect_ratio, size_distribution,
                 diffusion_limited, surface_limited]
        """
        batch_size = synthesis_params.shape[0]
        device = synthesis_params.device
        
        # Extract parameters
        cs_conc = synthesis_params[:, 0]
        pb_conc = synthesis_params[:, 1]
        temperature = synthesis_params[:, 3]
        solvent_idx = synthesis_params[:, 4].long()
        oa_conc = synthesis_params[:, 5]
        oam_conc = synthesis_params[:, 6]
        
        # Extract supersaturation from nucleation features
        supersaturation_3d = nucleation_features[:, 0]  # CsPbBr3 supersaturation
        
        solvent_names = ['DMSO', 'DMF', 'water', 'toluene', 'octadecene']
        
        growth_features = torch.zeros(batch_size, 6, device=device)
        
        for i in range(batch_size):
            cs_val = cs_conc[i].item()
            pb_val = pb_conc[i].item()
            temp_val = temperature[i].item()
            solvent = solvent_names[solvent_idx[i].item()]
            supersat = supersaturation_3d[i].item()
            oa_val = oa_conc[i].item()
            oam_val = oam_conc[i].item()
            
            # Estimate ligand coverage
            crystal_area = 1e-12  # m², typical nanocrystal
            ligand_effects = self.growth_model.calculate_ligand_effects(
                oa_val, oam_val, temp_val, crystal_area
            )
            ligand_coverage = ligand_effects['total_coverage']
            
            # Calculate growth
            result = self.growth_model.calculate_growth_rate(
                cs_val, pb_val, temp_val, supersat, solvent, ligand_coverage
            )
            
            # Apply learnable corrections
            corrected_rate = result.growth_rate * self.diffusion_corrections[solvent_idx[i]].item()
            
            growth_features[i] = torch.tensor([
                np.log(corrected_rate + 1e-15),  # Log growth rate
                np.log(result.final_size + 1e-12),  # Log size
                result.aspect_ratio,
                result.size_distribution_width,
                float(result.diffusion_limited),
                float(result.surface_limited)
            ], device=device)
        
        return growth_features


def create_growth_model() -> BurtonCabreraFrankModel:
    """Factory function to create growth model"""
    return BurtonCabreraFrankModel()


def create_pytorch_growth_model() -> PyTorchGrowthModel:
    """Factory function to create PyTorch growth model"""
    return PyTorchGrowthModel()
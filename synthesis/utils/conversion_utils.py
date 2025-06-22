#!/usr/bin/env python3
"""
Unit Conversion Utilities for CsPbBr₃ Digital Twin
Handle unit conversions and parameter normalization
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversionFactors:
    """Unit conversion factors"""
    # Temperature conversions (to Celsius)
    temperature = {
        'celsius': 1.0,
        'kelvin': lambda k: k - 273.15,
        'fahrenheit': lambda f: (f - 32) * 5/9
    }
    
    # Concentration conversions (to mol/L)
    concentration = {
        'mol/L': 1.0,
        'M': 1.0,
        'mM': 0.001,
        'mol/mL': 1000.0,
        'g/L': lambda g_per_l, molar_mass: g_per_l / molar_mass
    }
    
    # Time conversions (to minutes)
    time = {
        'minutes': 1.0,
        'min': 1.0,
        'seconds': 1/60,
        'hours': 60.0,
        'days': 1440.0
    }
    
    # Energy conversions (to eV)
    energy = {
        'eV': 1.0,
        'J': 6.242e18,
        'kJ/mol': 0.01036,
        'kcal/mol': 0.04336,
        'meV': 0.001
    }
    
    # Length conversions (to nm)
    length = {
        'nm': 1.0,
        'm': 1e9,
        'cm': 1e7,
        'mm': 1e6,
        'μm': 1e3,
        'pm': 1e-3,
        'Å': 0.1
    }


class UnitConverter:
    """Handle unit conversions for synthesis parameters"""
    
    def __init__(self):
        self.factors = ConversionFactors()
        
        # Molar masses (g/mol)
        self.molar_masses = {
            'CsBr': 212.81,
            'PbBr2': 367.01,
            'oleic_acid': 282.47,
            'oleylamine': 267.49,
            'DMSO': 78.13,
            'DMF': 73.09
        }
    
    def convert_temperature(self, value: float, from_unit: str, to_unit: str = 'celsius') -> float:
        """
        Convert temperature between units
        
        Args:
            value: Temperature value
            from_unit: Source unit ('celsius', 'kelvin', 'fahrenheit')
            to_unit: Target unit (default 'celsius')
            
        Returns:
            Converted temperature
        """
        if from_unit == to_unit:
            return value
        
        # Convert to Celsius first
        if from_unit == 'kelvin':
            celsius = value - 273.15
        elif from_unit == 'fahrenheit':
            celsius = (value - 32) * 5/9
        else:
            celsius = value
        
        # Convert from Celsius to target
        if to_unit == 'kelvin':
            return celsius + 273.15
        elif to_unit == 'fahrenheit':
            return celsius * 9/5 + 32
        else:
            return celsius
    
    def convert_concentration(self, 
                            value: float, 
                            from_unit: str, 
                            to_unit: str = 'mol/L',
                            compound: Optional[str] = None) -> float:
        """
        Convert concentration between units
        
        Args:
            value: Concentration value
            from_unit: Source unit
            to_unit: Target unit (default 'mol/L')
            compound: Compound name for g/L conversions
            
        Returns:
            Converted concentration
        """
        if from_unit == to_unit:
            return value
        
        # Convert to mol/L first
        if from_unit in ['mol/L', 'M']:
            mol_per_l = value
        elif from_unit == 'mM':
            mol_per_l = value * 0.001
        elif from_unit == 'mol/mL':
            mol_per_l = value * 1000.0
        elif from_unit == 'g/L' and compound:
            if compound in self.molar_masses:
                mol_per_l = value / self.molar_masses[compound]
            else:
                raise ValueError(f"Unknown molar mass for compound: {compound}")
        else:
            raise ValueError(f"Unknown concentration unit: {from_unit}")
        
        # Convert from mol/L to target
        if to_unit in ['mol/L', 'M']:
            return mol_per_l
        elif to_unit == 'mM':
            return mol_per_l * 1000.0
        elif to_unit == 'mol/mL':
            return mol_per_l / 1000.0
        elif to_unit == 'g/L' and compound:
            if compound in self.molar_masses:
                return mol_per_l * self.molar_masses[compound]
            else:
                raise ValueError(f"Unknown molar mass for compound: {compound}")
        else:
            raise ValueError(f"Unknown concentration unit: {to_unit}")
    
    def convert_time(self, value: float, from_unit: str, to_unit: str = 'minutes') -> float:
        """Convert time between units"""
        if from_unit == to_unit:
            return value
        
        # Convert to minutes first
        if from_unit in ['minutes', 'min']:
            minutes = value
        elif from_unit == 'seconds':
            minutes = value / 60
        elif from_unit == 'hours':
            minutes = value * 60
        elif from_unit == 'days':
            minutes = value * 1440
        else:
            raise ValueError(f"Unknown time unit: {from_unit}")
        
        # Convert from minutes to target
        if to_unit in ['minutes', 'min']:
            return minutes
        elif to_unit == 'seconds':
            return minutes * 60
        elif to_unit == 'hours':
            return minutes / 60
        elif to_unit == 'days':
            return minutes / 1440
        else:
            raise ValueError(f"Unknown time unit: {to_unit}")
    
    def convert_energy(self, value: float, from_unit: str, to_unit: str = 'eV') -> float:
        """Convert energy between units"""
        if from_unit == to_unit:
            return value
        
        # Convert to eV first
        if from_unit == 'eV':
            ev = value
        elif from_unit == 'J':
            ev = value * 6.242e18
        elif from_unit == 'kJ/mol':
            ev = value * 0.01036
        elif from_unit == 'kcal/mol':
            ev = value * 0.04336
        elif from_unit == 'meV':
            ev = value * 0.001
        else:
            raise ValueError(f"Unknown energy unit: {from_unit}")
        
        # Convert from eV to target
        if to_unit == 'eV':
            return ev
        elif to_unit == 'J':
            return ev / 6.242e18
        elif to_unit == 'kJ/mol':
            return ev / 0.01036
        elif to_unit == 'kcal/mol':
            return ev / 0.04336
        elif to_unit == 'meV':
            return ev / 0.001
        else:
            raise ValueError(f"Unknown energy unit: {to_unit}")
    
    def convert_length(self, value: float, from_unit: str, to_unit: str = 'nm') -> float:
        """Convert length between units"""
        if from_unit == to_unit:
            return value
        
        # Convert to nm first
        if from_unit == 'nm':
            nm = value
        elif from_unit == 'm':
            nm = value * 1e9
        elif from_unit == 'cm':
            nm = value * 1e7
        elif from_unit == 'mm':
            nm = value * 1e6
        elif from_unit == 'μm':
            nm = value * 1e3
        elif from_unit == 'pm':
            nm = value * 1e-3
        elif from_unit == 'Å':
            nm = value * 0.1
        else:
            raise ValueError(f"Unknown length unit: {from_unit}")
        
        # Convert from nm to target
        if to_unit == 'nm':
            return nm
        elif to_unit == 'm':
            return nm / 1e9
        elif to_unit == 'cm':
            return nm / 1e7
        elif to_unit == 'mm':
            return nm / 1e6
        elif to_unit == 'μm':
            return nm / 1e3
        elif to_unit == 'pm':
            return nm / 1e-3
        elif to_unit == 'Å':
            return nm / 0.1
        else:
            raise ValueError(f"Unknown length unit: {to_unit}")


def convert_temperature_units(temperature: Union[float, torch.Tensor],
                            from_unit: str,
                            to_unit: str = 'celsius') -> Union[float, torch.Tensor]:
    """Convert temperature with tensor support"""
    converter = UnitConverter()
    
    if isinstance(temperature, torch.Tensor):
        converted = torch.zeros_like(temperature)
        for i in range(temperature.numel()):
            converted.view(-1)[i] = converter.convert_temperature(
                temperature.view(-1)[i].item(), from_unit, to_unit
            )
        return converted
    else:
        return converter.convert_temperature(temperature, from_unit, to_unit)


def convert_concentration_units(concentration: Union[float, torch.Tensor],
                              from_unit: str,
                              to_unit: str = 'mol/L',
                              compound: Optional[str] = None) -> Union[float, torch.Tensor]:
    """Convert concentration with tensor support"""
    converter = UnitConverter()
    
    if isinstance(concentration, torch.Tensor):
        converted = torch.zeros_like(concentration)
        for i in range(concentration.numel()):
            converted.view(-1)[i] = converter.convert_concentration(
                concentration.view(-1)[i].item(), from_unit, to_unit, compound
            )
        return converted
    else:
        return converter.convert_concentration(concentration, from_unit, to_unit, compound)


def convert_time_units(time_val: Union[float, torch.Tensor],
                      from_unit: str,
                      to_unit: str = 'minutes') -> Union[float, torch.Tensor]:
    """Convert time with tensor support"""
    converter = UnitConverter()
    
    if isinstance(time_val, torch.Tensor):
        converted = torch.zeros_like(time_val)
        for i in range(time_val.numel()):
            converted.view(-1)[i] = converter.convert_time(
                time_val.view(-1)[i].item(), from_unit, to_unit
            )
        return converted
    else:
        return converter.convert_time(time_val, from_unit, to_unit)


def normalize_parameters(params: Dict[str, Union[float, torch.Tensor]],
                        method: str = 'min_max',
                        ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Normalize synthesis parameters
    
    Args:
        params: Dictionary of parameters
        method: Normalization method ('min_max', 'z_score', 'log_min_max')
        ranges: Custom ranges for min_max normalization
        
    Returns:
        Dictionary of normalized parameters
    """
    if ranges is None:
        # Default ranges
        ranges = {
            'cs_br_concentration': (0.1, 5.0),
            'pb_br2_concentration': (0.1, 3.0),
            'temperature': (60.0, 300.0),
            'oa_concentration': (0.0, 2.0),
            'oam_concentration': (0.0, 2.0),
            'reaction_time': (0.5, 120.0)
        }
    
    normalized = {}
    
    for param_name, value in params.items():
        if param_name in ranges:
            min_val, max_val = ranges[param_name]
            
            if method == 'min_max':
                # Scale to [0, 1]
                if isinstance(value, torch.Tensor):
                    normalized[param_name] = (value - min_val) / (max_val - min_val)
                else:
                    normalized[param_name] = (value - min_val) / (max_val - min_val)
            
            elif method == 'z_score':
                # Standardize to mean=0, std=1 (approximate for uniform distribution)
                mean_val = (min_val + max_val) / 2
                std_val = (max_val - min_val) / 3.464  # For uniform: std = range/sqrt(12)
                
                if isinstance(value, torch.Tensor):
                    normalized[param_name] = (value - mean_val) / std_val
                else:
                    normalized[param_name] = (value - mean_val) / std_val
            
            elif method == 'log_min_max':
                # Log transform then min-max scale
                if isinstance(value, torch.Tensor):
                    log_val = torch.log(value + 1e-8)
                    log_min = np.log(min_val + 1e-8)
                    log_max = np.log(max_val + 1e-8)
                    normalized[param_name] = (log_val - log_min) / (log_max - log_min)
                else:
                    log_val = np.log(value + 1e-8)
                    log_min = np.log(min_val + 1e-8)
                    log_max = np.log(max_val + 1e-8)
                    normalized[param_name] = (log_val - log_min) / (log_max - log_min)
            
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        else:
            # Keep unchanged if no range specified
            normalized[param_name] = value
    
    return normalized


def denormalize_parameters(normalized_params: Dict[str, Union[float, torch.Tensor]],
                          method: str = 'min_max',
                          ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Denormalize parameters back to original scale
    
    Args:
        normalized_params: Dictionary of normalized parameters
        method: Normalization method used
        ranges: Ranges used for normalization
        
    Returns:
        Dictionary of denormalized parameters
    """
    if ranges is None:
        ranges = {
            'cs_br_concentration': (0.1, 5.0),
            'pb_br2_concentration': (0.1, 3.0),
            'temperature': (60.0, 300.0),
            'oa_concentration': (0.0, 2.0),
            'oam_concentration': (0.0, 2.0),
            'reaction_time': (0.5, 120.0)
        }
    
    denormalized = {}
    
    for param_name, value in normalized_params.items():
        if param_name in ranges:
            min_val, max_val = ranges[param_name]
            
            if method == 'min_max':
                if isinstance(value, torch.Tensor):
                    denormalized[param_name] = value * (max_val - min_val) + min_val
                else:
                    denormalized[param_name] = value * (max_val - min_val) + min_val
            
            elif method == 'z_score':
                mean_val = (min_val + max_val) / 2
                std_val = (max_val - min_val) / 3.464
                
                if isinstance(value, torch.Tensor):
                    denormalized[param_name] = value * std_val + mean_val
                else:
                    denormalized[param_name] = value * std_val + mean_val
            
            elif method == 'log_min_max':
                log_min = np.log(min_val + 1e-8)
                log_max = np.log(max_val + 1e-8)
                
                if isinstance(value, torch.Tensor):
                    log_val = value * (log_max - log_min) + log_min
                    denormalized[param_name] = torch.exp(log_val) - 1e-8
                else:
                    log_val = value * (log_max - log_min) + log_min
                    denormalized[param_name] = np.exp(log_val) - 1e-8
            
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        else:
            denormalized[param_name] = value
    
    return denormalized


# Convenience functions for backward compatibility
def convert_temperature(temperature: Union[float, torch.Tensor],
                       from_unit: str,
                       to_unit: str = 'celsius') -> Union[float, torch.Tensor]:
    """Convert temperature - convenience wrapper"""
    return convert_temperature_units(temperature, from_unit, to_unit)


def convert_concentration(concentration: Union[float, torch.Tensor],
                         from_unit: str,
                         to_unit: str = 'mol/L',
                         compound: Optional[str] = None) -> Union[float, torch.Tensor]:
    """Convert concentration - convenience wrapper"""
    return convert_concentration_units(concentration, from_unit, to_unit, compound)


def convert_time(time_val: Union[float, torch.Tensor],
                from_unit: str,
                to_unit: str = 'minutes') -> Union[float, torch.Tensor]:
    """Convert time - convenience wrapper"""
    return convert_time_units(time_val, from_unit, to_unit)


def normalize_features(features: Dict[str, Union[float, torch.Tensor]],
                      method: str = 'min_max') -> Dict[str, Union[float, torch.Tensor]]:
    """Normalize features - convenience wrapper"""
    return normalize_parameters(features, method)
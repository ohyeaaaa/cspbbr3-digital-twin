#!/usr/bin/env python3
"""
Validation Utilities for CsPbBr₃ Digital Twin
Parameter validation and physics constraint checking
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParameterRanges:
    """Valid parameter ranges for synthesis"""
    cs_br_concentration: Tuple[float, float] = (0.1, 5.0)      # mol/L
    pb_br2_concentration: Tuple[float, float] = (0.1, 3.0)     # mol/L
    temperature: Tuple[float, float] = (60.0, 300.0)           # °C
    oa_concentration: Tuple[float, float] = (0.0, 2.0)         # mol/L
    oam_concentration: Tuple[float, float] = (0.0, 2.0)        # mol/L
    reaction_time: Tuple[float, float] = (0.5, 120.0)          # minutes
    
    # Property ranges
    bandgap: Tuple[float, float] = (0.5, 5.0)                  # eV
    plqy: Tuple[float, float] = (0.0, 1.0)                     # fraction
    emission_peak: Tuple[float, float] = (300.0, 800.0)        # nm
    particle_size: Tuple[float, float] = (1.0, 1000.0)         # nm


class PhysicsValidator:
    """Validate physics constraints and parameter consistency"""
    
    def __init__(self):
        self.ranges = ParameterRanges()
        self.tolerance = 1e-6
        
    def validate_mass_balance(self, 
                            cs_concentration: float,
                            pb_concentration: float,
                            br_concentration: float,
                            target_phase: str = "CsPbBr3") -> bool:
        """
        Validate mass balance for target phase formation
        
        Args:
            cs_concentration: Cs⁺ concentration (mol/L)
            pb_concentration: Pb²⁺ concentration (mol/L) 
            br_concentration: Br⁻ concentration (mol/L)
            target_phase: Target phase ("CsPbBr3", "Cs4PbBr6", "CsPb2Br5")
            
        Returns:
            True if mass balance is satisfied
        """
        # Stoichiometric requirements
        stoichiometry = {
            "CsPbBr3": {"cs": 1, "pb": 1, "br": 3},
            "Cs4PbBr6": {"cs": 4, "pb": 1, "br": 6},
            "CsPb2Br5": {"cs": 1, "pb": 2, "br": 5}
        }
        
        if target_phase not in stoichiometry:
            return False
        
        required = stoichiometry[target_phase]
        
        # Check if sufficient reactants are available
        limiting_factor = min(
            cs_concentration / required["cs"],
            pb_concentration / required["pb"],
            br_concentration / required["br"]
        )
        
        return limiting_factor > self.tolerance
    
    def validate_charge_balance(self,
                              cs_concentration: float,
                              pb_concentration: float,
                              br_concentration: float) -> bool:
        """Validate charge neutrality"""
        positive_charge = cs_concentration * 1 + pb_concentration * 2  # Cs⁺ + Pb²⁺
        negative_charge = br_concentration * 1                          # Br⁻
        
        charge_imbalance = abs(positive_charge - negative_charge)
        return charge_imbalance < 0.1 * max(positive_charge, negative_charge)
    
    def validate_thermodynamic_feasibility(self,
                                         temperature: float,
                                         supersaturation: float) -> bool:
        """Check if conditions are thermodynamically favorable"""
        # Basic thermodynamic checks
        if temperature < 0:  # Below absolute zero in Celsius is impossible
            return False
        
        if supersaturation < 1.0:  # No driving force for crystallization
            return False
        
        if supersaturation > 1000:  # Unrealistically high supersaturation
            return False
        
        return True
    
    def validate_kinetic_feasibility(self,
                                   temperature: float,
                                   reaction_time: float,
                                   nucleation_rate: float) -> bool:
        """Check if conditions allow for crystal formation"""
        # Temperature must be sufficient for molecular motion
        if temperature < 50:  # Below typical synthesis temperatures
            return False
        
        # Reasonable reaction time
        if reaction_time < 0.1 or reaction_time > 1440:  # 0.1 min to 24 hours
            return False
        
        # Nucleation rate should be reasonable
        if nucleation_rate < 1e5 or nucleation_rate > 1e20:  # nuclei/(m³·s)
            return False
        
        return True


def validate_synthesis_parameters(params: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate synthesis parameters against physical constraints
    
    Args:
        params: Dictionary of synthesis parameters
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    ranges = ParameterRanges()
    errors = []
    
    # Check concentration ranges
    for param in ['cs_br_concentration', 'pb_br2_concentration', 'oa_concentration', 'oam_concentration']:
        if param in params:
            value = params[param]
            min_val, max_val = getattr(ranges, param)
            if value < min_val or value > max_val:
                errors.append(f"{param} = {value} is outside valid range [{min_val}, {max_val}]")
    
    # Check temperature
    if 'temperature' in params:
        temp = params['temperature']
        min_temp, max_temp = ranges.temperature
        if temp < min_temp or temp > max_temp:
            errors.append(f"Temperature = {temp}°C is outside valid range [{min_temp}, {max_temp}]°C")
    
    # Check reaction time
    if 'reaction_time' in params:
        time = params['reaction_time']
        min_time, max_time = ranges.reaction_time
        if time < min_time or time > max_time:
            errors.append(f"Reaction time = {time} min is outside valid range [{min_time}, {max_time}] min")
    
    # Check solvent type
    if 'solvent_type' in params:
        solvent = params['solvent_type']
        valid_solvents = ['DMSO', 'DMF', 'water', 'toluene', 'octadecene']
        if isinstance(solvent, str) and solvent not in valid_solvents:
            errors.append(f"Solvent '{solvent}' not in valid list: {valid_solvents}")
        elif isinstance(solvent, (int, float)) and not (0 <= solvent <= 4):
            errors.append(f"Solvent index {solvent} not in valid range [0, 4]")
    
    # Physics-based validation
    validator = PhysicsValidator()
    
    if all(p in params for p in ['cs_br_concentration', 'pb_br2_concentration']):
        cs_conc = params['cs_br_concentration']
        pb_conc = params['pb_br2_concentration']
        br_conc = cs_conc + 2 * pb_conc  # Approximate total bromide
        
        # Mass balance check
        if not validator.validate_mass_balance(cs_conc, pb_conc, br_conc):
            errors.append("Insufficient reactants for target phase formation")
        
        # Charge balance check
        if not validator.validate_charge_balance(cs_conc, pb_conc, br_conc):
            errors.append("Charge imbalance in ionic composition")
    
    # Temperature-dependent checks
    if 'temperature' in params:
        temp = params['temperature']
        
        # Check for unrealistic temperature combinations
        if temp > 200 and any(params.get(p, 0) > 0.1 for p in ['oa_concentration', 'oam_concentration']):
            errors.append("High temperature with organic ligands may cause decomposition")
    
    return len(errors) == 0, errors


def validate_physics_constraints(predictions: Dict[str, torch.Tensor]) -> Tuple[bool, List[str]]:
    """
    Validate that predictions satisfy physics constraints
    
    Args:
        predictions: Model predictions
        
    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    violations = []
    
    # Check phase probabilities
    if 'phase' in predictions and 'probabilities' in predictions['phase']:
        phase_probs = predictions['phase']['probabilities']
        
        # Probabilities should sum to 1
        prob_sums = phase_probs.sum(dim=-1)
        if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=0.01):
            violations.append("Phase probabilities do not sum to 1")
        
        # Probabilities should be non-negative
        if (phase_probs < 0).any():
            violations.append("Negative phase probabilities detected")
    
    # Check property ranges
    if 'properties' in predictions:
        ranges = ParameterRanges()
        
        for prop_name, prop_output in predictions['properties'].items():
            if 'mean' in prop_output:
                prop_values = prop_output['mean']
                
                if hasattr(ranges, prop_name):
                    min_val, max_val = getattr(ranges, prop_name)
                    
                    if (prop_values < min_val).any():
                        violations.append(f"{prop_name} values below physical minimum {min_val}")
                    
                    if (prop_values > max_val).any():
                        violations.append(f"{prop_name} values above physical maximum {max_val}")
            
            # Check uncertainty values
            if 'std' in prop_output:
                uncertainty = prop_output['std']
                if (uncertainty < 0).any():
                    violations.append(f"Negative uncertainty in {prop_name}")
    
    return len(violations) == 0, violations


def check_parameter_ranges(params: Union[Dict[str, float], torch.Tensor], 
                         param_names: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Check if parameters are within valid ranges
    
    Args:
        params: Parameters to check (dict or tensor)
        param_names: Names of parameters if params is tensor
        
    Returns:
        Dictionary indicating which parameters are valid
    """
    ranges = ParameterRanges()
    results = {}
    
    if isinstance(params, torch.Tensor):
        if param_names is None:
            param_names = [f"param_{i}" for i in range(params.shape[-1])]
        
        params_dict = {name: params[..., i] for i, name in enumerate(param_names)}
    else:
        params_dict = params
    
    for param_name, values in params_dict.items():
        if hasattr(ranges, param_name):
            min_val, max_val = getattr(ranges, param_name)
            
            if isinstance(values, torch.Tensor):
                in_range = ((values >= min_val) & (values <= max_val)).all()
                results[param_name] = in_range.item()
            else:
                results[param_name] = min_val <= values <= max_val
        else:
            results[param_name] = True  # Unknown parameter, assume valid
    
    return results


def validate_batch_predictions(predictions: Dict[str, torch.Tensor],
                             targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Validate a batch of predictions against targets
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Dictionary of validation metrics
    """
    metrics = {}
    
    # Phase prediction validation
    if 'phase' in predictions and 'phase_labels' in targets:
        pred_phases = predictions['phase']['probabilities'].argmax(dim=-1)
        true_phases = targets['phase_labels']
        
        accuracy = (pred_phases == true_phases).float().mean()
        metrics['phase_accuracy'] = accuracy.item()
        
        # Check confidence calibration
        confidences = predictions['phase']['probabilities'].max(dim=-1)[0]
        correct = (pred_phases == true_phases).float()
        
        # Simple calibration metric
        avg_confidence = confidences.mean()
        avg_accuracy = correct.mean()
        calibration_error = abs(avg_confidence - avg_accuracy)
        metrics['calibration_error'] = calibration_error.item()
    
    # Property prediction validation
    if 'properties' in predictions and 'properties' in targets:
        property_errors = []
        
        for prop_name in predictions['properties']:
            if prop_name in targets['properties']:
                pred_mean = predictions['properties'][prop_name]['mean']
                true_values = targets['properties'][prop_name]
                
                mae = torch.abs(pred_mean - true_values).mean()
                property_errors.append(mae.item())
                metrics[f'{prop_name}_mae'] = mae.item()
        
        if property_errors:
            metrics['avg_property_mae'] = np.mean(property_errors)
    
    return metrics


def validate_prediction_result(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a prediction result for completeness and consistency
    
    Args:
        result: Prediction result dictionary
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check for required fields
    required_fields = ['phase', 'properties']
    for field in required_fields:
        if field not in result:
            issues.append(f"Missing required field: {field}")
    
    # Validate phase predictions
    if 'phase' in result:
        phase_data = result['phase']
        if 'probabilities' not in phase_data:
            issues.append("Phase predictions missing probabilities")
        elif 'predicted_class' not in phase_data:
            issues.append("Phase predictions missing predicted_class")
    
    # Validate property predictions
    if 'properties' in result:
        for prop_name, prop_data in result['properties'].items():
            if not isinstance(prop_data, dict):
                issues.append(f"Property '{prop_name}' should be a dictionary")
            elif 'mean' not in prop_data:
                issues.append(f"Property '{prop_name}' missing mean prediction")
    
    return len(issues) == 0, issues
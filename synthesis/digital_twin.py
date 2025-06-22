#!/usr/bin/env python3
"""
CsPbBr₃ Digital Twin - Main Integration Class
Orchestrates all components: physics models, feature engineering, and neural networks
"""

import torch
import numpy as np
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from pathlib import Path

# Import all components
from .schemas import (
    SynthesisParameters, MaterialProperties, PhysicsFeatures, 
    PredictionResult, PhaseType, SolventType, TrainingConfig
)
from .physics.nucleation import ClassicalNucleationTheory, create_nucleation_model
from .physics.growth import BurtonCabreraFrankModel, create_growth_model
from .physics.phase_selection import PhaseSelectionModel, create_phase_selection_model
from .physics.ligand_effects import CompetitiveLigandBinding, create_ligand_model
from .training.pytorch_feature_engineering import PhysicsInformedFeatureEngineer, create_feature_engineer
from .training.pytorch_neural_models import PhysicsInformedNeuralNetwork, create_model, load_pretrained_model
from .utils.validation_utils import validate_synthesis_parameters, validate_physics_constraints
from .utils.logging_utils import SynthesisLogger

logger = logging.getLogger(__name__)


class CsPbBr3DigitalTwin:
    """
    Main Digital Twin class for CsPbBr₃ synthesis prediction
    Integrates physics models with neural networks for comprehensive predictions
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 model_path: Optional[str] = None,
                 device: str = "auto"):
        """
        Initialize the digital twin
        
        Args:
            config: Configuration dictionary
            model_path: Path to pretrained model (optional)
            device: Device for computation ("auto", "cpu", "cuda")
        """
        self.config = config or self._default_config()
        self.device = self._setup_device(device)
        
        # Initialize physics models
        logger.info("Initializing physics models...")
        self.nucleation_model = create_nucleation_model()
        self.growth_model = create_growth_model()
        self.phase_selection_model = create_phase_selection_model()
        self.ligand_model = create_ligand_model()
        
        # Initialize feature engineering
        logger.info("Initializing feature engineering...")
        self.feature_engineer = create_feature_engineer(
            self.config.get('feature_engineering', {})
        ).to(self.device)
        
        # Initialize neural network
        logger.info("Initializing neural network...")
        if model_path and Path(model_path).exists():
            self.neural_network = load_pretrained_model(model_path)
            logger.info(f"Loaded pretrained model from {model_path}")
        else:
            self.neural_network = create_model(
                self.config.get('neural_network', {})
            )
            logger.info("Initialized new neural network model")
        
        self.neural_network.to(self.device)
        self.neural_network.eval()
        
        # Initialize logging
        self.logger = SynthesisLogger(
            log_dir=self.config.get('log_dir', 'logs'),
            enable_json_logging=True
        )
        
        # Performance tracking
        self.prediction_count = 0
        self.total_processing_time = 0.0
        self.start_time = time.time()
        
        logger.info("Digital Twin initialization complete")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'cache_size': 1000,
            'cache_ttl': 3600,
            'confidence_threshold': 0.5,
            'physics_weight': 0.3,
            'ml_weight': 0.7,
            'enable_caching': True,
            'enable_validation': True,
            'log_dir': 'logs',
            'neural_network': {
                'input_dim': 100,
                'hidden_dims': [256, 128, 64],
                'dropout_rate': 0.3,
                'physics_weight': 0.2,
                'uncertainty_weight': 0.1
            },
            'feature_engineering': {
                'normalize_features': True,
                'include_interactions': True
            }
        }
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Using CUDA GPU for acceleration")
            else:
                device = "cpu"
                logger.info("Using CPU for computation")
        
        return torch.device(device)
    
    def predict(self, 
                synthesis_params: Union[SynthesisParameters, Dict[str, float]],
                return_features: bool = False,
                enable_uncertainty: bool = True) -> PredictionResult:
        """
        Make a complete synthesis prediction
        
        Args:
            synthesis_params: Synthesis parameters
            return_features: Whether to include engineered features
            enable_uncertainty: Whether to calculate uncertainty
            
        Returns:
            Complete prediction result
        """
        start_time = time.time()
        prediction_id = str(uuid.uuid4())
        
        try:
            # Convert input to standard format
            if isinstance(synthesis_params, dict):
                params = SynthesisParameters(**synthesis_params)
            else:
                params = synthesis_params
            
            # Validate parameters
            if self.config.get('enable_validation', True):
                param_dict = params.to_dict()
                is_valid, errors = validate_synthesis_parameters(param_dict)
                if not is_valid:
                    raise ValueError(f"Invalid parameters: {errors}")
            
            # Calculate physics features
            physics_features = self._calculate_physics_features(params)
            
            # Engineer features for neural network
            param_tensor = params.to_tensor().unsqueeze(0).to(self.device)
            engineered_features = self.feature_engineer(
                {key: param_tensor[:, i] for i, key in enumerate([
                    'cs_br_concentration', 'pb_br2_concentration', 'temperature',
                    'solvent_type', 'oa_concentration', 'oam_concentration', 'reaction_time'
                ])}
            )
            
            # Neural network prediction
            with torch.no_grad():
                if enable_uncertainty:
                    # Monte Carlo uncertainty estimation
                    nn_outputs = self.neural_network.predict_with_uncertainty(
                        engineered_features, num_samples=50
                    )
                else:
                    # Single forward pass
                    nn_outputs = self.neural_network(engineered_features, training=False)
            
            # Process neural network outputs
            phase_probs = nn_outputs['phase']['probabilities'].cpu().numpy().flatten()
            phase_uncertainty = nn_outputs['phase'].get('uncertainty', torch.zeros_like(phase_probs)).cpu().numpy().flatten()
            
            # Determine primary phase
            primary_phase_idx = np.argmax(phase_probs)
            primary_phase = PhaseType(primary_phase_idx)
            
            # Extract property predictions
            properties = self._extract_properties(nn_outputs['properties'])
            property_uncertainties = self._extract_property_uncertainties(nn_outputs['properties'])
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(
                phase_probs, phase_uncertainty, property_uncertainties, physics_features
            )
            
            # Validate physics consistency
            physics_consistency = self._validate_physics_consistency(
                nn_outputs, physics_features
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Create result
            result = PredictionResult(
                primary_phase=primary_phase,
                phase_probabilities={PhaseType(i): float(prob) for i, prob in enumerate(phase_probs)},
                properties=properties,
                phase_uncertainty=float(np.mean(phase_uncertainty)),
                property_uncertainties=property_uncertainties,
                physics_features=physics_features,
                physics_consistency=float(physics_consistency),
                confidence=float(confidence),
                prediction_id=prediction_id,
                timestamp=datetime.utcnow().isoformat(),
                processing_time_ms=processing_time,
                model_version=getattr(self.neural_network, 'version', '2.0.0')
            )
            
            # Log prediction
            self.logger.log_synthesis_parameters(params.to_dict())
            self.logger.log_prediction_results(result.to_dict())
            
            # Update performance tracking
            self.prediction_count += 1
            self.total_processing_time += processing_time
            
            logger.info(f"Prediction completed in {processing_time:.2f}ms: {primary_phase.name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return self._create_failed_prediction(prediction_id, str(e), time.time() - start_time)
    
    def _calculate_physics_features(self, params: SynthesisParameters) -> PhysicsFeatures:
        """Calculate physics-based features"""
        # Calculate nucleation for all phases
        nucleation_results = {}
        for phase_type in [PhaseType.CSPBBR3_3D, PhaseType.CS4PBBR6_0D, PhaseType.CSPB2BR5_2D]:
            result = self.nucleation_model.calculate_nucleation_rate(
                params.cs_br_concentration,
                params.pb_br2_concentration,
                params.cs_br_concentration + 2 * params.pb_br2_concentration,  # Total Br
                params.temperature,
                params.solvent_type.to_string(),
                phase_type
            )
            nucleation_results[phase_type] = result
        
        # Calculate growth
        growth_result = self.growth_model.calculate_growth_rate(
            params.cs_br_concentration,
            params.pb_br2_concentration,
            params.temperature,
            nucleation_results[PhaseType.CSPBBR3_3D].supersaturation,
            params.solvent_type.to_string(),
            ligand_coverage=(params.oa_concentration + params.oam_concentration) / 
                           (params.cs_br_concentration + params.pb_br2_concentration + 1e-8)
        )
        
        # Calculate thermodynamic stability
        phase_probs = self.phase_selection_model.calculate_phase_probabilities(
            params.cs_br_concentration,
            params.pb_br2_concentration,
            params.temperature
        )
        
        # Calculate ligand effects
        crystal_area = 1e-12  # m², typical nanocrystal
        ligand_effects = self.ligand_model.calculate_ligand_effects(
            params.oa_concentration,
            params.oam_concentration,
            params.temperature,
            crystal_area
        )
        
        # Compile physics features
        return PhysicsFeatures(
            supersaturation=nucleation_results[PhaseType.CSPBBR3_3D].supersaturation,
            nucleation_rate_3d=nucleation_results[PhaseType.CSPBBR3_3D].nucleation_rate,
            nucleation_rate_0d=nucleation_results[PhaseType.CS4PBBR6_0D].nucleation_rate,
            nucleation_rate_2d=nucleation_results[PhaseType.CSPB2BR5_2D].nucleation_rate,
            critical_radius_3d=nucleation_results[PhaseType.CSPBBR3_3D].critical_radius,
            critical_radius_0d=nucleation_results[PhaseType.CS4PBBR6_0D].critical_radius,
            critical_radius_2d=nucleation_results[PhaseType.CSPB2BR5_2D].critical_radius,
            growth_rate=growth_result.growth_rate,
            diffusion_length=1e-6,  # Simplified
            ligand_coverage=ligand_effects['total_coverage'],
            gibbs_energy_3d=-125.4,  # kJ/mol, from thermodynamic data
            gibbs_energy_0d=-132.1,
            gibbs_energy_2d=-118.7,
            thermal_energy=1.380649e-23 * (params.temperature + 273.15) * 6.242e18,  # eV
            ionic_strength=0.5 * (params.cs_br_concentration + 4 * params.pb_br2_concentration)
        )
    
    def _extract_properties(self, property_outputs: Dict[str, Dict[str, torch.Tensor]]) -> MaterialProperties:
        """Extract material properties from neural network outputs"""
        # Extract mean values and convert to float
        properties = {}
        for prop_name, prop_data in property_outputs.items():
            if 'mean' in prop_data:
                properties[prop_name] = float(prop_data['mean'].cpu().item())
        
        # Create MaterialProperties with defaults for missing values
        return MaterialProperties(
            bandgap=properties.get('bandgap', 2.3),
            plqy=properties.get('plqy', 0.8),
            emission_peak=properties.get('emission_peak', 520.0),
            emission_fwhm=properties.get('emission_fwhm', 25.0),
            particle_size=properties.get('particle_size', 10.0),
            size_distribution_width=properties.get('size_distribution_width', 0.3),
            lifetime=properties.get('lifetime', 20.0),
            stability_score=properties.get('stability_score', 0.7),
            phase_purity=properties.get('phase_purity', 0.9)
        )
    
    def _extract_property_uncertainties(self, property_outputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Extract property uncertainties"""
        uncertainties = {}
        for prop_name, prop_data in property_outputs.items():
            if 'std' in prop_data:
                uncertainties[prop_name] = float(prop_data['std'].cpu().item())
            elif 'uncertainty' in prop_data:
                uncertainties[prop_name] = float(prop_data['uncertainty'].cpu().item())
            else:
                uncertainties[prop_name] = 0.1  # Default uncertainty
        
        return uncertainties
    
    def _calculate_overall_confidence(self, 
                                    phase_probs: np.ndarray,
                                    phase_uncertainty: np.ndarray,
                                    property_uncertainties: Dict[str, float],
                                    physics_features: PhysicsFeatures) -> float:
        """Calculate overall prediction confidence"""
        # Phase confidence (entropy-based)
        phase_entropy = -np.sum(phase_probs * np.log(phase_probs + 1e-8))
        max_entropy = -np.log(1.0 / len(phase_probs))  # Maximum possible entropy
        phase_confidence = 1.0 - (phase_entropy / max_entropy)
        
        # Property confidence (inverse of average uncertainty)
        avg_property_uncertainty = np.mean(list(property_uncertainties.values()))
        property_confidence = 1.0 / (1.0 + avg_property_uncertainty)
        
        # Physics confidence (based on supersaturation)
        physics_confidence = min(1.0, physics_features.supersaturation / 10.0)
        
        # Weighted combination
        weights = [0.5, 0.3, 0.2]  # Phase, property, physics
        confidences = [phase_confidence, property_confidence, physics_confidence]
        
        overall_confidence = np.average(confidences, weights=weights)
        return max(0.0, min(1.0, overall_confidence))
    
    def _validate_physics_consistency(self, 
                                    nn_outputs: Dict[str, Any],
                                    physics_features: PhysicsFeatures) -> float:
        """Validate that predictions are consistent with physics"""
        violations = 0
        total_checks = 0
        
        # Check phase probabilities sum to 1
        phase_probs = nn_outputs['phase']['probabilities'].cpu().numpy().flatten()
        prob_sum = np.sum(phase_probs)
        if abs(prob_sum - 1.0) > 0.01:
            violations += 1
        total_checks += 1
        
        # Check property ranges
        for prop_name, prop_data in nn_outputs['properties'].items():
            prop_value = float(prop_data['mean'].cpu().item())
            total_checks += 1
            
            if prop_name == 'bandgap' and not (0.5 <= prop_value <= 5.0):
                violations += 1
            elif prop_name == 'plqy' and not (0.0 <= prop_value <= 1.0):
                violations += 1
            elif prop_name == 'particle_size' and not (1.0 <= prop_value <= 1000.0):
                violations += 1
        
        # Physics consistency with supersaturation
        if physics_features.supersaturation < 1.0:
            # Should predict failed synthesis or very low phase probabilities
            max_phase_prob = np.max(phase_probs[:-1])  # Exclude failed synthesis
            if max_phase_prob > 0.5:
                violations += 1
        total_checks += 1
        
        consistency_score = 1.0 - (violations / total_checks) if total_checks > 0 else 1.0
        return consistency_score
    
    def _create_failed_prediction(self, prediction_id: str, error_msg: str, duration: float) -> PredictionResult:
        """Create a failed prediction result"""
        failed_properties = MaterialProperties(
            bandgap=0.0, plqy=0.0, emission_peak=0.0, emission_fwhm=0.0,
            particle_size=0.0, size_distribution_width=0.0, lifetime=0.0,
            stability_score=0.0, phase_purity=0.0
        )
        
        failed_physics = PhysicsFeatures(
            supersaturation=1.0, nucleation_rate_3d=0.0, nucleation_rate_0d=0.0,
            nucleation_rate_2d=0.0, critical_radius_3d=0.0, critical_radius_0d=0.0,
            critical_radius_2d=0.0, growth_rate=0.0, diffusion_length=0.0,
            ligand_coverage=0.0, gibbs_energy_3d=0.0, gibbs_energy_0d=0.0,
            gibbs_energy_2d=0.0, thermal_energy=0.0, ionic_strength=0.0
        )
        
        return PredictionResult(
            primary_phase=PhaseType.FAILED_SYNTHESIS,
            phase_probabilities={phase: 0.0 for phase in PhaseType},
            properties=failed_properties,
            phase_uncertainty=1.0,
            property_uncertainties={},
            physics_features=failed_physics,
            physics_consistency=0.0,
            confidence=0.0,
            prediction_id=prediction_id,
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=duration * 1000,
            model_version="failed"
        )
    
    def batch_predict(self, 
                     parameters_list: List[Union[SynthesisParameters, Dict[str, float]]],
                     batch_size: int = 32) -> List[PredictionResult]:
        """
        Batch prediction for multiple parameter sets
        
        Args:
            parameters_list: List of synthesis parameters
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        logger.info(f"Starting batch prediction for {len(parameters_list)} parameter sets")
        
        results = []
        for i in range(0, len(parameters_list), batch_size):
            batch = parameters_list[i:i+batch_size]
            batch_results = []
            
            for params in batch:
                try:
                    result = self.predict(params)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Batch prediction failed for sample {i}: {e}")
                    failed_result = self._create_failed_prediction(
                        str(uuid.uuid4()), str(e), 0.0
                    )
                    batch_results.append(failed_result)
            
            results.extend(batch_results)
            
            if i + batch_size < len(parameters_list):
                logger.info(f"Processed {i + batch_size}/{len(parameters_list)} predictions")
        
        successful_count = sum(1 for r in results if r.primary_phase != PhaseType.FAILED_SYNTHESIS)
        logger.info(f"Batch prediction completed: {successful_count}/{len(parameters_list)} successful")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get digital twin performance statistics"""
        uptime = time.time() - self.start_time
        avg_processing_time = self.total_processing_time / self.prediction_count if self.prediction_count > 0 else 0
        
        return {
            'total_predictions': self.prediction_count,
            'uptime_seconds': uptime,
            'average_processing_time_ms': avg_processing_time,
            'predictions_per_second': self.prediction_count / uptime if uptime > 0 else 0,
            'model_loaded': self.neural_network is not None,
            'device': str(self.device),
            'physics_models_loaded': {
                'nucleation': self.nucleation_model is not None,
                'growth': self.growth_model is not None,
                'phase_selection': self.phase_selection_model is not None,
                'ligand_effects': self.ligand_model is not None
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check digital twin health status"""
        try:
            # Test with dummy parameters
            test_params = SynthesisParameters(
                cs_br_concentration=1.0,
                pb_br2_concentration=1.0,
                temperature=150.0,
                solvent_type=SolventType.DMSO
            )
            
            start_time = time.time()
            result = self.predict(test_params)
            processing_time = (time.time() - start_time) * 1000
            
            status = "healthy" if result.primary_phase != PhaseType.FAILED_SYNTHESIS else "degraded"
            
        except Exception as e:
            status = "unhealthy"
            processing_time = 0
            logger.error(f"Health check failed: {e}")
        
        return {
            'status': status,
            'response_time_ms': processing_time,
            'uptime_seconds': time.time() - self.start_time,
            **self.get_performance_stats()
        }


def create_digital_twin(config: Optional[Dict[str, Any]] = None,
                       model_path: Optional[str] = None,
                       device: str = "auto") -> CsPbBr3DigitalTwin:
    """Factory function to create configured digital twin"""
    return CsPbBr3DigitalTwin(config, model_path, device)
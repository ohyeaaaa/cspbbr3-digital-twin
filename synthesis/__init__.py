#!/usr/bin/env python3
"""
CsPbBr3 Digital Twin - Synthesis Package
Main module exposing all digital twin components for easy import
"""

# Version information
__version__ = "2.0.0"
__author__ = "CsPbBr3 Digital Twin Team"
__email__ = "digital-twin@example.com"

# Core digital twin class (conditional import for environments without PyTorch)
try:
    from .digital_twin import CsPbBr3DigitalTwin, create_digital_twin
except ImportError:
    CsPbBr3DigitalTwin = None
    create_digital_twin = None

# Schema definitions
from .schemas import (
    SynthesisParameters,
    MaterialProperties, 
    PhysicsFeatures,
    PredictionResult,
    PhaseType,
    SolventType,
    TrainingConfig,
    ModelMetrics,
    ExperimentConfig,
    PHASE_NAMES,
    SOLVENT_NAMES,
    PARAMETER_RANGES,
    PROPERTY_RANGES
)

# Physics models (conditional imports)
try:
    from .physics.nucleation import ClassicalNucleationTheory, create_nucleation_model
    from .physics.growth import BurtonCabreraFrankModel, create_growth_model
    from .physics.phase_selection import PhaseSelectionModel, create_phase_selection_model
    from .physics.ligand_effects import CompetitiveLigandBinding, create_ligand_model
except ImportError:
    ClassicalNucleationTheory = None
    BurtonCabreraFrankModel = None
    PhaseSelectionModel = None
    CompetitiveLigandBinding = None
    create_nucleation_model = None
    create_growth_model = None
    create_phase_selection_model = None
    create_ligand_model = None

# Neural network models (conditional imports)
try:
    from .training.pytorch_neural_models import (
        PhysicsInformedNeuralNetwork,
        EnsembleModel,
        ModelConfig,
        create_model,
        load_pretrained_model
    )
except ImportError:
    PhysicsInformedNeuralNetwork = None
    EnsembleModel = None
    ModelConfig = None
    create_model = None
    load_pretrained_model = None

# Feature engineering (conditional imports)
try:
    from .training.pytorch_feature_engineering import (
        PhysicsInformedFeatureEngineer,
        create_feature_engineer
    )
except ImportError:
    PhysicsInformedFeatureEngineer = None
    create_feature_engineer = None

# Data handling (conditional imports)
try:
    from .training.data_loaders import (
        SynthesisDataset,
        StratifiedSynthesisDataModule,
        create_data_module
    )
except ImportError:
    SynthesisDataset = None
    StratifiedSynthesisDataModule = None
    create_data_module = None

# Training callbacks (conditional imports)
try:
    from .training.callbacks import (
        PhysicsConsistencyCallback,
        UncertaintyCalibrationCallback,
        ModelComplexityCallback,
        create_callbacks
    )
except ImportError:
    PhysicsConsistencyCallback = None
    UncertaintyCalibrationCallback = None
    ModelComplexityCallback = None
    create_callbacks = None

# Utilities (conditional imports)
try:
    from .utils.validation_utils import (
        validate_synthesis_parameters,
        validate_physics_constraints,
        check_parameter_ranges,
        validate_prediction_result
    )
except ImportError:
    validate_synthesis_parameters = None
    validate_physics_constraints = None
    check_parameter_ranges = None
    validate_prediction_result = None

try:
    from .utils.conversion_utils import (
        UnitConverter,
        convert_temperature,
        convert_concentration,
        convert_time,
        normalize_features
    )
except ImportError:
    UnitConverter = None
    convert_temperature = None
    convert_concentration = None
    convert_time = None
    normalize_features = None

try:
    from .utils.logging_utils import (
        SynthesisLogger,
        setup_logging,
        log_experiment_config
    )
except ImportError:
    SynthesisLogger = None
    setup_logging = None
    log_experiment_config = None

try:
    from .utils.file_utils import (
        FileManager,
        save_experiment_data,
        load_experiment_data,
        create_experiment_directory
    )
except ImportError:
    FileManager = None
    save_experiment_data = None
    load_experiment_data = None
    create_experiment_directory = None

# Factory functions for easy instantiation
def create_complete_digital_twin(
    model_path: str = None,
    config: dict = None,
    device: str = "auto"
):
    """
    Create a fully configured digital twin with all components
    
    Args:
        model_path: Path to pretrained model (optional)
        config: Configuration dictionary (optional)
        device: Computing device ("auto", "cpu", "cuda")
        
    Returns:
        Configured CsPbBr3 Digital Twin instance or None if dependencies missing
    """
    if create_digital_twin is None:
        raise ImportError("PyTorch dependencies not available. Cannot create digital twin.")
    return create_digital_twin(config, model_path, device)


def create_training_pipeline(
    config = None,
    data_config: dict = None
) -> tuple:
    """
    Create complete training pipeline with model, data, and callbacks
    
    Args:
        config: Training configuration
        data_config: Data loading configuration
        
    Returns:
        Tuple of (model, data_module, callbacks) or raises ImportError
    """
    if create_model is None or create_data_module is None or create_callbacks is None:
        raise ImportError("PyTorch dependencies not available. Cannot create training pipeline.")
    
    # Create model
    model_config = config.__dict__ if config else {}
    model = create_model(model_config)
    
    # Create data module
    data_module = create_data_module(data_config or {})
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    return model, data_module, callbacks


def quick_prediction(
    cs_concentration: float,
    pb_concentration: float, 
    temperature: float,
    solvent: str = "DMSO",
    model_path: str = None
):
    """
    Quick prediction with minimal setup
    
    Args:
        cs_concentration: Cs-Br concentration (mol/L)
        pb_concentration: Pb-Br2 concentration (mol/L)
        temperature: Temperature (C)
        solvent: Solvent type
        model_path: Optional pretrained model path
        
    Returns:
        Prediction result or raises ImportError
    """
    if create_complete_digital_twin is None:
        raise ImportError("PyTorch dependencies not available. Cannot make predictions.")
    
    # Create digital twin
    twin = create_complete_digital_twin(model_path)
    
    # Create parameters
    params = SynthesisParameters(
        cs_br_concentration=cs_concentration,
        pb_br2_concentration=pb_concentration,
        temperature=temperature,
        solvent_type=SolventType.from_string(solvent)
    )
    
    # Make prediction
    return twin.predict(params)


# Package metadata
__all__ = [
    # Version info
    "__version__", "__author__", "__email__",
    
    # Main classes
    "CsPbBr3DigitalTwin", "create_digital_twin",
    
    # Schemas
    "SynthesisParameters", "MaterialProperties", "PhysicsFeatures", 
    "PredictionResult", "PhaseType", "SolventType", "TrainingConfig",
    "ModelMetrics", "ExperimentConfig",
    
    # Constants
    "PHASE_NAMES", "SOLVENT_NAMES", "PARAMETER_RANGES", "PROPERTY_RANGES",
    
    # Physics models
    "ClassicalNucleationTheory", "BurtonCabreraFrankModel", 
    "PhaseSelectionModel", "CompetitiveLigandBinding",
    "create_nucleation_model", "create_growth_model", 
    "create_phase_selection_model", "create_ligand_model",
    
    # Neural networks
    "PhysicsInformedNeuralNetwork", "EnsembleModel", "ModelConfig",
    "create_model", "load_pretrained_model",
    
    # Feature engineering
    "PhysicsInformedFeatureEngineer", "create_feature_engineer",
    
    # Data handling
    "SynthesisDataset", "StratifiedSynthesisDataModule", "create_data_module",
    
    # Training callbacks
    "PhysicsConsistencyCallback", "UncertaintyCalibrationCallback",
    "ModelComplexityCallback", "create_callbacks",
    
    # Utilities
    "validate_synthesis_parameters", "validate_physics_constraints",
    "check_parameter_ranges", "validate_prediction_result",
    "UnitConverter", "convert_temperature", "convert_concentration", 
    "convert_time", "normalize_features",
    "SynthesisLogger", "setup_logging", "log_experiment_config",
    "FileManager", "save_experiment_data", "load_experiment_data",
    "create_experiment_directory",
    
    # Factory functions
    "create_complete_digital_twin", "create_training_pipeline", "quick_prediction"
]
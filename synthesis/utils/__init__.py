#!/usr/bin/env python3
"""
Synthesis Utils Package
Utility functions for the CsPbBr3 Digital Twin synthesis module
"""

from .logging_utils import (
    SynthesisLogger,
    setup_synthesis_logger,
    log_synthesis_parameters,
    log_prediction_results
)

from .file_utils import (
    FileManager,
    save_synthesis_data,
    load_synthesis_data,
    export_results_to_csv,
    create_experiment_directory
)

from .validation_utils import (
    validate_synthesis_parameters,
    validate_physics_constraints,
    check_parameter_ranges,
    validate_prediction_result
)

from .conversion_utils import (
    UnitConverter,
    convert_temperature,
    convert_concentration,
    convert_time,
    normalize_features
)

# Simple wrapper functions for backward compatibility
def setup_logging(level: str = "INFO"):
    """Setup basic logging configuration"""
    import logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def save_experiment_data(output_dir: str, filename: str, data):
    """Save experiment data to file"""
    import json
    import os
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

__all__ = [
    'SynthesisLogger',
    'setup_synthesis_logger', 
    'log_synthesis_parameters',
    'log_prediction_results',
    'FileManager',
    'save_synthesis_data',
    'load_synthesis_data', 
    'export_results_to_csv',
    'create_experiment_directory',
    'validate_synthesis_parameters',
    'validate_physics_constraints',
    'check_parameter_ranges',
    'validate_prediction_result',
    'UnitConverter',
    'convert_temperature',
    'convert_concentration',
    'convert_time',
    'normalize_features',
    'setup_logging',
    'save_experiment_data'
]
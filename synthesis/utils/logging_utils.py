#!/usr/bin/env python3
"""
Logging Utilities for CsPbBr₃ Digital Twin
Structured logging for synthesis experiments and predictions
"""

import logging
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import torch
import numpy as np
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SynthesisExperiment:
    """Data class for synthesis experiment logging"""
    experiment_id: str
    timestamp: str
    parameters: Dict[str, Any]
    predictions: Dict[str, Any]
    actual_results: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class SynthesisLogger:
    """Advanced logging for synthesis experiments"""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 enable_file_logging: bool = True,
                 enable_json_logging: bool = True):
        """
        Initialize synthesis logger
        
        Args:
            log_dir: Directory for log files
            log_level: Logging level
            enable_file_logging: Whether to log to files
            enable_json_logging: Whether to create JSON experiment logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_file_logging = enable_file_logging
        self.enable_json_logging = enable_json_logging
        
        # Setup structured logging
        self.setup_logger(log_level)
        
        # Experiment tracking
        self.experiments = []
        self.current_experiment = None
        
    def setup_logger(self, log_level: str):
        """Setup structured logging configuration"""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        if self.enable_file_logging:
            file_handler = logging.FileHandler(
                self.log_dir / f"synthesis_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Set level
        logger.setLevel(getattr(logging, log_level.upper()))
    
    def start_experiment(self, 
                        experiment_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new experiment session
        
        Args:
            experiment_id: Unique experiment identifier
            metadata: Additional experiment metadata
            
        Returns:
            Experiment ID
        """
        if experiment_id is None:
            experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_experiment = SynthesisExperiment(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            parameters={},
            predictions={},
            metadata=metadata or {}
        )
        
        logger.info(f"Started experiment: {experiment_id}")
        return experiment_id
    
    def log_synthesis_parameters(self, parameters: Dict[str, Any]):
        """Log synthesis parameters for current experiment"""
        if self.current_experiment is None:
            self.start_experiment()
        
        # Convert tensors to JSON-serializable format
        serializable_params = self._serialize_data(parameters)
        
        self.current_experiment.parameters.update(serializable_params)
        
        logger.info(f"Synthesis parameters: {json.dumps(serializable_params, indent=2)}")
    
    def log_prediction_results(self, predictions: Dict[str, Any]):
        """Log model predictions for current experiment"""
        if self.current_experiment is None:
            logger.warning("No active experiment. Starting new one.")
            self.start_experiment()
        
        serializable_predictions = self._serialize_data(predictions)
        self.current_experiment.predictions.update(serializable_predictions)
        
        # Log summary
        if 'phase' in predictions and 'probabilities' in predictions['phase']:
            phase_probs = predictions['phase']['probabilities']
            if isinstance(phase_probs, torch.Tensor):
                phase_probs = phase_probs.detach().cpu().numpy()
            
            predicted_phase = np.argmax(phase_probs)
            confidence = np.max(phase_probs)
            
            logger.info(f"Predicted phase: {predicted_phase}, Confidence: {confidence:.3f}")
        
        if 'properties' in predictions:
            prop_summary = {}
            for prop_name, prop_data in predictions['properties'].items():
                if 'mean' in prop_data:
                    mean_val = prop_data['mean']
                    if isinstance(mean_val, torch.Tensor):
                        mean_val = mean_val.detach().cpu().numpy()
                    prop_summary[prop_name] = float(mean_val) if np.isscalar(mean_val) else mean_val.tolist()
            
            logger.info(f"Property predictions: {json.dumps(prop_summary, indent=2)}")
    
    def log_actual_results(self, results: Dict[str, Any]):
        """Log actual experimental results for comparison"""
        if self.current_experiment is None:
            logger.warning("No active experiment. Starting new one.")
            self.start_experiment()
        
        serializable_results = self._serialize_data(results)
        self.current_experiment.actual_results = serializable_results
        
        logger.info(f"Actual results: {json.dumps(serializable_results, indent=2)}")
    
    def end_experiment(self, save_to_file: bool = True) -> Optional[SynthesisExperiment]:
        """
        End current experiment and save data
        
        Args:
            save_to_file: Whether to save experiment to file
            
        Returns:
            Completed experiment data
        """
        if self.current_experiment is None:
            logger.warning("No active experiment to end")
            return None
        
        experiment = self.current_experiment
        self.experiments.append(experiment)
        
        if save_to_file and self.enable_json_logging:
            self._save_experiment_to_file(experiment)
        
        logger.info(f"Ended experiment: {experiment.experiment_id}")
        
        self.current_experiment = None
        return experiment
    
    def _serialize_data(self, data: Any) -> Any:
        """Convert tensors and other objects to JSON-serializable format"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().tolist()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {key: self._serialize_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._serialize_data(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        else:
            return data
    
    def _save_experiment_to_file(self, experiment: SynthesisExperiment):
        """Save experiment data to JSON file"""
        experiment_dict = asdict(experiment)
        
        filename = f"{experiment.experiment_id}.json"
        filepath = self.log_dir / "experiments" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(experiment_dict, f, indent=2, default=str)
        
        logger.debug(f"Saved experiment to {filepath}")
    
    def export_experiments_to_csv(self, 
                                 output_file: Optional[str] = None,
                                 include_predictions: bool = True,
                                 include_actual: bool = True) -> str:
        """
        Export all experiments to CSV format
        
        Args:
            output_file: Output CSV file path
            include_predictions: Whether to include prediction columns
            include_actual: Whether to include actual result columns
            
        Returns:
            Path to created CSV file
        """
        if output_file is None:
            output_file = self.log_dir / f"experiments_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.experiments:
            logger.warning("No experiments to export")
            return str(output_path)
        
        # Prepare data for CSV
        rows = []
        headers = set(['experiment_id', 'timestamp'])
        
        for exp in self.experiments:
            row = {
                'experiment_id': exp.experiment_id,
                'timestamp': exp.timestamp
            }
            
            # Add parameters
            for key, value in exp.parameters.items():
                param_key = f"param_{key}"
                row[param_key] = value
                headers.add(param_key)
            
            # Add predictions
            if include_predictions and exp.predictions:
                for key, value in exp.predictions.items():
                    pred_key = f"pred_{key}"
                    if isinstance(value, (list, dict)):
                        row[pred_key] = json.dumps(value)
                    else:
                        row[pred_key] = value
                    headers.add(pred_key)
            
            # Add actual results
            if include_actual and exp.actual_results:
                for key, value in exp.actual_results.items():
                    actual_key = f"actual_{key}"
                    if isinstance(value, (list, dict)):
                        row[actual_key] = json.dumps(value)
                    else:
                        row[actual_key] = value
                    headers.add(actual_key)
            
            rows.append(row)
        
        # Write CSV
        headers = sorted(list(headers))
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        
        logger.info(f"Exported {len(rows)} experiments to {output_path}")
        return str(output_path)
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all experiments"""
        if not self.experiments:
            return {"total_experiments": 0}
        
        summary = {
            "total_experiments": len(self.experiments),
            "date_range": {
                "earliest": min(exp.timestamp for exp in self.experiments),
                "latest": max(exp.timestamp for exp in self.experiments)
            }
        }
        
        # Parameter statistics
        all_params = {}
        for exp in self.experiments:
            for key, value in exp.parameters.items():
                if key not in all_params:
                    all_params[key] = []
                if isinstance(value, (int, float)):
                    all_params[key].append(value)
        
        param_stats = {}
        for param, values in all_params.items():
            if values:
                param_stats[param] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
        
        summary["parameter_statistics"] = param_stats
        return summary


def setup_synthesis_logger(config: Dict[str, Any]) -> SynthesisLogger:
    """
    Setup synthesis logger from configuration
    
    Args:
        config: Logger configuration
        
    Returns:
        Configured SynthesisLogger instance
    """
    return SynthesisLogger(
        log_dir=config.get('log_dir', 'logs'),
        log_level=config.get('log_level', 'INFO'),
        enable_file_logging=config.get('enable_file_logging', True),
        enable_json_logging=config.get('enable_json_logging', True)
    )


def log_synthesis_parameters(params: Dict[str, Any], 
                           logger_instance: Optional[SynthesisLogger] = None):
    """Convenience function to log synthesis parameters"""
    if logger_instance is None:
        logger_instance = SynthesisLogger()
    
    logger_instance.log_synthesis_parameters(params)


def log_prediction_results(predictions: Dict[str, Any],
                          logger_instance: Optional[SynthesisLogger] = None):
    """Convenience function to log prediction results"""
    if logger_instance is None:
        logger_instance = SynthesisLogger()
    
    logger_instance.log_prediction_results(predictions)
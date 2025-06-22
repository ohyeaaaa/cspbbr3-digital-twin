#!/usr/bin/env python3
"""
File Management Utilities for CsPbBr₃ Digital Twin
Data persistence, export/import, and file organization
"""

import pickle
import json
import csv
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
import zipfile
import shutil

logger = logging.getLogger(__name__)


class FileManager:
    """Centralized file management for synthesis data"""
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize file manager
        
        Args:
            base_dir: Base directory for all data files
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.subdirs = {
            'models': self.base_dir / 'models',
            'training': self.base_dir / 'training',
            'experiments': self.base_dir / 'experiments',
            'cache': self.base_dir / 'cache',
            'exports': self.base_dir / 'exports',
            'literature': self.base_dir / 'literature'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
    
    def save_synthesis_data(self, 
                           data: Dict[str, Any], 
                           filename: str,
                           format: str = 'pickle',
                           subdir: str = 'training') -> str:
        """
        Save synthesis data to file
        
        Args:
            data: Data to save
            filename: Output filename
            format: File format ('pickle', 'json', 'hdf5')
            subdir: Subdirectory to save in
            
        Returns:
            Path to saved file
        """
        if subdir not in self.subdirs:
            raise ValueError(f"Unknown subdirectory: {subdir}")
        
        save_path = self.subdirs[subdir] / filename
        
        try:
            if format == 'pickle':
                with open(save_path, 'wb') as f:
                    pickle.dump(data, f)
            
            elif format == 'json':
                # Convert numpy arrays and tensors to lists
                serializable_data = self._make_json_serializable(data)
                with open(save_path, 'w') as f:
                    json.dump(serializable_data, f, indent=2)
            
            elif format == 'hdf5':
                self._save_to_hdf5(data, save_path)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved data to {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to save data to {save_path}: {e}")
            raise
    
    def load_synthesis_data(self, 
                           filename: str,
                           format: str = 'pickle',
                           subdir: str = 'training') -> Dict[str, Any]:
        """
        Load synthesis data from file
        
        Args:
            filename: Input filename
            format: File format ('pickle', 'json', 'hdf5')
            subdir: Subdirectory to load from
            
        Returns:
            Loaded data
        """
        if subdir not in self.subdirs:
            raise ValueError(f"Unknown subdirectory: {subdir}")
        
        load_path = self.subdirs[subdir] / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"File not found: {load_path}")
        
        try:
            if format == 'pickle':
                with open(load_path, 'rb') as f:
                    data = pickle.load(f)
            
            elif format == 'json':
                with open(load_path, 'r') as f:
                    data = json.load(f)
            
            elif format == 'hdf5':
                data = self._load_from_hdf5(load_path)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Loaded data from {load_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data from {load_path}: {e}")
            raise
    
    def export_results_to_csv(self, 
                             results: List[Dict[str, Any]],
                             filename: str,
                             include_metadata: bool = True) -> str:
        """
        Export prediction results to CSV format
        
        Args:
            results: List of result dictionaries
            filename: Output CSV filename
            include_metadata: Whether to include metadata columns
            
        Returns:
            Path to exported CSV file
        """
        if not results:
            logger.warning("No results to export")
            return ""
        
        export_path = self.subdirs['exports'] / filename
        
        # Flatten nested dictionaries for CSV
        flattened_results = []
        for result in results:
            flattened = self._flatten_dict(result)
            flattened_results.append(flattened)
        
        # Get all unique keys
        all_keys = set()
        for result in flattened_results:
            all_keys.update(result.keys())
        
        # Sort keys for consistent column order
        sorted_keys = sorted(all_keys)
        
        try:
            with open(export_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=sorted_keys)
                writer.writeheader()
                writer.writerows(flattened_results)
            
            logger.info(f"Exported {len(results)} results to {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Failed to export CSV to {export_path}: {e}")
            raise
    
    def create_experiment_directory(self, 
                                  experiment_id: str,
                                  include_subdirs: bool = True) -> str:
        """
        Create directory structure for an experiment
        
        Args:
            experiment_id: Unique experiment identifier
            include_subdirs: Whether to create standard subdirectories
            
        Returns:
            Path to experiment directory
        """
        exp_dir = self.subdirs['experiments'] / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        if include_subdirs:
            subdirs = ['data', 'models', 'plots', 'logs', 'results']
            for subdir in subdirs:
                (exp_dir / subdir).mkdir(exist_ok=True)
        
        logger.info(f"Created experiment directory: {exp_dir}")
        return str(exp_dir)
    
    def backup_experiment(self, 
                         experiment_id: str,
                         backup_dir: Optional[str] = None) -> str:
        """
        Create backup archive of experiment
        
        Args:
            experiment_id: Experiment to backup
            backup_dir: Backup directory (default: backups/)
            
        Returns:
            Path to backup archive
        """
        exp_dir = self.subdirs['experiments'] / experiment_id
        
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
        
        if backup_dir is None:
            backup_dir = self.base_dir / 'backups'
        else:
            backup_dir = Path(backup_dir)
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped archive
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_name = f"{experiment_id}_backup_{timestamp}.zip"
        archive_path = backup_dir / archive_name
        
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in exp_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(exp_dir.parent)
                        zipf.write(file_path, arcname)
            
            logger.info(f"Created backup archive: {archive_path}")
            return str(archive_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def cleanup_old_files(self, 
                         days_old: int = 30,
                         subdirs: Optional[List[str]] = None,
                         dry_run: bool = True) -> List[str]:
        """
        Clean up old files from specified subdirectories
        
        Args:
            days_old: Files older than this many days
            subdirs: Subdirectories to clean (default: cache, exports)
            dry_run: If True, only list files without deleting
            
        Returns:
            List of files that were (or would be) deleted
        """
        if subdirs is None:
            subdirs = ['cache', 'exports']
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
        files_to_delete = []
        
        for subdir in subdirs:
            if subdir not in self.subdirs:
                logger.warning(f"Unknown subdirectory: {subdir}")
                continue
            
            subdir_path = self.subdirs[subdir]
            for file_path in subdir_path.rglob('*'):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    files_to_delete.append(str(file_path))
        
        if dry_run:
            logger.info(f"Found {len(files_to_delete)} files older than {days_old} days")
            return files_to_delete
        
        # Actually delete files
        deleted_files = []
        for file_path in files_to_delete:
            try:
                Path(file_path).unlink()
                deleted_files.append(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
        
        logger.info(f"Deleted {len(deleted_files)} old files")
        return deleted_files
    
    def _make_json_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().tolist()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {key: self._make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif hasattr(data, '__dict__'):
            # Handle dataclasses and custom objects
            return self._make_json_serializable(data.__dict__)
        else:
            return data
    
    def _save_to_hdf5(self, data: Dict[str, Any], filepath: Path):
        """Save data to HDF5 format"""
        with h5py.File(filepath, 'w') as f:
            self._write_to_hdf5_group(f, data)
    
    def _write_to_hdf5_group(self, group: h5py.Group, data: Dict[str, Any]):
        """Recursively write data to HDF5 group"""
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._write_to_hdf5_group(subgroup, value)
            elif isinstance(value, torch.Tensor):
                group.create_dataset(key, data=value.detach().cpu().numpy())
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
            elif isinstance(value, (list, tuple)):
                array_data = np.array(value)
                group.create_dataset(key, data=array_data)
            elif isinstance(value, (int, float, str)):
                group.attrs[key] = value
            else:
                # Convert to string as fallback
                group.attrs[key] = str(value)
    
    def _load_from_hdf5(self, filepath: Path) -> Dict[str, Any]:
        """Load data from HDF5 format"""
        data = {}
        with h5py.File(filepath, 'r') as f:
            self._read_from_hdf5_group(f, data)
        return data
    
    def _read_from_hdf5_group(self, group: h5py.Group, data: Dict[str, Any]):
        """Recursively read data from HDF5 group"""
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                data[key] = {}
                self._read_from_hdf5_group(item, data[key])
            elif isinstance(item, h5py.Dataset):
                data[key] = item[()]
        
        # Read attributes
        for key, value in group.attrs.items():
            data[key] = value
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)) and len(v) > 0 and not isinstance(v[0], (dict, list)):
                # Convert simple lists to string
                items.append((new_key, str(v)))
            elif isinstance(v, (torch.Tensor, np.ndarray)):
                # Convert arrays to string representation
                items.append((new_key, str(v.tolist() if hasattr(v, 'tolist') else v)))
            else:
                items.append((new_key, v))
        return dict(items)


# Convenience functions
def save_synthesis_data(data: Dict[str, Any], 
                       filename: str,
                       format: str = 'pickle',
                       base_dir: str = "data") -> str:
    """Save synthesis data using default file manager"""
    manager = FileManager(base_dir)
    return manager.save_synthesis_data(data, filename, format)


def load_synthesis_data(filename: str,
                       format: str = 'pickle',
                       base_dir: str = "data") -> Dict[str, Any]:
    """Load synthesis data using default file manager"""
    manager = FileManager(base_dir)
    return manager.load_synthesis_data(filename, format)


def export_results_to_csv(results: List[Dict[str, Any]],
                          filename: str,
                          base_dir: str = "data") -> str:
    """Export results to CSV using default file manager"""
    manager = FileManager(base_dir)
    return manager.export_results_to_csv(results, filename)


def create_experiment_directory(experiment_id: str,
                               base_dir: str = "data") -> str:
    """Create experiment directory using default file manager"""
    manager = FileManager(base_dir)
    return manager.create_experiment_directory(experiment_id)
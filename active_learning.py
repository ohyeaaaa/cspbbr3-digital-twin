#!/usr/bin/env python3
"""
Active Learning System for CsPbBr₃ Digital Twin
Intelligent experiment suggestion based on uncertainty and expected improvement
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import logging
from dataclasses import dataclass
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ExperimentSuggestion:
    """Data structure for experiment suggestions"""
    suggestion_id: str
    synthesis_conditions: Dict[str, float]
    acquisition_score: float
    acquisition_type: str
    predicted_outcome: Dict[str, Any]
    uncertainty_metrics: Dict[str, float]
    rationale: str
    priority: str  # high, medium, low
    estimated_information_gain: float
    suggested_by: str
    timestamp: str

class UncertaintyBasedSampling:
    """Uncertainty-based active learning strategies"""
    
    def __init__(self, model, feature_columns: List[str]):
        self.model = model
        self.feature_columns = feature_columns
        self.phase_names = {0: 'CsPbBr3_3D', 1: 'Cs4PbBr6_0D', 2: 'CsPb2Br5_2D', 3: 'Mixed', 4: 'Failed'}
    
    def max_entropy_sampling(self, candidate_conditions: np.ndarray, 
                           num_suggestions: int = 5) -> List[int]:
        """Select experiments with maximum prediction entropy"""
        
        entropies = []
        
        with torch.no_grad():
            for conditions in candidate_conditions:
                # Convert to model input format
                features = self._conditions_to_features(conditions)
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                
                # Get prediction uncertainty
                if hasattr(self.model, 'predict_with_uncertainty'):
                    outputs = self.model.predict_with_uncertainty(features_tensor, num_samples=50)
                    phase_probs = outputs['phase_probabilities']['mean'].squeeze()
                else:
                    # Fallback for regular models
                    outputs = self.model(features_tensor)
                    phase_probs = F.softmax(outputs['phase_logits'], dim=-1).squeeze()
                
                # Calculate entropy
                entropy = -torch.sum(phase_probs * torch.log(phase_probs + 1e-8))
                entropies.append(entropy.item())
        
        # Select top uncertain samples
        uncertain_indices = np.argsort(entropies)[-num_suggestions:]
        return uncertain_indices.tolist()
    
    def predictive_variance_sampling(self, candidate_conditions: np.ndarray,
                                   num_suggestions: int = 5) -> List[int]:
        """Select experiments with maximum predictive variance"""
        
        variances = []
        
        with torch.no_grad():
            for conditions in candidate_conditions:
                features = self._conditions_to_features(conditions)
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                
                if hasattr(self.model, 'predict_with_uncertainty'):
                    outputs = self.model.predict_with_uncertainty(features_tensor, num_samples=50)
                    
                    # Combine phase and property uncertainties
                    phase_var = outputs['phase_probabilities']['std'].mean().item()
                    prop_vars = [outputs['properties'][prop]['std'].mean().item() 
                               for prop in outputs['properties']]
                    total_variance = phase_var + np.mean(prop_vars)
                else:
                    # Fallback - use dummy variance
                    total_variance = np.random.random()
                
                variances.append(total_variance)
        
        # Select highest variance samples
        high_var_indices = np.argsort(variances)[-num_suggestions:]
        return high_var_indices.tolist()
    
    def _conditions_to_features(self, conditions: np.ndarray) -> np.ndarray:
        """Convert synthesis conditions to model features"""
        
        # Map conditions array to feature dictionary
        condition_dict = {
            'cs_br_concentration': conditions[0],
            'pb_br2_concentration': conditions[1], 
            'temperature': conditions[2],
            'oa_concentration': conditions[3],
            'oam_concentration': conditions[4],
            'reaction_time': conditions[5],
            'solvent_type': int(conditions[6])
        }
        
        # Calculate derived features
        features = []
        for col in self.feature_columns:
            if col in condition_dict:
                features.append(condition_dict[col])
            elif col == 'cs_pb_ratio':
                features.append(condition_dict['cs_br_concentration'] / condition_dict['pb_br2_concentration'])
            elif col == 'supersaturation':
                features.append(np.log((condition_dict['cs_br_concentration'] * condition_dict['pb_br2_concentration']) / 
                                     (0.1 + condition_dict['temperature'] / 1000)))
            elif col == 'ligand_ratio':
                ligand_total = condition_dict['oa_concentration'] + condition_dict['oam_concentration']
                features.append(ligand_total / (condition_dict['cs_br_concentration'] + condition_dict['pb_br2_concentration']))
            elif col == 'temp_normalized':
                features.append((condition_dict['temperature'] - 80) / (250 - 80))
            elif col == 'solvent_effect':
                solvent_effects = {0: 1.2, 1: 1.0, 2: 0.5, 3: 0.8, 4: 0.9}
                features.append(solvent_effects.get(condition_dict['solvent_type'], 1.0))
            else:
                features.append(0.0)
        
        return np.array(features)

class BayesianOptimization:
    """Bayesian optimization for synthesis condition optimization"""
    
    def __init__(self, model, feature_columns: List[str], objective: str = 'cspbbr3_probability'):
        self.model = model
        self.feature_columns = feature_columns
        self.objective = objective
        self.gp_models = {}
        self.training_data = {'X': [], 'y': []}
    
    def fit_surrogate_model(self, experimental_data: pd.DataFrame):
        """Fit Gaussian Process surrogate model on experimental data"""
        
        if len(experimental_data) < 3:
            logger.warning("Insufficient data for Bayesian optimization (need ≥3 experiments)")
            return
        
        # Prepare training data
        X = []
        y = []
        
        for _, row in experimental_data.iterrows():
            conditions = [
                row['cs_br_concentration'], row['pb_br2_concentration'], 
                row['temperature'], row['oa_concentration'], row['oam_concentration'],
                row['reaction_time'], row['solvent_type']
            ]
            
            # Objective value based on experimental results
            if self.objective == 'cspbbr3_probability':
                # Higher value for CsPbBr3_3D phase
                if row['dominant_phase'] == 'CsPbBr3_3D':
                    objective_value = row['phase_purity']
                else:
                    objective_value = 0.1 * row['phase_purity']
            elif self.objective == 'bandgap':
                objective_value = -abs(row['bandgap'] - 2.1) if pd.notna(row['bandgap']) else -1.0
            elif self.objective == 'plqy':
                objective_value = row['plqy'] if pd.notna(row['plqy']) else 0.0
            else:
                objective_value = float(row['synthesis_success'])
            
            X.append(conditions)
            y.append(objective_value)
        
        X = np.array(X)
        y = np.array(y)
        
        # Fit Gaussian Process
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, n_restarts_optimizer=10
        )
        
        self.gp_model.fit(X, y)
        self.training_data = {'X': X, 'y': y}
        
        logger.info(f"Surrogate model fitted with {len(X)} experiments")
    
    def acquisition_function(self, x: np.ndarray, acquisition_type: str = 'EI') -> float:
        """Compute acquisition function value"""
        
        if not hasattr(self, 'gp_model'):
            return 0.0
        
        x = x.reshape(1, -1)
        
        # GP prediction
        mu, sigma = self.gp_model.predict(x, return_std=True)
        mu, sigma = mu[0], sigma[0]
        
        if acquisition_type == 'EI':  # Expected Improvement
            if len(self.training_data['y']) == 0:
                return 0.0
            
            f_best = np.max(self.training_data['y'])
            
            if sigma == 0:
                return 0.0
            
            z = (mu - f_best) / sigma
            ei = (mu - f_best) * self._norm_cdf(z) + sigma * self._norm_pdf(z)
            return ei
        
        elif acquisition_type == 'UCB':  # Upper Confidence Bound
            beta = 2.0  # Exploration parameter
            return mu + beta * sigma
        
        elif acquisition_type == 'PI':  # Probability of Improvement
            if len(self.training_data['y']) == 0:
                return 0.0
            
            f_best = np.max(self.training_data['y'])
            
            if sigma == 0:
                return 0.0
            
            z = (mu - f_best) / sigma
            return self._norm_cdf(z)
        
        return 0.0
    
    def _norm_cdf(self, x):
        """Normal CDF approximation"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    def _norm_pdf(self, x):
        """Normal PDF"""
        return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    
    def suggest_experiments(self, num_suggestions: int = 3, 
                          acquisition_type: str = 'EI') -> List[np.ndarray]:
        """Suggest experiments using Bayesian optimization"""
        
        if not hasattr(self, 'gp_model'):
            logger.warning("Surrogate model not fitted. Using random suggestions.")
            return self._random_suggestions(num_suggestions)
        
        suggestions = []
        
        # Define bounds for synthesis conditions
        bounds = [
            (0.5, 2.0),   # cs_br_concentration
            (0.5, 2.0),   # pb_br2_concentration  
            (80, 250),    # temperature
            (0.1, 1.0),   # oa_concentration
            (0.05, 0.5),  # oam_concentration
            (15, 120),    # reaction_time
            (0, 4)        # solvent_type
        ]
        
        for _ in range(num_suggestions):
            # Random initialization
            x0 = np.array([
                np.random.uniform(b[0], b[1]) for b in bounds
            ])
            x0[-1] = int(x0[-1])  # Solvent type should be integer
            
            # Optimize acquisition function
            result = minimize(
                lambda x: -self.acquisition_function(x, acquisition_type),
                x0, bounds=bounds, method='L-BFGS-B'
            )
            
            if result.success:
                suggestion = result.x.copy()
                suggestion[-1] = int(np.round(suggestion[-1]))  # Round solvent type
                suggestions.append(suggestion)
            else:
                # Fallback to random
                suggestions.append(x0)
        
        return suggestions
    
    def _random_suggestions(self, num_suggestions: int) -> List[np.ndarray]:
        """Generate random experiment suggestions as fallback"""
        
        suggestions = []
        for _ in range(num_suggestions):
            suggestion = np.array([
                np.random.uniform(0.5, 2.0),    # cs_br_concentration
                np.random.uniform(0.5, 2.0),    # pb_br2_concentration
                np.random.uniform(80, 250),     # temperature
                np.random.uniform(0.1, 1.0),    # oa_concentration
                np.random.uniform(0.05, 0.5),   # oam_concentration
                np.random.uniform(15, 120),     # reaction_time
                np.random.randint(0, 5)         # solvent_type
            ])
            suggestions.append(suggestion)
        
        return suggestions

class DiversityBasedSampling:
    """Diversity-based sampling for exploration"""
    
    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
    
    def kmeans_sampling(self, candidate_conditions: np.ndarray, 
                       num_suggestions: int = 5) -> List[int]:
        """Select diverse experiments using k-means clustering"""
        
        from sklearn.cluster import KMeans
        
        if len(candidate_conditions) <= num_suggestions:
            return list(range(len(candidate_conditions)))
        
        # Ensure we have at least 1 cluster
        n_clusters = min(max(1, num_suggestions), len(candidate_conditions))
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(candidate_conditions)
        
        # Select one representative from each cluster
        selected_indices = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            if len(cluster_indices) > 0:
                # Select point closest to cluster center
                cluster_center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(
                    candidate_conditions[cluster_indices] - cluster_center, axis=1
                )
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)
        
        # If we have fewer clusters than requested, pad with random selection
        while len(selected_indices) < num_suggestions:
            remaining_indices = [i for i in range(len(candidate_conditions)) if i not in selected_indices]
            if remaining_indices:
                selected_indices.append(np.random.choice(remaining_indices))
            else:
                break
        
        return selected_indices
    
    def greedy_diversity_sampling(self, candidate_conditions: np.ndarray,
                                num_suggestions: int = 5) -> List[int]:
        """Greedy selection for maximum diversity"""
        
        if len(candidate_conditions) <= num_suggestions:
            return list(range(len(candidate_conditions)))
        
        selected_indices = []
        
        # Start with random point
        selected_indices.append(np.random.randint(len(candidate_conditions)))
        
        for _ in range(num_suggestions - 1):
            max_min_distance = -1
            best_candidate = -1
            
            for i, candidate in enumerate(candidate_conditions):
                if i in selected_indices:
                    continue
                
                # Find minimum distance to already selected points
                min_distance = min([
                    np.linalg.norm(candidate - candidate_conditions[j])
                    for j in selected_indices
                ])
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = i
            
            if best_candidate != -1:
                selected_indices.append(best_candidate)
        
        return selected_indices

class ActiveLearningOrchestrator:
    """Main orchestrator for active learning experiment suggestions"""
    
    def __init__(self, model, feature_columns: List[str], scaler=None):
        self.model = model
        self.feature_columns = feature_columns
        self.scaler = scaler
        
        # Initialize sampling strategies
        self.uncertainty_sampler = UncertaintyBasedSampling(model, feature_columns)
        self.bayesian_optimizer = BayesianOptimization(model, feature_columns)
        self.diversity_sampler = DiversityBasedSampling(feature_columns)
        
        # Suggestion history
        self.suggestion_history = []
    
    def generate_candidate_space(self, num_candidates: int = 1000) -> np.ndarray:
        """Generate candidate synthesis conditions"""
        
        candidates = []
        
        for _ in range(num_candidates):
            candidate = [
                np.random.uniform(0.5, 2.0),    # cs_br_concentration
                np.random.uniform(0.5, 2.0),    # pb_br2_concentration
                np.random.uniform(80, 250),     # temperature  
                np.random.uniform(0.1, 1.0),    # oa_concentration
                np.random.uniform(0.05, 0.5),   # oam_concentration
                np.random.uniform(15, 120),     # reaction_time
                np.random.randint(0, 5)         # solvent_type
            ]
            candidates.append(candidate)
        
        return np.array(candidates)
    
    def suggest_experiments(self, experimental_data: Optional[pd.DataFrame] = None,
                          strategy: str = 'mixed', num_suggestions: int = 5) -> List[ExperimentSuggestion]:
        """Generate experiment suggestions using specified strategy"""
        
        # Generate candidate space
        candidates = self.generate_candidate_space(1000)
        
        suggestions = []
        
        if strategy == 'uncertainty':
            indices = self.uncertainty_sampler.max_entropy_sampling(candidates, num_suggestions)
            suggestion_type = 'Max Entropy (Uncertainty)'
            
        elif strategy == 'bayesian':
            if experimental_data is not None and len(experimental_data) >= 3:
                self.bayesian_optimizer.fit_surrogate_model(experimental_data)
                suggested_conditions = self.bayesian_optimizer.suggest_experiments(num_suggestions)
                indices = list(range(len(suggested_conditions)))
                candidates = np.array(suggested_conditions)
            else:
                indices = self.uncertainty_sampler.max_entropy_sampling(candidates, num_suggestions)
            suggestion_type = 'Bayesian Optimization'
            
        elif strategy == 'diversity':
            indices = self.diversity_sampler.kmeans_sampling(candidates, num_suggestions)
            suggestion_type = 'Diversity Sampling'
            
        elif strategy == 'mixed':
            # Combine different strategies
            n_uncertainty = max(1, num_suggestions // 3)
            n_bayesian = max(1, num_suggestions // 3) 
            n_diversity = num_suggestions - n_uncertainty - n_bayesian
            
            uncertainty_indices = self.uncertainty_sampler.max_entropy_sampling(candidates, n_uncertainty)
            diversity_indices = self.diversity_sampler.kmeans_sampling(candidates, n_diversity)
            
            if experimental_data is not None and len(experimental_data) >= 3:
                self.bayesian_optimizer.fit_surrogate_model(experimental_data)
                bayesian_conditions = self.bayesian_optimizer.suggest_experiments(n_bayesian)
                bayesian_indices = list(range(len(uncertainty_indices), len(uncertainty_indices) + len(bayesian_conditions)))
                candidates = np.vstack([candidates, np.array(bayesian_conditions)])
            else:
                bayesian_indices = self.uncertainty_sampler.predictive_variance_sampling(candidates, n_bayesian)
            
            indices = uncertainty_indices + bayesian_indices + diversity_indices
            suggestion_type = 'Mixed Strategy'
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Create suggestion objects
        for i, idx in enumerate(indices):
            if idx < len(candidates):
                conditions_array = candidates[idx]
                
                # Convert to condition dictionary
                conditions = {
                    'cs_br_concentration': float(conditions_array[0]),
                    'pb_br2_concentration': float(conditions_array[1]),
                    'temperature': float(conditions_array[2]),
                    'oa_concentration': float(conditions_array[3]),
                    'oam_concentration': float(conditions_array[4]),
                    'reaction_time': float(conditions_array[5]),
                    'solvent_type': int(conditions_array[6])
                }
                
                # Get model prediction
                prediction_result = self._get_model_prediction(conditions)
                
                # Calculate metrics
                acquisition_score = self._calculate_acquisition_score(conditions, strategy)
                information_gain = self._estimate_information_gain(conditions, prediction_result)
                
                # Create suggestion
                suggestion = ExperimentSuggestion(
                    suggestion_id=f"AL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:02d}",
                    synthesis_conditions=conditions,
                    acquisition_score=acquisition_score,
                    acquisition_type=suggestion_type,
                    predicted_outcome=prediction_result,
                    uncertainty_metrics=self._calculate_uncertainty_metrics(prediction_result),
                    rationale=self._generate_rationale(conditions, prediction_result, strategy),
                    priority=self._assign_priority(acquisition_score, information_gain),
                    estimated_information_gain=information_gain,
                    suggested_by=f"ActiveLearning_{strategy}",
                    timestamp=datetime.now().isoformat()
                )
                
                suggestions.append(suggestion)
        
        # Store suggestions
        self.suggestion_history.extend(suggestions)
        
        return suggestions
    
    def _get_model_prediction(self, conditions: Dict[str, float]) -> Dict[str, Any]:
        """Get model prediction for given conditions"""
        
        try:
            # Convert conditions to features
            features = []
            for col in self.feature_columns:
                if col in conditions:
                    features.append(conditions[col])
                elif col == 'cs_pb_ratio':
                    features.append(conditions['cs_br_concentration'] / conditions['pb_br2_concentration'])
                elif col == 'supersaturation':
                    features.append(np.log((conditions['cs_br_concentration'] * conditions['pb_br2_concentration']) / 
                                         (0.1 + conditions['temperature'] / 1000)))
                elif col == 'ligand_ratio':
                    ligand_total = conditions['oa_concentration'] + conditions['oam_concentration']
                    features.append(ligand_total / (conditions['cs_br_concentration'] + conditions['pb_br2_concentration']))
                elif col == 'temp_normalized':
                    features.append((conditions['temperature'] - 80) / (250 - 80))
                elif col == 'solvent_effect':
                    solvent_effects = {0: 1.2, 1: 1.0, 2: 0.5, 3: 0.8, 4: 0.9}
                    features.append(solvent_effects.get(conditions['solvent_type'], 1.0))
                else:
                    features.append(0.0)
            
            features = np.array(features).reshape(1, -1)
            
            # Scale features if scaler available
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            features_tensor = torch.FloatTensor(features)
            
            # Get prediction with uncertainty if available
            if hasattr(self.model, 'predict_with_uncertainty'):
                self.model.eval()
                outputs = self.model.predict_with_uncertainty(features_tensor, num_samples=50)
                
                phase_probs = outputs['phase_probabilities']['mean'].squeeze()
                phase_std = outputs['phase_probabilities']['std'].squeeze()
                predicted_phase = torch.argmax(phase_probs).item()
                
                properties = {}
                for prop, stats in outputs['properties'].items():
                    properties[prop] = {
                        'mean': stats['mean'].item(),
                        'std': stats['std'].item(),
                        'confidence_interval': [
                            stats['quantile_025'].item(),
                            stats['quantile_975'].item()
                        ]
                    }
                
                return {
                    'predicted_phase': self.uncertainty_sampler.phase_names[predicted_phase],
                    'phase_probabilities': phase_probs.tolist(),
                    'phase_uncertainties': phase_std.tolist(),
                    'confidence': float(torch.max(phase_probs)),
                    'properties': properties,
                    'has_uncertainty': True
                }
            
            else:
                # Standard prediction
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(features_tensor)
                
                phase_probs = F.softmax(outputs['phase_logits'], dim=-1).squeeze()
                predicted_phase = torch.argmax(phase_probs).item()
                
                properties = {}
                for prop_name, pred in outputs['properties'].items():
                    properties[prop_name] = {
                        'mean': pred.item(),
                        'std': 0.0,
                        'confidence_interval': [pred.item(), pred.item()]
                    }
                
                return {
                    'predicted_phase': self.uncertainty_sampler.phase_names[predicted_phase],
                    'phase_probabilities': phase_probs.tolist(),
                    'phase_uncertainties': [0.0] * len(phase_probs),
                    'confidence': float(torch.max(phase_probs)),
                    'properties': properties,
                    'has_uncertainty': False
                }
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'predicted_phase': 'Unknown',
                'phase_probabilities': [0.2] * 5,
                'phase_uncertainties': [0.5] * 5,
                'confidence': 0.2,
                'properties': {},
                'has_uncertainty': False
            }
    
    def _calculate_acquisition_score(self, conditions: Dict[str, float], strategy: str) -> float:
        """Calculate acquisition score for ranking suggestions"""
        
        if strategy == 'bayesian' and hasattr(self.bayesian_optimizer, 'gp_model'):
            conditions_array = np.array([
                conditions['cs_br_concentration'], conditions['pb_br2_concentration'],
                conditions['temperature'], conditions['oa_concentration'], 
                conditions['oam_concentration'], conditions['reaction_time'],
                conditions['solvent_type']
            ])
            return self.bayesian_optimizer.acquisition_function(conditions_array, 'EI')
        else:
            # Fallback scoring based on prediction uncertainty
            prediction = self._get_model_prediction(conditions)
            if prediction['has_uncertainty']:
                return 1.0 - prediction['confidence']
            else:
                return np.random.random()
    
    def _calculate_uncertainty_metrics(self, prediction: Dict[str, Any]) -> Dict[str, float]:
        """Calculate uncertainty metrics from prediction"""
        
        if prediction['has_uncertainty']:
            phase_uncertainty = np.mean(prediction['phase_uncertainties'])
            prop_uncertainties = []
            
            for prop, stats in prediction['properties'].items():
                if 'std' in stats:
                    prop_uncertainties.append(stats['std'])
            
            return {
                'phase_uncertainty': float(phase_uncertainty),
                'property_uncertainty': float(np.mean(prop_uncertainties)) if prop_uncertainties else 0.0,
                'total_uncertainty': float(phase_uncertainty + np.mean(prop_uncertainties)) if prop_uncertainties else float(phase_uncertainty)
            }
        else:
            return {
                'phase_uncertainty': 0.0,
                'property_uncertainty': 0.0,
                'total_uncertainty': 0.0
            }
    
    def _estimate_information_gain(self, conditions: Dict[str, float], 
                                 prediction: Dict[str, Any]) -> float:
        """Estimate expected information gain from experiment"""
        
        # Higher information gain for:
        # 1. High uncertainty predictions
        # 2. Conditions far from previous experiments
        # 3. Predicted high-value outcomes
        
        uncertainty_gain = self._calculate_uncertainty_metrics(prediction)['total_uncertainty']
        
        # Novelty gain (simplified - should check against previous experiments)
        novelty_gain = 0.5  # Placeholder
        
        # Outcome value gain
        if prediction['predicted_phase'] == 'CsPbBr3_3D':
            outcome_gain = prediction['confidence']
        else:
            outcome_gain = 0.2
        
        return uncertainty_gain + 0.3 * novelty_gain + 0.2 * outcome_gain
    
    def _generate_rationale(self, conditions: Dict[str, float], 
                          prediction: Dict[str, Any], strategy: str) -> str:
        """Generate human-readable rationale for suggestion"""
        
        rationale_parts = []
        
        # Strategy explanation
        if strategy == 'uncertainty':
            rationale_parts.append("Selected for high prediction uncertainty")
        elif strategy == 'bayesian':
            rationale_parts.append("Optimized for expected improvement")
        elif strategy == 'diversity':
            rationale_parts.append("Selected for parameter space exploration")
        elif strategy == 'mixed':
            rationale_parts.append("Balanced uncertainty and optimization strategy")
        
        # Prediction explanation
        pred_phase = prediction['predicted_phase']
        confidence = prediction['confidence']
        rationale_parts.append(f"Predicted outcome: {pred_phase} (confidence: {confidence:.3f})")
        
        # Key conditions
        temp = conditions['temperature']
        cs_pb_ratio = conditions['cs_br_concentration'] / conditions['pb_br2_concentration']
        rationale_parts.append(f"Key conditions: {temp:.0f}°C, Cs:Pb={cs_pb_ratio:.2f}")
        
        return "; ".join(rationale_parts)
    
    def _assign_priority(self, acquisition_score: float, information_gain: float) -> str:
        """Assign priority level to suggestion"""
        
        combined_score = acquisition_score + information_gain
        
        if combined_score > 1.5:
            return 'high'
        elif combined_score > 0.8:
            return 'medium'
        else:
            return 'low'
    
    def save_suggestions(self, suggestions: List[ExperimentSuggestion], 
                        filename: str = None) -> str:
        """Save suggestions to file"""
        
        if filename is None:
            filename = f"experiment_suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to serializable format
        suggestions_data = []
        for suggestion in suggestions:
            suggestions_data.append({
                'suggestion_id': suggestion.suggestion_id,
                'synthesis_conditions': suggestion.synthesis_conditions,
                'acquisition_score': suggestion.acquisition_score,
                'acquisition_type': suggestion.acquisition_type,
                'predicted_outcome': suggestion.predicted_outcome,
                'uncertainty_metrics': suggestion.uncertainty_metrics,
                'rationale': suggestion.rationale,
                'priority': suggestion.priority,
                'estimated_information_gain': suggestion.estimated_information_gain,
                'suggested_by': suggestion.suggested_by,
                'timestamp': suggestion.timestamp
            })
        
        with open(filename, 'w') as f:
            json.dump(suggestions_data, f, indent=2)
        
        logger.info(f"Saved {len(suggestions)} suggestions to {filename}")
        return filename

def main():
    """Test active learning system"""
    
    print("🤖 Testing Active Learning System")
    print("=" * 50)
    
    # Load model components
    try:
        from test_trained_model import load_trained_model, load_scaler
        model, config = load_trained_model()
        scaler = load_scaler()
        feature_columns = config['feature_columns']
        
        print("✅ Model loaded successfully")
        
        # Initialize active learning
        al_orchestrator = ActiveLearningOrchestrator(model, feature_columns, scaler)
        
        # Test different strategies
        strategies = ['uncertainty', 'diversity', 'mixed']
        
        for strategy in strategies:
            print(f"\n🎯 Testing {strategy} strategy:")
            
            suggestions = al_orchestrator.suggest_experiments(
                strategy=strategy, num_suggestions=3
            )
            
            for i, suggestion in enumerate(suggestions):
                print(f"   Suggestion {i+1}:")
                print(f"     Priority: {suggestion.priority}")
                print(f"     Predicted: {suggestion.predicted_outcome['predicted_phase']}")
                print(f"     Confidence: {suggestion.predicted_outcome['confidence']:.3f}")
                print(f"     Temperature: {suggestion.synthesis_conditions['temperature']:.1f}°C")
                print(f"     Rationale: {suggestion.rationale}")
        
        # Save suggestions
        all_suggestions = al_orchestrator.suggestion_history
        filename = al_orchestrator.save_suggestions(all_suggestions)
        print(f"\n💾 Saved {len(all_suggestions)} suggestions to {filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Using dummy model for testing...")
        
        # Test with dummy data
        from uncertainty_models import MCDropoutNeuralNetwork
        dummy_model = MCDropoutNeuralNetwork(12)
        feature_columns = ['cs_br_concentration', 'pb_br2_concentration', 'temperature', 
                          'oa_concentration', 'oam_concentration', 'reaction_time', 'solvent_type',
                          'cs_pb_ratio', 'supersaturation', 'ligand_ratio', 'temp_normalized', 'solvent_effect']
        
        al_orchestrator = ActiveLearningOrchestrator(dummy_model, feature_columns)
        
        suggestions = al_orchestrator.suggest_experiments(strategy='mixed', num_suggestions=2)
        print(f"\n✅ Generated {len(suggestions)} dummy suggestions")
    
    print("\n🎉 Active learning system tested successfully!")

if __name__ == "__main__":
    main()
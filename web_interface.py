#!/usr/bin/env python3
"""
Web Interface for CsPbBr₃ Digital Twin
Flask-based dashboard for model interaction and experiment management
"""

from flask import Flask, render_template, request, jsonify, send_file
import torch
import numpy as np
import pandas as pd
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import our modules
from validation_pipeline import ValidationPipeline
from active_learning import ActiveLearningOrchestrator
from uncertainty_models import MCDropoutNeuralNetwork
from experimental_validation import ExperimentalValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cspbbr3-digital-twin-secret-key'

# Global variables for model components
model = None
scaler = None
feature_columns = None
validation_pipeline = None
active_learner = None

def initialize_system():
    """Initialize the digital twin system"""
    global model, scaler, feature_columns, validation_pipeline, active_learner
    
    try:
        # Load model components
        from test_trained_model import load_trained_model, load_scaler
        model, config = load_trained_model()
        scaler = load_scaler()
        feature_columns = config['feature_columns']
        
        # Initialize pipeline and active learning
        validation_pipeline = ValidationPipeline("web_experimental_data")
        active_learner = ActiveLearningOrchestrator(model, feature_columns, scaler)
        
        logger.info("✅ Digital twin system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        
        # Fallback to dummy system
        logger.info("Using dummy system for demonstration")
        model = MCDropoutNeuralNetwork(12)
        feature_columns = [
            'cs_br_concentration', 'pb_br2_concentration', 'temperature', 
            'oa_concentration', 'oam_concentration', 'reaction_time', 'solvent_type',
            'cs_pb_ratio', 'supersaturation', 'ligand_ratio', 'temp_normalized', 'solvent_effect'
        ]
        validation_pipeline = ValidationPipeline("web_experimental_data")
        active_learner = ActiveLearningOrchestrator(model, feature_columns)
        return False

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make synthesis predictions"""
    try:
        data = request.json
        
        # Extract synthesis conditions
        conditions = {
            'cs_br_concentration': float(data.get('cs_concentration', 1.1)),
            'pb_br2_concentration': float(data.get('pb_concentration', 1.1)),
            'temperature': float(data.get('temperature', 190)),
            'oa_concentration': float(data.get('oa_concentration', 0.4)),
            'oam_concentration': float(data.get('oam_concentration', 0.2)),
            'reaction_time': float(data.get('reaction_time', 75)),
            'solvent_type': int(data.get('solvent_type', 0))
        }
        
        # Make prediction
        prediction_result = active_learner._get_model_prediction(conditions)
        
        # Add synthesis recommendation
        recommendation = generate_synthesis_recommendation(conditions, prediction_result)
        
        return jsonify({
            'success': True,
            'prediction': prediction_result,
            'recommendation': recommendation,
            'conditions': conditions
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/suggest_experiments', methods=['POST'])
def suggest_experiments():
    """Generate active learning experiment suggestions"""
    try:
        data = request.json
        strategy = data.get('strategy', 'mixed')
        num_suggestions = int(data.get('num_suggestions', 3))
        
        # Load experimental data if available
        experimental_data = None
        try:
            if validation_pipeline.validator.experiments_file.exists():
                experimental_data = pd.read_csv(validation_pipeline.validator.experiments_file)
        except:
            pass
        
        # Generate suggestions
        suggestions = active_learner.suggest_experiments(
            experimental_data=experimental_data,
            strategy=strategy,
            num_suggestions=num_suggestions
        )
        
        # Convert to JSON-serializable format
        suggestions_data = []
        for suggestion in suggestions:
            suggestions_data.append({
                'id': suggestion.suggestion_id,
                'conditions': suggestion.synthesis_conditions,
                'predicted_phase': suggestion.predicted_outcome['predicted_phase'],
                'confidence': suggestion.predicted_outcome['confidence'],
                'priority': suggestion.priority,
                'rationale': suggestion.rationale,
                'acquisition_score': suggestion.acquisition_score,
                'information_gain': suggestion.estimated_information_gain
            })
        
        return jsonify({
            'success': True,
            'suggestions': suggestions_data
        })
        
    except Exception as e:
        logger.error(f"Suggestion error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/optimization', methods=['POST'])
def optimization():
    """Run synthesis optimization"""
    try:
        data = request.json
        objective = data.get('objective', 'cspbbr3_probability')
        method = data.get('method', 'bayesian')
        
        if method == 'bayesian':
            # Load experimental data
            experimental_data = None
            try:
                if validation_pipeline.validator.results_file.exists():
                    experimental_data = pd.read_csv(validation_pipeline.validator.results_file)
            except:
                pass
            
            if experimental_data is not None and len(experimental_data) >= 3:
                # Fit surrogate model and optimize
                active_learner.bayesian_optimizer.objective = objective
                active_learner.bayesian_optimizer.fit_surrogate_model(experimental_data)
                
                optimal_conditions = active_learner.bayesian_optimizer.suggest_experiments(
                    num_suggestions=1, acquisition_type='EI'
                )
                
                if optimal_conditions:
                    conditions_dict = {
                        'cs_br_concentration': float(optimal_conditions[0][0]),
                        'pb_br2_concentration': float(optimal_conditions[0][1]),
                        'temperature': float(optimal_conditions[0][2]),
                        'oa_concentration': float(optimal_conditions[0][3]),
                        'oam_concentration': float(optimal_conditions[0][4]),
                        'reaction_time': float(optimal_conditions[0][5]),
                        'solvent_type': int(optimal_conditions[0][6])
                    }
                    
                    # Get prediction for optimal conditions
                    prediction = active_learner._get_model_prediction(conditions_dict)
                    
                    return jsonify({
                        'success': True,
                        'optimal_conditions': conditions_dict,
                        'prediction': prediction,
                        'method': 'Bayesian Optimization',
                        'objective': objective
                    })
                else:
                    raise ValueError("Optimization failed to find conditions")
            else:
                raise ValueError("Insufficient experimental data for Bayesian optimization")
        
        else:
            # Grid search optimization
            optimal_conditions, best_score = grid_search_optimization(objective)
            prediction = active_learner._get_model_prediction(optimal_conditions)
            
            return jsonify({
                'success': True,
                'optimal_conditions': optimal_conditions,
                'prediction': prediction,
                'method': 'Grid Search',
                'objective': objective,
                'score': best_score
            })
        
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/validation_status')
def validation_status():
    """Get validation system status"""
    try:
        status = validation_pipeline.get_pipeline_status()
        
        # Add validation metrics if available
        if validation_pipeline.validator.validation_file.exists():
            report = validation_pipeline.validator.generate_validation_report()
            status['validation_metrics'] = {
                'total_experiments': report.get('total_experiments', 0),
                'phase_accuracy': report.get('phase_prediction_accuracy_percent', 0),
                'mean_confidence': report.get('confidence_analysis', {}).get('mean_confidence', 0)
            }
        else:
            status['validation_metrics'] = {
                'total_experiments': 0,
                'phase_accuracy': 0,
                'mean_confidence': 0
            }
        
        return jsonify({'success': True, 'status': status})
        
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/validation_plots')
def validation_plots():
    """Generate validation plots"""
    try:
        # Create validation plots
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('CsPbBr₃ Digital Twin Validation Dashboard', fontsize=14)
        
        # Check if validation data exists
        if validation_pipeline.validator.validation_file.exists():
            with open(validation_pipeline.validator.validation_file, 'r') as f:
                analyses = json.load(f)
            
            if analyses:
                # Plot 1: Phase Prediction Accuracy
                ax1 = axes[0, 0]
                correct = [a['phase_prediction_correct'] for a in analyses]
                phases = [a['predicted_phase'] for a in analyses]
                
                phase_counts = {}
                phase_correct = {}
                for phase, is_correct in zip(phases, correct):
                    phase_counts[phase] = phase_counts.get(phase, 0) + 1
                    if is_correct:
                        phase_correct[phase] = phase_correct.get(phase, 0) + 1
                
                if phase_counts:
                    phase_names = list(phase_counts.keys())
                    accuracies = [phase_correct.get(p, 0) / phase_counts[p] * 100 for p in phase_names]
                    
                    bars = ax1.bar(phase_names, accuracies)
                    ax1.set_title('Phase Prediction Accuracy')
                    ax1.set_ylabel('Accuracy (%)')
                    ax1.tick_params(axis='x', rotation=45)
                    
                    for bar, acc in zip(bars, accuracies):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                f'{acc:.1f}%', ha='center', va='bottom')
                
                # Plot 2: Confidence vs Accuracy
                ax2 = axes[0, 1]
                confidences = [a['prediction_confidence'] for a in analyses]
                correct_numeric = [1 if c else 0 for c in correct]
                
                if len(confidences) > 1:
                    ax2.scatter(confidences, correct_numeric, alpha=0.7)
                    ax2.set_xlabel('Prediction Confidence')
                    ax2.set_ylabel('Correct (1) / Incorrect (0)')
                    ax2.set_title('Confidence vs Accuracy')
                
                # Plot 3: Property Errors
                ax3 = axes[1, 0]
                property_errors = {'bandgap': [], 'plqy': [], 'particle_size': [], 'emission_peak': []}
                
                for analysis in analyses:
                    if 'property_accuracies' in analysis:
                        for prop in property_errors.keys():
                            if prop in analysis['property_accuracies']:
                                property_errors[prop].append(
                                    analysis['property_accuracies'][prop]['relative_error_percent']
                                )
                
                error_data = [errors for errors in property_errors.values() if errors]
                error_labels = [prop for prop, errors in property_errors.items() if errors]
                
                if error_data:
                    ax3.boxplot(error_data, labels=error_labels)
                    ax3.set_title('Property Prediction Errors')
                    ax3.set_ylabel('Relative Error (%)')
                    ax3.tick_params(axis='x', rotation=45)
                
                # Plot 4: Cumulative Accuracy
                ax4 = axes[1, 1]
                cumulative_accuracy = []
                for i in range(len(analyses)):
                    current_correct = sum(1 for a in analyses[:i+1] if a['phase_prediction_correct'])
                    cumulative_accuracy.append(current_correct / (i + 1) * 100)
                
                ax4.plot(range(1, len(analyses) + 1), cumulative_accuracy, 'b-o', linewidth=2)
                ax4.set_xlabel('Experiment Number')
                ax4.set_ylabel('Cumulative Accuracy (%)')
                ax4.set_title('Model Performance Over Time')
                ax4.grid(True, alpha=0.3)
            
            else:
                # No data available
                for ax in axes.flat:
                    ax.text(0.5, 0.5, 'No validation data available', 
                           ha='center', va='center', transform=ax.transAxes)
        
        else:
            # No validation file
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No validation data available', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'plot_data': img_data
        })
        
    except Exception as e:
        logger.error(f"Plot generation error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/record_experiment', methods=['POST'])
def record_experiment():
    """Record experimental results"""
    try:
        data = request.json
        
        # Setup experiment conditions
        conditions = {
            'cs_br_concentration': float(data['conditions']['cs_concentration']),
            'pb_br2_concentration': float(data['conditions']['pb_concentration']),
            'temperature': float(data['conditions']['temperature']),
            'oa_concentration': float(data['conditions']['oa_concentration']),
            'oam_concentration': float(data['conditions']['oam_concentration']),
            'reaction_time': float(data['conditions']['reaction_time']),
            'solvent_type': int(data['conditions']['solvent_type']),
            'date_conducted': datetime.now().strftime("%Y-%m-%d"),
            'researcher': data.get('researcher', 'Web User'),
            'experiment_id': validation_pipeline.validator.generate_experiment_id(),
            'notes': data.get('notes', 'Experiment recorded via web interface')
        }
        
        # Setup experimental results
        results = {
            'analysis_date': datetime.now().strftime("%Y-%m-%d"),
            'dominant_phase': data['results']['dominant_phase'],
            'phase_purity': float(data['results']['phase_purity']),
            'synthesis_success': bool(data['results']['synthesis_success']),
            'solution_color': data['results'].get('solution_color', 'green'),
            'precipitate_observed': bool(data['results'].get('precipitate_observed', False)),
            'characterization_methods': data['results'].get('characterization_methods', ['XRD']),
            'secondary_phases': data['results'].get('secondary_phases', [])
        }
        
        # Add optional measurements
        if 'bandgap' in data['results'] and data['results']['bandgap']:
            results['bandgap'] = float(data['results']['bandgap'])
        if 'plqy' in data['results'] and data['results']['plqy']:
            results['plqy'] = float(data['results']['plqy'])
        if 'particle_size' in data['results'] and data['results']['particle_size']:
            results['particle_size'] = float(data['results']['particle_size'])
        if 'emission_peak' in data['results'] and data['results']['emission_peak']:
            results['emission_peak'] = float(data['results']['emission_peak'])
        
        # Record experiment
        exp_id = validation_pipeline.setup_new_experiment(conditions)
        success = validation_pipeline.record_experiment_results(exp_id, results)
        
        if success:
            return jsonify({
                'success': True,
                'experiment_id': exp_id,
                'message': 'Experiment recorded successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to record experiment'
            })
        
    except Exception as e:
        logger.error(f"Record experiment error: {e}")
        return jsonify({'success': False, 'error': str(e)})

def generate_synthesis_recommendation(conditions: Dict[str, float], 
                                    prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Generate synthesis recommendations based on prediction"""
    
    recommendations = []
    warnings = []
    
    # Temperature recommendations
    temp = conditions['temperature']
    if temp < 150:
        recommendations.append("Consider increasing temperature to 150-200°C for better phase formation")
    elif temp > 220:
        warnings.append("High temperature may lead to decomposition")
    
    # Stoichiometry recommendations
    cs_pb_ratio = conditions['cs_br_concentration'] / conditions['pb_br2_concentration']
    if cs_pb_ratio < 0.9:
        recommendations.append("Consider increasing Cs concentration for better stoichiometry")
    elif cs_pb_ratio > 1.3:
        recommendations.append("Consider decreasing Cs concentration to avoid excess")
    
    # Prediction-based recommendations
    if prediction['predicted_phase'] == 'CsPbBr3_3D':
        if prediction['confidence'] > 0.8:
            recommendations.append("High confidence prediction - excellent conditions")
        elif prediction['confidence'] > 0.6:
            recommendations.append("Good conditions with moderate confidence")
        else:
            recommendations.append("Consider optimizing conditions for higher confidence")
    else:
        recommendations.append(f"Predicted {prediction['predicted_phase']} - may want to adjust conditions")
    
    # Ligand recommendations
    ligand_total = conditions['oa_concentration'] + conditions['oam_concentration']
    if ligand_total < 0.3:
        recommendations.append("Consider increasing ligand concentration for better surface passivation")
    elif ligand_total > 1.0:
        warnings.append("High ligand concentration may inhibit growth")
    
    return {
        'recommendations': recommendations,
        'warnings': warnings,
        'overall_assessment': 'Good' if prediction['confidence'] > 0.6 else 'Needs optimization',
        'confidence_level': 'High' if prediction['confidence'] > 0.8 else 'Medium' if prediction['confidence'] > 0.6 else 'Low'
    }

def grid_search_optimization(objective: str) -> tuple:
    """Simple grid search optimization"""
    
    # Define parameter grids
    cs_conc_range = np.linspace(0.8, 1.5, 5)
    pb_conc_range = np.linspace(0.8, 1.5, 5)
    temp_range = np.linspace(160, 220, 4)
    
    best_score = -float('inf')
    best_conditions = None
    
    for cs_conc in cs_conc_range:
        for pb_conc in pb_conc_range:
            for temp in temp_range:
                conditions = {
                    'cs_br_concentration': cs_conc,
                    'pb_br2_concentration': pb_conc,
                    'temperature': temp,
                    'oa_concentration': 0.4,
                    'oam_concentration': 0.2,
                    'reaction_time': 60,
                    'solvent_type': 0
                }
                
                prediction = active_learner._get_model_prediction(conditions)
                
                # Calculate score based on objective
                if objective == 'cspbbr3_probability':
                    if prediction['predicted_phase'] == 'CsPbBr3_3D':
                        score = prediction['confidence']
                    else:
                        score = 0.1
                elif objective == 'confidence':
                    score = prediction['confidence']
                else:
                    score = prediction['confidence']
                
                if score > best_score:
                    best_score = score
                    best_conditions = conditions
    
    return best_conditions, best_score

# HTML Templates
def create_html_templates():
    """Create HTML templates for the web interface"""
    
    # Create templates directory
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Dashboard template
    dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CsPbBr₃ Digital Twin Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .panel {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .panel h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        .btn {
            background: #667eea;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin: 5px;
        }
        .btn:hover {
            background: #5a6fd8;
        }
        .btn-success {
            background: #28a745;
        }
        .btn-warning {
            background: #ffc107;
            color: #333;
        }
        .prediction-result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
            background: #f8f9fa;
        }
        .suggestion-item {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            background: #fff;
        }
        .priority-high {
            border-left: 4px solid #dc3545;
        }
        .priority-medium {
            border-left: 4px solid #ffc107;
        }
        .priority-low {
            border-left: 4px solid #28a745;
        }
        .status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }
        .status-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .status-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        .tab {
            background: #f8f9fa;
            border: 1px solid #ddd;
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
        }
        .tab.active {
            background: white;
            border-bottom: 1px solid white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .validation-plot {
            text-align: center;
            margin: 20px 0;
        }
        .validation-plot img {
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .loading {
            text-align: center;
            color: #667eea;
            font-style: italic;
        }
        .error {
            color: #dc3545;
            background: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .success {
            color: #155724;
            background: #d4edda;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 CsPbBr₃ Digital Twin Dashboard</h1>
            <p>AI-Powered Perovskite Synthesis Prediction & Optimization</p>
        </div>

        <!-- Status Overview -->
        <div class="status-grid">
            <div class="status-card">
                <div class="status-value" id="total-experiments">0</div>
                <div>Total Experiments</div>
            </div>
            <div class="status-card">
                <div class="status-value" id="phase-accuracy">0%</div>
                <div>Phase Accuracy</div>
            </div>
            <div class="status-card">
                <div class="status-value" id="mean-confidence">0.0</div>
                <div>Mean Confidence</div>
            </div>
        </div>

        <!-- Main Dashboard -->
        <div class="tabs">
            <div class="tab active" onclick="showTab('prediction')">🔮 Prediction</div>
            <div class="tab" onclick="showTab('suggestions')">🤖 AI Suggestions</div>
            <div class="tab" onclick="showTab('optimization')">🎯 Optimization</div>
            <div class="tab" onclick="showTab('validation')">📊 Validation</div>
            <div class="tab" onclick="showTab('record')">📝 Record Results</div>
        </div>

        <!-- Prediction Tab -->
        <div id="prediction-tab" class="tab-content active">
            <div class="dashboard-grid">
                <div class="panel">
                    <h3>🧪 Synthesis Conditions</h3>
                    <div class="form-group">
                        <label>Cs-Br Concentration (mol/L)</label>
                        <input type="number" id="cs-concentration" value="1.1" step="0.1" min="0.1" max="3.0">
                    </div>
                    <div class="form-group">
                        <label>Pb-Br₂ Concentration (mol/L)</label>
                        <input type="number" id="pb-concentration" value="1.1" step="0.1" min="0.1" max="3.0">
                    </div>
                    <div class="form-group">
                        <label>Temperature (°C)</label>
                        <input type="number" id="temperature" value="190" step="10" min="80" max="300">
                    </div>
                    <div class="form-group">
                        <label>Oleic Acid (mol/L)</label>
                        <input type="number" id="oa-concentration" value="0.4" step="0.1" min="0.0" max="2.0">
                    </div>
                    <div class="form-group">
                        <label>Oleylamine (mol/L)</label>
                        <input type="number" id="oam-concentration" value="0.2" step="0.1" min="0.0" max="1.0">
                    </div>
                    <div class="form-group">
                        <label>Reaction Time (minutes)</label>
                        <input type="number" id="reaction-time" value="75" step="5" min="5" max="200">
                    </div>
                    <div class="form-group">
                        <label>Solvent</label>
                        <select id="solvent-type">
                            <option value="0">DMSO</option>
                            <option value="1">DMF</option>
                            <option value="2">Toluene</option>
                            <option value="3">Octadecene</option>
                            <option value="4">Mixed</option>
                        </select>
                    </div>
                    <button class="btn" onclick="makePrediction()">🔮 Predict Outcome</button>
                </div>

                <div class="panel">
                    <h3>📊 Prediction Results</h3>
                    <div id="prediction-results">
                        <p class="loading">Enter synthesis conditions and click "Predict Outcome" to see results.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- AI Suggestions Tab -->
        <div id="suggestions-tab" class="tab-content">
            <div class="panel">
                <h3>🤖 AI-Powered Experiment Suggestions</h3>
                <div style="margin-bottom: 20px;">
                    <label>Strategy:</label>
                    <select id="suggestion-strategy" style="margin: 0 10px;">
                        <option value="mixed">Mixed (Recommended)</option>
                        <option value="uncertainty">Uncertainty-Based</option>
                        <option value="bayesian">Bayesian Optimization</option>
                        <option value="diversity">Diversity Exploration</option>
                    </select>
                    
                    <label>Number of Suggestions:</label>
                    <input type="number" id="num-suggestions" value="3" min="1" max="10" style="width: 60px; margin: 0 10px;">
                    
                    <button class="btn" onclick="getSuggestions()">🎯 Generate Suggestions</button>
                </div>
                <div id="suggestions-results"></div>
            </div>
        </div>

        <!-- Optimization Tab -->
        <div id="optimization-tab" class="tab-content">
            <div class="panel">
                <h3>🎯 Synthesis Optimization</h3>
                <div style="margin-bottom: 20px;">
                    <label>Optimization Objective:</label>
                    <select id="optimization-objective" style="margin: 0 10px;">
                        <option value="cspbbr3_probability">CsPbBr₃ 3D Probability</option>
                        <option value="confidence">Prediction Confidence</option>
                        <option value="bandgap">Target Bandgap</option>
                        <option value="plqy">PLQY Maximization</option>
                    </select>
                    
                    <label>Method:</label>
                    <select id="optimization-method" style="margin: 0 10px;">
                        <option value="bayesian">Bayesian Optimization</option>
                        <option value="grid">Grid Search</option>
                    </select>
                    
                    <button class="btn" onclick="runOptimization()">⚡ Optimize</button>
                </div>
                <div id="optimization-results"></div>
            </div>
        </div>

        <!-- Validation Tab -->
        <div id="validation-tab" class="tab-content">
            <div class="panel">
                <h3>📊 Model Validation Analysis</h3>
                <button class="btn" onclick="loadValidationPlots()">📈 Generate Validation Plots</button>
                <div id="validation-plots"></div>
            </div>
        </div>

        <!-- Record Results Tab -->
        <div id="record-tab" class="tab-content">
            <div class="dashboard-grid">
                <div class="panel">
                    <h3>🧪 Experiment Conditions</h3>
                    <div class="form-group">
                        <label>Researcher Name</label>
                        <input type="text" id="researcher-name" placeholder="Your name">
                    </div>
                    <div class="form-group">
                        <label>Cs-Br Concentration (mol/L)</label>
                        <input type="number" id="record-cs-concentration" step="0.1" min="0.1" max="3.0">
                    </div>
                    <div class="form-group">
                        <label>Pb-Br₂ Concentration (mol/L)</label>
                        <input type="number" id="record-pb-concentration" step="0.1" min="0.1" max="3.0">
                    </div>
                    <div class="form-group">
                        <label>Temperature (°C)</label>
                        <input type="number" id="record-temperature" step="10" min="80" max="300">
                    </div>
                    <div class="form-group">
                        <label>Oleic Acid (mol/L)</label>
                        <input type="number" id="record-oa-concentration" step="0.1" min="0.0" max="2.0">
                    </div>
                    <div class="form-group">
                        <label>Oleylamine (mol/L)</label>
                        <input type="number" id="record-oam-concentration" step="0.1" min="0.0" max="1.0">
                    </div>
                    <div class="form-group">
                        <label>Reaction Time (minutes)</label>
                        <input type="number" id="record-reaction-time" step="5" min="5" max="200">
                    </div>
                    <div class="form-group">
                        <label>Solvent</label>
                        <select id="record-solvent-type">
                            <option value="0">DMSO</option>
                            <option value="1">DMF</option>
                            <option value="2">Toluene</option>
                            <option value="3">Octadecene</option>
                            <option value="4">Mixed</option>
                        </select>
                    </div>
                </div>

                <div class="panel">
                    <h3>📊 Experimental Results</h3>
                    <div class="form-group">
                        <label>Dominant Phase</label>
                        <select id="dominant-phase">
                            <option value="CsPbBr3_3D">CsPbBr₃ 3D</option>
                            <option value="Cs4PbBr6_0D">Cs₄PbBr₆ 0D</option>
                            <option value="CsPb2Br5_2D">CsPb₂Br₅ 2D</option>
                            <option value="Mixed">Mixed Phases</option>
                            <option value="Failed">Failed Synthesis</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Phase Purity (0-1)</label>
                        <input type="number" id="phase-purity" step="0.01" min="0" max="1">
                    </div>
                    <div class="form-group">
                        <label>Synthesis Success</label>
                        <select id="synthesis-success">
                            <option value="true">Success</option>
                            <option value="false">Failed</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Bandgap (eV) - Optional</label>
                        <input type="number" id="bandgap" step="0.01" min="0" max="5">
                    </div>
                    <div class="form-group">
                        <label>PLQY (0-1) - Optional</label>
                        <input type="number" id="plqy" step="0.01" min="0" max="1">
                    </div>
                    <div class="form-group">
                        <label>Particle Size (nm) - Optional</label>
                        <input type="number" id="particle-size" step="0.1" min="0" max="1000">
                    </div>
                    <div class="form-group">
                        <label>Emission Peak (nm) - Optional</label>
                        <input type="number" id="emission-peak" step="1" min="400" max="700">
                    </div>
                    <div class="form-group">
                        <label>Solution Color</label>
                        <input type="text" id="solution-color" placeholder="e.g., bright green">
                    </div>
                    <div class="form-group">
                        <label>Notes</label>
                        <textarea id="experiment-notes" rows="3" style="width: 100%; padding: 8px;"></textarea>
                    </div>
                    <button class="btn btn-success" onclick="recordExperiment()">💾 Record Experiment</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize dashboard
        $(document).ready(function() {
            loadValidationStatus();
        });

        function showTab(tabName) {
            // Hide all tab contents
            $('.tab-content').removeClass('active');
            $('.tab').removeClass('active');
            
            // Show selected tab
            $('#' + tabName + '-tab').addClass('active');
            event.target.classList.add('active');
        }

        function makePrediction() {
            const conditions = {
                cs_concentration: $('#cs-concentration').val(),
                pb_concentration: $('#pb-concentration').val(),
                temperature: $('#temperature').val(),
                oa_concentration: $('#oa-concentration').val(),
                oam_concentration: $('#oam-concentration').val(),
                reaction_time: $('#reaction-time').val(),
                solvent_type: $('#solvent-type').val()
            };

            $('#prediction-results').html('<p class="loading">Making prediction...</p>');

            $.ajax({
                url: '/predict',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(conditions),
                success: function(response) {
                    if (response.success) {
                        displayPredictionResults(response);
                    } else {
                        $('#prediction-results').html('<div class="error">Error: ' + response.error + '</div>');
                    }
                },
                error: function() {
                    $('#prediction-results').html('<div class="error">Network error occurred</div>');
                }
            });
        }

        function displayPredictionResults(response) {
            const prediction = response.prediction;
            const recommendation = response.recommendation;
            
            let html = `
                <div class="prediction-result">
                    <h4>🎯 Predicted Outcome</h4>
                    <p><strong>Phase:</strong> ${prediction.predicted_phase}</p>
                    <p><strong>Confidence:</strong> ${(prediction.confidence * 100).toFixed(1)}%</p>
                    
                    <h5>Phase Probabilities:</h5>
                    <ul>
                        <li>CsPbBr₃ 3D: ${(prediction.phase_probabilities[0] * 100).toFixed(1)}%</li>
                        <li>Cs₄PbBr₆ 0D: ${(prediction.phase_probabilities[1] * 100).toFixed(1)}%</li>
                        <li>CsPb₂Br₅ 2D: ${(prediction.phase_probabilities[2] * 100).toFixed(1)}%</li>
                        <li>Mixed: ${(prediction.phase_probabilities[3] * 100).toFixed(1)}%</li>
                        <li>Failed: ${(prediction.phase_probabilities[4] * 100).toFixed(1)}%</li>
                    </ul>
            `;
            
            if (prediction.has_uncertainty) {
                html += `
                    <h5>Predicted Properties (with uncertainty):</h5>
                    <ul>
                `;
                for (const [prop, stats] of Object.entries(prediction.properties)) {
                    html += `<li>${prop}: ${stats.mean.toFixed(3)} ± ${stats.std.toFixed(3)}</li>`;
                }
                html += '</ul>';
            }
            
            html += `
                    <h5>💡 Recommendations:</h5>
                    <ul>
            `;
            for (const rec of recommendation.recommendations) {
                html += `<li>${rec}</li>`;
            }
            html += '</ul>';
            
            if (recommendation.warnings.length > 0) {
                html += `<h5>⚠️ Warnings:</h5><ul>`;
                for (const warning of recommendation.warnings) {
                    html += `<li>${warning}</li>`;
                }
                html += '</ul>';
            }
            
            html += `
                    <p><strong>Overall Assessment:</strong> ${recommendation.overall_assessment}</p>
                </div>
            `;
            
            $('#prediction-results').html(html);
        }

        function getSuggestions() {
            const strategy = $('#suggestion-strategy').val();
            const numSuggestions = $('#num-suggestions').val();
            
            $('#suggestions-results').html('<p class="loading">Generating AI suggestions...</p>');
            
            $.ajax({
                url: '/suggest_experiments',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    strategy: strategy,
                    num_suggestions: parseInt(numSuggestions)
                }),
                success: function(response) {
                    if (response.success) {
                        displaySuggestions(response.suggestions);
                    } else {
                        $('#suggestions-results').html('<div class="error">Error: ' + response.error + '</div>');
                    }
                },
                error: function() {
                    $('#suggestions-results').html('<div class="error">Network error occurred</div>');
                }
            });
        }

        function displaySuggestions(suggestions) {
            let html = '<h4>🎯 Recommended Experiments:</h4>';
            
            for (const suggestion of suggestions) {
                html += `
                    <div class="suggestion-item priority-${suggestion.priority}">
                        <h5>Experiment ${suggestion.id}</h5>
                        <p><strong>Priority:</strong> ${suggestion.priority.charAt(0).toUpperCase() + suggestion.priority.slice(1)}</p>
                        <p><strong>Predicted:</strong> ${suggestion.predicted_phase} (confidence: ${(suggestion.confidence * 100).toFixed(1)}%)</p>
                        <p><strong>Rationale:</strong> ${suggestion.rationale}</p>
                        
                        <h6>Conditions:</h6>
                        <ul>
                            <li>Cs-Br: ${suggestion.conditions.cs_br_concentration.toFixed(2)} mol/L</li>
                            <li>Pb-Br₂: ${suggestion.conditions.pb_br2_concentration.toFixed(2)} mol/L</li>
                            <li>Temperature: ${suggestion.conditions.temperature.toFixed(0)}°C</li>
                            <li>OA: ${suggestion.conditions.oa_concentration.toFixed(2)} mol/L</li>
                            <li>OAm: ${suggestion.conditions.oam_concentration.toFixed(2)} mol/L</li>
                            <li>Time: ${suggestion.conditions.reaction_time.toFixed(0)} min</li>
                            <li>Solvent: ${['DMSO', 'DMF', 'Toluene', 'Octadecene', 'Mixed'][suggestion.conditions.solvent_type]}</li>
                        </ul>
                        
                        <button class="btn btn-success" onclick="useSuggestion('${suggestion.id}')">📋 Use This Suggestion</button>
                    </div>
                `;
            }
            
            $('#suggestions-results').html(html);
        }

        function useSuggestion(suggestionId) {
            // Find suggestion and populate prediction tab
            // This would need more implementation to actually populate the form
            showTab('prediction');
            alert('Suggestion loaded! Check the prediction tab.');
        }

        function runOptimization() {
            const objective = $('#optimization-objective').val();
            const method = $('#optimization-method').val();
            
            $('#optimization-results').html('<p class="loading">Running optimization...</p>');
            
            $.ajax({
                url: '/optimization',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    objective: objective,
                    method: method
                }),
                success: function(response) {
                    if (response.success) {
                        displayOptimizationResults(response);
                    } else {
                        $('#optimization-results').html('<div class="error">Error: ' + response.error + '</div>');
                    }
                },
                error: function() {
                    $('#optimization-results').html('<div class="error">Network error occurred</div>');
                }
            });
        }

        function displayOptimizationResults(response) {
            const conditions = response.optimal_conditions;
            const prediction = response.prediction;
            
            let html = `
                <div class="prediction-result">
                    <h4>🎯 Optimized Conditions</h4>
                    <p><strong>Method:</strong> ${response.method}</p>
                    <p><strong>Objective:</strong> ${response.objective}</p>
                    
                    <h5>Optimal Synthesis Conditions:</h5>
                    <ul>
                        <li>Cs-Br: ${conditions.cs_br_concentration.toFixed(2)} mol/L</li>
                        <li>Pb-Br₂: ${conditions.pb_br2_concentration.toFixed(2)} mol/L</li>
                        <li>Temperature: ${conditions.temperature.toFixed(0)}°C</li>
                        <li>Oleic Acid: ${conditions.oa_concentration.toFixed(2)} mol/L</li>
                        <li>Oleylamine: ${conditions.oam_concentration.toFixed(2)} mol/L</li>
                        <li>Reaction Time: ${conditions.reaction_time.toFixed(0)} minutes</li>
                        <li>Solvent: ${['DMSO', 'DMF', 'Toluene', 'Octadecene', 'Mixed'][conditions.solvent_type]}</li>
                    </ul>
                    
                    <h5>Expected Outcome:</h5>
                    <p><strong>Phase:</strong> ${prediction.predicted_phase}</p>
                    <p><strong>Confidence:</strong> ${(prediction.confidence * 100).toFixed(1)}%</p>
                    
                    <button class="btn" onclick="copyOptimalConditions()">📋 Copy to Prediction Tab</button>
                </div>
            `;
            
            $('#optimization-results').html(html);
        }

        function copyOptimalConditions() {
            // This would copy the optimal conditions to the prediction tab
            alert('Optimal conditions copied! Check the prediction tab.');
        }

        function loadValidationPlots() {
            $('#validation-plots').html('<p class="loading">Generating validation plots...</p>');
            
            $.ajax({
                url: '/validation_plots',
                method: 'GET',
                success: function(response) {
                    if (response.success) {
                        $('#validation-plots').html(`
                            <div class="validation-plot">
                                <img src="data:image/png;base64,${response.plot_data}" alt="Validation Plots">
                            </div>
                        `);
                    } else {
                        $('#validation-plots').html('<div class="error">Error generating plots: ' + response.error + '</div>');
                    }
                },
                error: function() {
                    $('#validation-plots').html('<div class="error">Network error occurred</div>');
                }
            });
        }

        function loadValidationStatus() {
            $.ajax({
                url: '/validation_status',
                method: 'GET',
                success: function(response) {
                    if (response.success && response.status.validation_metrics) {
                        const metrics = response.status.validation_metrics;
                        $('#total-experiments').text(metrics.total_experiments);
                        $('#phase-accuracy').text(metrics.phase_accuracy.toFixed(1) + '%');
                        $('#mean-confidence').text(metrics.mean_confidence.toFixed(2));
                    }
                }
            });
        }

        function recordExperiment() {
            const conditions = {
                cs_concentration: $('#record-cs-concentration').val(),
                pb_concentration: $('#record-pb-concentration').val(),
                temperature: $('#record-temperature').val(),
                oa_concentration: $('#record-oa-concentration').val(),
                oam_concentration: $('#record-oam-concentration').val(),
                reaction_time: $('#record-reaction-time').val(),
                solvent_type: $('#record-solvent-type').val()
            };

            const results = {
                dominant_phase: $('#dominant-phase').val(),
                phase_purity: $('#phase-purity').val(),
                synthesis_success: $('#synthesis-success').val() === 'true',
                solution_color: $('#solution-color').val(),
                characterization_methods: ['XRD'] // Default
            };

            // Add optional measurements
            if ($('#bandgap').val()) results.bandgap = $('#bandgap').val();
            if ($('#plqy').val()) results.plqy = $('#plqy').val();
            if ($('#particle-size').val()) results.particle_size = $('#particle-size').val();
            if ($('#emission-peak').val()) results.emission_peak = $('#emission-peak').val();

            const data = {
                researcher: $('#researcher-name').val(),
                conditions: conditions,
                results: results,
                notes: $('#experiment-notes').val()
            };

            $.ajax({
                url: '/record_experiment',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(data),
                success: function(response) {
                    if (response.success) {
                        alert('Experiment recorded successfully! ID: ' + response.experiment_id);
                        // Clear form
                        $('#record-tab input, #record-tab select, #record-tab textarea').val('');
                        // Refresh validation status
                        loadValidationStatus();
                    } else {
                        alert('Error recording experiment: ' + response.error);
                    }
                },
                error: function() {
                    alert('Network error occurred');
                }
            });
        }
    </script>
</body>
</html>
    """
    
    with open(templates_dir / "dashboard.html", "w") as f:
        f.write(dashboard_html)
    
    logger.info("HTML templates created successfully")

def main():
    """Run the web interface"""
    
    # Initialize system
    success = initialize_system()
    
    # Create HTML templates
    create_html_templates()
    
    # Print startup info
    print("🌐 CsPbBr₃ Digital Twin Web Interface")
    print("=" * 50)
    if success:
        print("✅ System initialized with trained model")
    else:
        print("⚠️ Using dummy system for demonstration")
    
    print("\n🚀 Starting web server...")
    print("   URL: http://localhost:5000")
    print("   Press Ctrl+C to stop")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
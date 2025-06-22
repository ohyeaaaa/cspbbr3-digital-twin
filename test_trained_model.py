#!/usr/bin/env python3
"""
Test Trained Model for CsPbBr₃ Digital Twin
Load the trained model and make predictions
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for synthesis prediction"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64], n_phases=5, n_properties=4):
        super().__init__()
        
        # Feature extraction layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Phase classification head
        self.phase_classifier = nn.Linear(prev_dim, n_phases)
        
        # Property regression heads
        self.property_regressors = nn.ModuleDict({
            'bandgap': nn.Linear(prev_dim, 1),
            'plqy': nn.Linear(prev_dim, 1),
            'particle_size': nn.Linear(prev_dim, 1),
            'emission_peak': nn.Linear(prev_dim, 1)
        })
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        phase_logits = self.phase_classifier(features)
        phase_probs = torch.softmax(phase_logits, dim=-1)
        
        properties = {}
        for prop_name, regressor in self.property_regressors.items():
            properties[prop_name] = regressor(features).squeeze()
        
        return {
            'phase_logits': phase_logits,
            'phase_probabilities': phase_probs,
            'properties': properties
        }

def load_trained_model(model_path='best_model.pth', config_path='training_output/training_results.json'):
    """Load the trained model"""
    logger.info(f"Loading model from {model_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_config = config['model_config']
    
    # Create model
    model = SimpleNeuralNetwork(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        n_phases=model_config['n_phases'],
        n_properties=model_config['n_properties']
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    logger.info("✅ Model loaded successfully")
    return model, config

def load_scaler(scaler_path='training_output/scaler.pkl'):
    """Load the feature scaler"""
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def predict_synthesis(model, scaler, synthesis_params, feature_cols):
    """Make prediction for synthesis parameters"""
    
    # Create feature vector
    features = []
    for col in feature_cols:
        if col in synthesis_params:
            features.append(synthesis_params[col])
        else:
            # Calculate derived features
            if col == 'cs_pb_ratio':
                features.append(synthesis_params['cs_br_concentration'] / synthesis_params['pb_br2_concentration'])
            elif col == 'supersaturation':
                features.append(np.log((synthesis_params['cs_br_concentration'] * synthesis_params['pb_br2_concentration']) / 
                                     (0.1 + synthesis_params['temperature'] / 1000)))
            elif col == 'ligand_ratio':
                ligand_total = synthesis_params.get('oa_concentration', 0) + synthesis_params.get('oam_concentration', 0)
                features.append(ligand_total / (synthesis_params['cs_br_concentration'] + synthesis_params['pb_br2_concentration']))
            elif col == 'temp_normalized':
                features.append((synthesis_params['temperature'] - 80) / (250 - 80))
            elif col == 'solvent_effect':
                solvent_effects = {0: 1.2, 1: 1.0, 2: 0.5, 3: 0.8, 4: 0.9}
                features.append(solvent_effects.get(synthesis_params.get('solvent_type', 0), 1.0))
            else:
                features.append(0.0)  # Default value
    
    # Normalize features
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features_scaled)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(features_tensor)
    
    # Process outputs
    phase_probs = outputs['phase_probabilities'][0].numpy()
    predicted_phase = np.argmax(phase_probs)
    
    properties = {}
    for prop_name, prop_tensor in outputs['properties'].items():
        if prop_tensor.dim() == 0:
            properties[prop_name] = prop_tensor.item()
        else:
            properties[prop_name] = prop_tensor[0].item()
    
    return {
        'phase': {
            'predicted_class': predicted_phase,
            'probabilities': phase_probs.tolist(),
            'confidence': float(np.max(phase_probs))
        },
        'properties': properties
    }

def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="Test trained CsPbBr₃ Digital Twin model")
    parser.add_argument('--cs-concentration', type=float, default=1.5, help='Cs-Br concentration (mol/L)')
    parser.add_argument('--pb-concentration', type=float, default=1.0, help='Pb-Br2 concentration (mol/L)')
    parser.add_argument('--temperature', type=float, default=150.0, help='Temperature (°C)')
    parser.add_argument('--oa-concentration', type=float, default=0.5, help='Oleic acid concentration (mol/L)')
    parser.add_argument('--oam-concentration', type=float, default=0.3, help='Oleylamine concentration (mol/L)')
    parser.add_argument('--reaction-time', type=float, default=60.0, help='Reaction time (minutes)')
    parser.add_argument('--solvent-type', type=int, default=0, help='Solvent type (0-4)')
    
    args = parser.parse_args()
    
    logger.info("🧬 Testing Trained CsPbBr₃ Digital Twin Model")
    logger.info("=" * 50)
    
    # Load model and scaler
    model, config = load_trained_model()
    scaler = load_scaler()
    
    # Prepare synthesis parameters
    synthesis_params = {
        'cs_br_concentration': args.cs_concentration,
        'pb_br2_concentration': args.pb_concentration,
        'temperature': args.temperature,
        'oa_concentration': args.oa_concentration,
        'oam_concentration': args.oam_concentration,
        'reaction_time': args.reaction_time,
        'solvent_type': args.solvent_type
    }
    
    logger.info("Input Parameters:")
    for key, value in synthesis_params.items():
        logger.info(f"  {key}: {value}")
    
    # Make prediction
    feature_cols = config['feature_columns']
    prediction = predict_synthesis(model, scaler, synthesis_params, feature_cols)
    
    # Display results
    phase_names = {0: 'CsPbBr3_3D', 1: 'Cs4PbBr6_0D', 2: 'CsPb2Br5_2D', 3: 'Mixed', 4: 'Failed'}
    
    logger.info("\n🔮 Prediction Results:")
    logger.info("=" * 30)
    
    phase_result = prediction['phase']
    predicted_phase_name = phase_names[phase_result['predicted_class']]
    logger.info(f"📊 Predicted Phase: {predicted_phase_name}")
    logger.info(f"🎯 Confidence: {phase_result['confidence']:.3f}")
    
    logger.info("\n📈 Phase Probabilities:")
    for i, prob in enumerate(phase_result['probabilities']):
        logger.info(f"  {phase_names[i]}: {prob:.3f}")
    
    logger.info("\n⚗️ Material Properties:")
    properties = prediction['properties']
    logger.info(f"  Bandgap: {properties['bandgap']:.3f} eV")
    logger.info(f"  PLQY: {properties['plqy']:.3f}")
    logger.info(f"  Particle Size: {properties['particle_size']:.1f} nm")
    logger.info(f"  Emission Peak: {properties['emission_peak']:.1f} nm")
    
    logger.info("\n✅ Prediction completed successfully!")

if __name__ == "__main__":
    main()
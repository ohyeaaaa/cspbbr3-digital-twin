#!/usr/bin/env python3
"""
Demonstration of the CsPbBr₃ Experimental Validation System
Shows complete workflow from setup to analysis
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import random
import numpy as np

from validation_pipeline import ValidationPipeline
from experimental_validation import ExperimentalConditions, ExperimentalResults

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_demo_experimental_results(exp_id: str, prediction, conditions: ExperimentalConditions) -> dict:
    """Generate realistic demo experimental results based on prediction"""
    
    # Simulate phase outcome based on prediction with some noise
    predicted_phase = prediction.predicted_phase if prediction else "CsPbBr3_3D"
    confidence = prediction.confidence if prediction else 0.4
    
    # Higher confidence predictions are more likely to be correct
    phase_correct_prob = 0.3 + 0.5 * confidence  # 30-80% accuracy range
    
    if random.random() < phase_correct_prob:
        actual_phase = predicted_phase
        phase_purity = 0.7 + 0.2 * confidence + random.gauss(0, 0.1)
    else:
        # Incorrect prediction - usually get mixed or different phase
        possible_phases = ["CsPbBr3_3D", "Cs4PbBr6_0D", "CsPb2Br5_2D", "Mixed", "Failed"]
        possible_phases.remove(predicted_phase)
        actual_phase = random.choice(possible_phases)
        phase_purity = 0.4 + random.gauss(0, 0.15)
    
    phase_purity = max(0.1, min(0.95, phase_purity))
    
    # Generate property measurements with realistic errors
    if prediction:
        # Add realistic measurement errors (5-20% typical)
        bandgap_error = random.gauss(0, 0.1)  # 10% std dev
        bandgap = prediction.predicted_properties['bandgap'] * (1 + bandgap_error)
        bandgap = max(1.0, min(2.5, bandgap))  # Physical limits
        
        plqy_error = random.gauss(0, 0.15)  # 15% std dev
        plqy = prediction.predicted_properties['plqy'] * (1 + plqy_error)
        plqy = max(0.01, min(1.0, plqy))  # Physical limits
        
        size_error = random.gauss(0, 0.2)  # 20% std dev
        particle_size = prediction.predicted_properties['particle_size'] * (1 + size_error)
        particle_size = max(1.0, min(100.0, particle_size))  # Physical limits
        
        emission_error = random.gauss(0, 0.05)  # 5% std dev
        emission_peak = prediction.predicted_properties['emission_peak'] * (1 + emission_error)
        emission_peak = max(450, min(600, emission_peak))  # Physical limits
    else:
        # Default values if no prediction
        bandgap = random.uniform(1.8, 2.3)
        plqy = random.uniform(0.1, 0.8)
        particle_size = random.uniform(5, 25)
        emission_peak = random.uniform(510, 530)
    
    # Synthesis success based on conditions and phase outcome
    temp_factor = 1.0 if conditions.temperature >= 180 else 0.7
    stoich_factor = 1.0 if abs(conditions.cs_br_concentration - conditions.pb_br2_concentration) < 0.2 else 0.8
    phase_factor = 1.0 if actual_phase == "CsPbBr3_3D" else 0.6
    
    success_prob = temp_factor * stoich_factor * phase_factor * 0.8
    synthesis_success = random.random() < success_prob
    
    # Generate realistic visual observations
    color_map = {
        "CsPbBr3_3D": "bright green",
        "Cs4PbBr6_0D": "pale green", 
        "CsPb2Br5_2D": "yellow-green",
        "Mixed": "green with yellow tint",
        "Failed": "colorless or pale yellow"
    }
    
    results_data = {
        "analysis_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "dominant_phase": actual_phase,
        "phase_purity": round(phase_purity, 3),
        "secondary_phases": ["CsPb2Br5_2D"] if actual_phase == "Mixed" else [],
        "solution_color": color_map.get(actual_phase, "unknown"),
        "precipitate_observed": actual_phase == "Failed",
        "bandgap": round(bandgap, 3),
        "plqy": round(plqy, 3),
        "particle_size": round(particle_size, 1),
        "emission_peak": round(emission_peak, 1),
        "synthesis_success": synthesis_success,
        "yield_percentage": round(random.uniform(60, 95), 1) if synthesis_success else round(random.uniform(10, 40), 1),
        "characterization_methods": ["XRD", "UV-vis", "PL", "TEM"],
        "additional_properties": {
            "emission_fwhm": round(random.uniform(15, 25), 1),
            "stokes_shift": round(random.uniform(8, 15), 1)
        },
        "notes": f"Demo experiment simulating {actual_phase} formation"
    }
    
    return results_data

def run_validation_demo():
    """Run complete validation demonstration"""
    
    print("🧪 CsPbBr₃ Digital Twin Validation System Demo")
    print("=" * 60)
    
    # Initialize pipeline
    demo_dir = "demo_experimental_data"
    pipeline = ValidationPipeline(demo_dir)
    
    # Define test conditions based on synthesis protocol recommendations
    test_conditions = [
        {
            # Optimal conditions
            "cs_br_concentration": 1.1,
            "pb_br2_concentration": 1.1,
            "temperature": 190.0,
            "oa_concentration": 0.4,
            "oam_concentration": 0.2,
            "reaction_time": 75.0,
            "solvent_type": 0,
            "date_conducted": datetime.now().strftime("%Y-%m-%d"),
            "researcher": "Demo User",
            "notes": "Optimal conditions from digital twin"
        },
        {
            # High temperature variant
            "cs_br_concentration": 1.2,
            "pb_br2_concentration": 1.0,
            "temperature": 200.0,
            "oa_concentration": 0.4,
            "oam_concentration": 0.2,
            "reaction_time": 45.0,
            "solvent_type": 0,
            "date_conducted": datetime.now().strftime("%Y-%m-%d"),
            "researcher": "Demo User",
            "notes": "High temperature test"
        },
        {
            # Lower confidence conditions
            "cs_br_concentration": 0.8,
            "pb_br2_concentration": 1.0,
            "temperature": 120.0,
            "oa_concentration": 0.3,
            "oam_concentration": 0.2,
            "reaction_time": 30.0,
            "solvent_type": 1,
            "date_conducted": datetime.now().strftime("%Y-%m-%d"),
            "researcher": "Demo User",
            "notes": "Lower temperature/concentration test"
        },
        {
            # Different solvent
            "cs_br_concentration": 1.2,
            "pb_br2_concentration": 1.0,
            "temperature": 160.0,
            "oa_concentration": 0.4,
            "oam_concentration": 0.2,
            "reaction_time": 45.0,
            "solvent_type": 2,
            "date_conducted": datetime.now().strftime("%Y-%m-%d"),
            "researcher": "Demo User",
            "notes": "Toluene solvent test"
        },
        {
            # Mixed phase conditions
            "cs_br_concentration": 1.0,
            "pb_br2_concentration": 0.8,
            "temperature": 140.0,
            "oa_concentration": 0.4,
            "oam_concentration": 0.1,
            "reaction_time": 60.0,
            "solvent_type": 3,
            "date_conducted": datetime.now().strftime("%Y-%m-%d"),
            "researcher": "Demo User",
            "notes": "Octadecene solvent with Cs excess"
        }
    ]
    
    print(f"🔬 Setting up {len(test_conditions)} validation experiments...")
    
    # Set up experiments and collect predictions
    experiments = []
    for i, conditions in enumerate(test_conditions):
        print(f"\n📝 Setting up experiment {i+1}/{len(test_conditions)}")
        # Add experiment_id to conditions
        conditions['experiment_id'] = pipeline.validator.generate_experiment_id()
        exp_id = pipeline.setup_new_experiment(conditions)
        
        # Get the prediction that was generated
        exp_data = pipeline.current_experiments[exp_id]
        prediction = exp_data['prediction']
        conditions_obj = exp_data['conditions']
        
        experiments.append({
            'exp_id': exp_id,
            'conditions': conditions_obj,
            'prediction': prediction
        })
        
        if prediction:
            print(f"   🔮 Predicted: {prediction.predicted_phase} (confidence: {prediction.confidence:.3f})")
        else:
            print(f"   ⚠️ No prediction available")
    
    print(f"\n✅ All experiments set up successfully!")
    
    # Simulate experimental results
    print(f"\n🧬 Simulating experimental synthesis and characterization...")
    
    for i, exp in enumerate(experiments):
        print(f"\n📊 Recording results for experiment {i+1}/{len(experiments)} ({exp['exp_id']})")
        
        # Generate simulated results
        results_data = generate_demo_experimental_results(
            exp['exp_id'], exp['prediction'], exp['conditions']
        )
        
        # Record results
        pipeline.record_experiment_results(exp['exp_id'], results_data)
        
        print(f"   🎯 Actual: {results_data['dominant_phase']} "
              f"(purity: {results_data['phase_purity']:.3f})")
        print(f"   ⚗️ Success: {'✅' if results_data['synthesis_success'] else '❌'}")
    
    print(f"\n📋 Generating validation analysis...")
    
    # Generate comprehensive validation report
    report = pipeline.validator.generate_validation_report()
    
    if report:
        print(f"\n📈 Validation Results Summary:")
        print(f"   Total Experiments: {report['total_experiments']}")
        print(f"   Phase Prediction Accuracy: {report['phase_prediction_accuracy_percent']:.1f}%")
        print(f"   Mean Confidence: {report['confidence_analysis']['mean_confidence']:.3f}")
        
        if report['property_accuracy_statistics']:
            print(f"   Property Prediction Errors:")
            for prop, stats in report['property_accuracy_statistics'].items():
                print(f"     {prop}: {stats['mean_error_percent']:.1f}% mean error")
    
    # Generate dashboard
    print(f"\n📊 Generating validation dashboard...")
    dashboard_file = pipeline.generate_validation_dashboard()
    print(f"   Dashboard saved to: {dashboard_file}")
    
    # Show pipeline status
    status = pipeline.get_pipeline_status()
    print(f"\n📋 Final Pipeline Status:")
    print(f"   Total experiments: {status['total_experiments']}")
    for status_type, count in status['by_status'].items():
        print(f"   {status_type}: {count}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📂 All demo data saved in: {demo_dir}/")
    print(f"\n💡 Next steps:")
    print(f"   1. Review generated lab notebooks")
    print(f"   2. Examine validation plots and dashboard")  
    print(f"   3. Use templates for real experiments")
    print(f"   4. Import actual experimental data")

def main():
    """Run the validation demo"""
    try:
        run_validation_demo()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()
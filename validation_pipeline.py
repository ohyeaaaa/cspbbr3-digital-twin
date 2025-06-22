#!/usr/bin/env python3
"""
Complete Experimental Validation Pipeline for CsPbBr₃ Digital Twin
Orchestrates the entire validation workflow from planning to analysis
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from experimental_validation import (
    ExperimentalValidator, ExperimentalConditions, ExperimentalResults, ModelPrediction
)
from experimental_data_templates import (
    create_experiment_template, create_results_template,
    create_csv_templates, create_json_templates, create_lab_notebook_template
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValidationPipeline:
    """Complete pipeline for experimental validation"""
    
    def __init__(self, data_dir: str = "experimental_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.validator = ExperimentalValidator(str(self.data_dir))
        
        # Pipeline state
        self.current_experiments = {}  # exp_id -> status
        
    def setup_new_experiment(self, conditions: Dict[str, Any] = None) -> str:
        """Set up a new experiment with optional conditions"""
        
        logger.info("🧪 Setting up new experiment")
        
        # Generate experiment ID
        exp_id = self.validator.generate_experiment_id()
        
        # Create experimental conditions
        if conditions is None:
            # Use default optimal conditions from synthesis protocol
            conditions_obj = ExperimentalConditions(
                experiment_id=exp_id,
                cs_br_concentration=1.1,
                pb_br2_concentration=1.1,
                temperature=190.0,
                oa_concentration=0.4,
                oam_concentration=0.2,
                reaction_time=75.0,
                solvent_type=0,  # DMSO
                date_conducted=datetime.now().strftime("%Y-%m-%d"),
                researcher="",
                notes="Validation experiment using optimal digital twin conditions"
            )
        else:
            conditions_obj = ExperimentalConditions(**conditions)
        
        # Record conditions and get prediction
        self.validator.record_experimental_conditions(conditions_obj)
        prediction = self.validator.get_model_prediction(conditions_obj)
        
        # Update pipeline state
        self.current_experiments[exp_id] = {
            'status': 'planned',
            'conditions': conditions_obj,
            'prediction': prediction,
            'setup_date': datetime.now().isoformat()
        }
        
        # Generate lab notebook
        self._generate_experiment_notebook(exp_id, conditions_obj, prediction)
        
        logger.info(f"✅ Experiment {exp_id} set up successfully")
        if prediction:
            logger.info(f"🔮 Model prediction: {prediction.predicted_phase} (confidence: {prediction.confidence:.3f})")
        
        return exp_id
    
    def _generate_experiment_notebook(self, exp_id: str, conditions: ExperimentalConditions, prediction: Optional[ModelPrediction]):
        """Generate customized lab notebook for specific experiment"""
        
        notebook_content = f"""# CsPbBr₃ Synthesis Lab Notebook - {exp_id}

## Experiment Information
- **Experiment ID**: {exp_id}
- **Date Planned**: {conditions.date_conducted}
- **Researcher**: {conditions.researcher or "_________________"}
- **Objective**: Validate digital twin predictions for CsPbBr₃ synthesis

## Digital Twin Prediction
"""
        
        if prediction:
            notebook_content += f"""- **Predicted Phase**: {prediction.predicted_phase}
- **Confidence**: {prediction.confidence:.3f}
- **Phase Probabilities**:
  - CsPbBr₃ 3D: {prediction.phase_probabilities['CsPbBr3_3D']:.3f}
  - Cs₄PbBr₆ 0D: {prediction.phase_probabilities['Cs4PbBr6_0D']:.3f}
  - CsPb₂Br₅ 2D: {prediction.phase_probabilities['CsPb2Br5_2D']:.3f}
  - Mixed: {prediction.phase_probabilities['Mixed']:.3f}
  - Failed: {prediction.phase_probabilities['Failed']:.3f}
- **Predicted Properties**:
  - Bandgap: {prediction.predicted_properties['bandgap']:.3f} eV
  - PLQY: {prediction.predicted_properties['plqy']:.3f}
  - Particle Size: {prediction.predicted_properties['particle_size']:.1f} nm
  - Emission Peak: {prediction.predicted_properties['emission_peak']:.1f} nm

## Synthesis Conditions (From Digital Twin Optimization)
"""
        else:
            notebook_content += "- **Model prediction not available**\n\n## Synthesis Conditions\n"
        
        solvent_names = {0: "DMSO", 1: "DMF", 2: "Toluene", 3: "Octadecene", 4: "Mixed"}
        
        notebook_content += f"""
### Reactant Concentrations
- **CsBr concentration**: {conditions.cs_br_concentration} mol/L
- **PbBr₂ concentration**: {conditions.pb_br2_concentration} mol/L
- **Cs:Pb molar ratio**: {conditions.cs_br_concentration/conditions.pb_br2_concentration:.2f}

### Ligands
- **Oleic acid concentration**: {conditions.oa_concentration} mol/L
- **Oleylamine concentration**: {conditions.oam_concentration} mol/L

### Reaction Parameters
- **Temperature**: {conditions.temperature} °C
- **Reaction time**: {conditions.reaction_time} minutes
- **Solvent**: {solvent_names.get(conditions.solvent_type, "Unknown")}
- **Atmosphere**: [ ] Nitrogen [✓] Argon [ ] Air
- **Stirring rate**: _________ rpm

### Equipment
- **Reaction vessel**: _________
- **Heating method**: _________
- **Total volume**: _________ mL

## Pre-Synthesis Checklist
- [ ] All chemicals weighed and ready
- [ ] CsBr solution prepared: _____ mL of {conditions.cs_br_concentration} mol/L
- [ ] PbBr₂ solution prepared: _____ mL of {conditions.pb_br2_concentration} mol/L
- [ ] Oleic acid measured: _____ mL ({conditions.oa_concentration} mol/L)
- [ ] Oleylamine measured: _____ mL ({conditions.oam_concentration} mol/L)
- [ ] Equipment cleaned and dried
- [ ] Inert atmosphere established
- [ ] Heating bath set to {conditions.temperature}°C
- [ ] Safety equipment ready
- [ ] Timer set for {conditions.reaction_time} minutes

## Synthesis Timeline

### Time: ___:___ | Pre-heating
- Target temperature: {conditions.temperature}°C
- Actual temperature: _____°C
- Observations: _________________________________

### Time: ___:___ | Injection
- CsBr solution added: [ ] Complete [ ] Partial
- PbBr₂ solution added: [ ] Complete [ ] Partial
- Initial color: _________________________________
- Temperature after injection: _____°C

### Time: ___:___ | +15 minutes
- Temperature: _____°C
- Color: _________________________________
- Clarity: [ ] Clear [ ] Turbid [ ] Opaque
- Precipitation: [ ] None [ ] Slight [ ] Heavy

### Time: ___:___ | +30 minutes
- Temperature: _____°C
- Color: _________________________________
- Observations: _________________________________

### Time: ___:___ | +45 minutes
- Temperature: _____°C
- Color: _________________________________
- Observations: _________________________________

### Time: ___:___ | +60 minutes
- Temperature: _____°C
- Color: _________________________________
- Observations: _________________________________

### Time: ___:___ | End ({conditions.reaction_time} min)
- Final temperature: _____°C
- Final color: _________________________________
- Solution clarity: [ ] Clear [ ] Turbid [ ] Opaque
- Precipitate: [ ] Yes [ ] No
- Overall assessment: [ ] Success [ ] Partial [ ] Failed

## Post-Synthesis
- **Cooling method**: _________
- **Final volume**: _________ mL
- **Yield estimate**: _________ mg
- **Storage**: [ ] 4°C [ ] RT [ ] Freezer [ ] Inert atmosphere
- **Aliquots prepared**: _________

## Characterization Schedule
- [ ] XRD analysis - Date: _________
- [ ] UV-vis spectroscopy - Date: _________
- [ ] Photoluminescence - Date: _________
- [ ] TEM imaging - Date: _________
- [ ] SEM imaging - Date: _________
- [ ] DLS size analysis - Date: _________

## Results Summary (Complete after characterization)

### Phase Analysis (XRD)
- **Dominant phase**: _________ vs Predicted: {prediction.predicted_phase if prediction else "N/A"}
- **Phase purity**: _________ %
- **Secondary phases**: _________
- **Prediction accuracy**: [ ] Correct [ ] Incorrect

### Optical Properties
- **Bandgap**: _________ eV vs Predicted: {prediction.predicted_properties['bandgap']:.3f} eV
- **Error**: _________ %
- **Emission peak**: _________ nm vs Predicted: {prediction.predicted_properties['emission_peak']:.1f} nm
- **Error**: _________ %
- **PLQY**: _________ vs Predicted: {prediction.predicted_properties['plqy']:.3f}
- **Error**: _________ %

### Morphology
- **Particle size**: _________ nm vs Predicted: {prediction.predicted_properties['particle_size']:.1f} nm
- **Error**: _________ %
- **Morphology**: _________
- **Size distribution**: _________

## Digital Twin Validation Summary
- **Phase prediction correct**: [ ] Yes [ ] No
- **Overall model performance**: [ ] Excellent [ ] Good [ ] Fair [ ] Poor
- **Most accurate property**: _________
- **Least accurate property**: _________
- **Confidence vs reality**: [ ] Well calibrated [ ] Overconfident [ ] Underconfident

## Lessons Learned
- **What worked well**: _________
- **What could be improved**: _________
- **Suggested model updates**: _________
- **Next experiments**: _________

## Data Files
- **Experiment folder**: {exp_id}/
- **XRD data**: {exp_id}_XRD.xy
- **UV-vis data**: {exp_id}_UVvis.csv
- **PL data**: {exp_id}_PL.csv
- **TEM images**: {exp_id}_TEM_*.tif
- **Analysis notebook**: {exp_id}_analysis.ipynb

---
**Experiment completed by**: _________________ **Date**: _________
**Data uploaded to validation system**: [ ] Yes [ ] No **Date**: _________
"""
        
        # Save notebook
        notebook_file = self.data_dir / f"{exp_id}_lab_notebook.md"
        with open(notebook_file, 'w') as f:
            f.write(notebook_content)
        
        logger.info(f"📓 Lab notebook generated: {notebook_file}")
        return notebook_file
    
    def record_experiment_results(self, exp_id: str, results_data: Dict[str, Any]):
        """Record experimental results and trigger validation analysis"""
        
        if exp_id not in self.current_experiments:
            logger.error(f"Experiment {exp_id} not found in pipeline")
            return False
        
        logger.info(f"📊 Recording results for experiment {exp_id}")
        
        # Create results object
        results = ExperimentalResults(
            experiment_id=exp_id,
            **results_data
        )
        
        # Record results
        self.validator.record_experimental_results(results)
        
        # Update pipeline state
        self.current_experiments[exp_id]['status'] = 'completed'
        self.current_experiments[exp_id]['results'] = results
        self.current_experiments[exp_id]['completion_date'] = datetime.now().isoformat()
        
        logger.info(f"✅ Results recorded for {exp_id}")
        
        # Generate experiment summary
        self._generate_experiment_summary(exp_id)
        
        return True
    
    def _generate_experiment_summary(self, exp_id: str):
        """Generate summary report for completed experiment"""
        
        exp_data = self.current_experiments[exp_id]
        conditions = exp_data['conditions']
        prediction = exp_data['prediction']
        results = exp_data['results']
        
        # Load validation analysis
        analysis_file = self.data_dir / "validation_analysis.json"
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                analyses = json.load(f)
            
            # Find this experiment's analysis
            exp_analysis = None
            for analysis in analyses:
                if analysis['experiment_id'] == exp_id:
                    exp_analysis = analysis
                    break
        else:
            exp_analysis = None
        
        # Generate summary
        summary = {
            'experiment_id': exp_id,
            'summary_date': datetime.now().isoformat(),
            'experiment_duration': self._calculate_duration(
                exp_data['setup_date'], exp_data['completion_date']
            ),
            'conditions_summary': {
                'cs_pb_ratio': conditions.cs_br_concentration / conditions.pb_br2_concentration,
                'temperature': conditions.temperature,
                'reaction_time': conditions.reaction_time,
                'solvent': ['DMSO', 'DMF', 'Toluene', 'Octadecene', 'Mixed'][conditions.solvent_type]
            },
            'prediction_summary': {
                'predicted_phase': prediction.predicted_phase if prediction else None,
                'confidence': prediction.confidence if prediction else None,
                'top_predicted_property': max(prediction.predicted_properties.items(), key=lambda x: abs(x[1])) if prediction else None
            },
            'results_summary': {
                'actual_phase': results.dominant_phase,
                'synthesis_success': results.synthesis_success,
                'characterization_methods': results.characterization_methods,
                'key_properties': {
                    'bandgap': results.bandgap,
                    'plqy': results.plqy,
                    'particle_size': results.particle_size,
                    'emission_peak': results.emission_peak
                }
            },
            'validation_summary': exp_analysis,
            'key_insights': self._extract_insights(conditions, prediction, results, exp_analysis)
        }
        
        # Save summary
        summary_file = self.data_dir / f"{exp_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📋 Experiment summary generated: {summary_file}")
        return summary_file
    
    def _calculate_duration(self, start_iso: str, end_iso: str) -> str:
        """Calculate experiment duration in human-readable format"""
        try:
            start = datetime.fromisoformat(start_iso)
            end = datetime.fromisoformat(end_iso)
            duration = end - start
            return str(duration)
        except:
            return "Unknown"
    
    def _extract_insights(self, conditions, prediction, results, analysis) -> List[str]:
        """Extract key insights from experiment"""
        insights = []
        
        if analysis:
            if analysis['phase_prediction_correct']:
                insights.append(f"✅ Model correctly predicted {results.dominant_phase} phase")
            else:
                insights.append(f"❌ Model predicted {prediction.predicted_phase}, got {results.dominant_phase}")
            
            if 'property_accuracies' in analysis:
                best_prop = min(analysis['property_accuracies'].items(), 
                              key=lambda x: x[1]['relative_error_percent'])
                worst_prop = max(analysis['property_accuracies'].items(), 
                               key=lambda x: x[1]['relative_error_percent'])
                
                insights.append(f"Best property prediction: {best_prop[0]} ({best_prop[1]['relative_error_percent']:.1f}% error)")
                insights.append(f"Worst property prediction: {worst_prop[0]} ({worst_prop[1]['relative_error_percent']:.1f}% error)")
        
        # Synthesis insights
        if results.synthesis_success:
            if conditions.temperature >= 190:
                insights.append("High temperature (≥190°C) contributed to synthesis success")
            if abs(conditions.cs_br_concentration - conditions.pb_br2_concentration) < 0.1:
                insights.append("Balanced Cs:Pb stoichiometry supported successful synthesis")
        
        return insights
    
    def run_batch_validation(self, conditions_list: List[Dict[str, Any]]) -> List[str]:
        """Set up multiple validation experiments"""
        
        logger.info(f"🔬 Setting up batch validation with {len(conditions_list)} experiments")
        
        exp_ids = []
        for i, conditions in enumerate(conditions_list):
            logger.info(f"Setting up experiment {i+1}/{len(conditions_list)}")
            exp_id = self.setup_new_experiment(conditions)
            exp_ids.append(exp_id)
        
        logger.info(f"✅ Batch validation setup complete. Experiment IDs: {exp_ids}")
        return exp_ids
    
    def generate_validation_dashboard(self) -> str:
        """Generate comprehensive validation dashboard"""
        
        # Generate overall validation report
        report = self.validator.generate_validation_report()
        
        # Generate plots
        self.validator.plot_validation_results(save_plots=True)
        
        # Create dashboard HTML
        dashboard_content = self._create_dashboard_html(report)
        
        dashboard_file = self.data_dir / f"validation_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_content)
        
        logger.info(f"📊 Validation dashboard generated: {dashboard_file}")
        return str(dashboard_file)
    
    def _create_dashboard_html(self, report: Dict[str, Any]) -> str:
        """Create HTML dashboard from validation report"""
        
        if not report:
            return "<html><body><h1>No validation data available</h1></body></html>"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CsPbBr₃ Digital Twin Validation Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .metric {{ background: #ecf0f1; padding: 15px; margin: 10px; border-radius: 5px; }}
        .success {{ background: #d5f4e6; }}
        .warning {{ background: #ffeaa7; }}
        .error {{ background: #fab1a0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #34495e; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CsPbBr₃ Digital Twin Validation Dashboard</h1>
        <p>Generated on {report['report_date']}</p>
    </div>
    
    <div class="metric {'success' if report['phase_prediction_accuracy_percent'] > 70 else 'warning' if report['phase_prediction_accuracy_percent'] > 50 else 'error'}">
        <h2>Overall Performance</h2>
        <p><strong>Phase Prediction Accuracy:</strong> {report['phase_prediction_accuracy_percent']:.1f}%</p>
        <p><strong>Total Experiments:</strong> {report['total_experiments']}</p>
        <p><strong>Successful Predictions:</strong> {report['experiments_with_correct_phase']}</p>
    </div>
    
    <div class="metric">
        <h2>Confidence Analysis</h2>
        <p><strong>Mean Confidence:</strong> {report['confidence_analysis']['mean_confidence']:.3f}</p>
        <p><strong>Confidence vs Accuracy Correlation:</strong> {report['confidence_analysis']['confidence_vs_accuracy_correlation']:.3f}</p>
    </div>
"""
        
        # Property accuracies
        if report['property_accuracy_statistics']:
            html_content += """
    <div class="metric">
        <h2>Property Prediction Accuracy</h2>
        <table>
            <tr><th>Property</th><th>Mean Error (%)</th><th>Median Error (%)</th><th>Measurements</th></tr>
"""
            for prop, stats in report['property_accuracy_statistics'].items():
                html_content += f"""
            <tr>
                <td>{prop.replace('_', ' ').title()}</td>
                <td>{stats['mean_error_percent']:.1f}%</td>
                <td>{stats['median_error_percent']:.1f}%</td>
                <td>{stats['n_measurements']}</td>
            </tr>
"""
            html_content += """
        </table>
    </div>
"""
        
        # Recent experiments
        recent_experiments = report['individual_experiments'][-5:] if len(report['individual_experiments']) > 5 else report['individual_experiments']
        html_content += """
    <div class="metric">
        <h2>Recent Experiments</h2>
        <table>
            <tr><th>Experiment ID</th><th>Predicted Phase</th><th>Actual Phase</th><th>Correct</th><th>Confidence</th></tr>
"""
        for exp in recent_experiments:
            correct_icon = "✅" if exp['phase_prediction_correct'] else "❌"
            html_content += f"""
            <tr>
                <td>{exp['experiment_id']}</td>
                <td>{exp['predicted_phase']}</td>
                <td>{exp['actual_phase']}</td>
                <td>{correct_icon}</td>
                <td>{exp['prediction_confidence']:.3f}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
    
    <div class="metric">
        <h2>Recommendations</h2>
        <ul>
"""
        
        # Generate recommendations
        accuracy = report['phase_prediction_accuracy_percent']
        if accuracy < 50:
            html_content += "<li>⚠️ Model accuracy is low. Consider retraining with more experimental data.</li>"
        elif accuracy < 70:
            html_content += "<li>📈 Model shows moderate accuracy. Focus on improving property predictions.</li>"
        else:
            html_content += "<li>✅ Model shows good accuracy. Continue validation and fine-tuning.</li>"
        
        if report['confidence_analysis']['confidence_vs_accuracy_correlation'] < 0.3:
            html_content += "<li>🎯 Confidence calibration needs improvement. Model may be over/under-confident.</li>"
        
        html_content += """
        </ul>
    </div>
    
</body>
</html>
"""
        
        return html_content
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        
        status = {
            'total_experiments': len(self.current_experiments),
            'by_status': {},
            'recent_activity': []
        }
        
        # Count by status
        for exp_data in self.current_experiments.values():
            status_key = exp_data['status']
            status['by_status'][status_key] = status['by_status'].get(status_key, 0) + 1
        
        # Recent activity
        sorted_experiments = sorted(
            self.current_experiments.items(),
            key=lambda x: x[1]['setup_date'],
            reverse=True
        )
        
        for exp_id, exp_data in sorted_experiments[:5]:
            status['recent_activity'].append({
                'experiment_id': exp_id,
                'status': exp_data['status'],
                'setup_date': exp_data['setup_date']
            })
        
        return status

def main():
    """Main pipeline interface"""
    
    parser = argparse.ArgumentParser(description="CsPbBr₃ Experimental Validation Pipeline")
    parser.add_argument("command", choices=["setup", "record", "batch", "dashboard", "status"],
                       help="Pipeline command to execute")
    parser.add_argument("--data-dir", type=str, default="experimental_data",
                       help="Data directory for validation files")
    parser.add_argument("--exp-id", type=str, help="Experiment ID for record command")
    parser.add_argument("--conditions", type=str, help="JSON file with experimental conditions")
    parser.add_argument("--results", type=str, help="JSON file with experimental results")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ValidationPipeline(args.data_dir)
    
    if args.command == "setup":
        # Set up new experiment
        conditions = None
        if args.conditions and Path(args.conditions).exists():
            with open(args.conditions, 'r') as f:
                conditions = json.load(f)
        
        exp_id = pipeline.setup_new_experiment(conditions)
        print(f"✅ Experiment {exp_id} set up successfully")
        
    elif args.command == "record":
        # Record experimental results
        if not args.exp_id:
            print("❌ --exp-id required for record command")
            return
        
        if not args.results or not Path(args.results).exists():
            print("❌ --results file required and must exist")
            return
        
        with open(args.results, 'r') as f:
            results_data = json.load(f)
        
        success = pipeline.record_experiment_results(args.exp_id, results_data)
        if success:
            print(f"✅ Results recorded for {args.exp_id}")
        else:
            print(f"❌ Failed to record results for {args.exp_id}")
            
    elif args.command == "batch":
        # Set up batch experiments
        if not args.conditions or not Path(args.conditions).exists():
            print("❌ --conditions file required for batch command")
            return
        
        with open(args.conditions, 'r') as f:
            conditions_list = json.load(f)
        
        exp_ids = pipeline.run_batch_validation(conditions_list)
        print(f"✅ Batch validation set up: {len(exp_ids)} experiments")
        
    elif args.command == "dashboard":
        # Generate validation dashboard
        dashboard_file = pipeline.generate_validation_dashboard()
        print(f"📊 Dashboard generated: {dashboard_file}")
        
    elif args.command == "status":
        # Show pipeline status
        status = pipeline.get_pipeline_status()
        print("📋 Pipeline Status:")
        print(f"   Total experiments: {status['total_experiments']}")
        for status_type, count in status['by_status'].items():
            print(f"   {status_type}: {count}")

if __name__ == "__main__":
    main()
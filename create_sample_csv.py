#!/usr/bin/env python3
"""
Simple CSV creator for CsPbBr3 Digital Twin (no dependencies)
Creates basic sample data to demonstrate structure
"""

import csv
import random
import math
import os
from pathlib import Path

def create_simple_sample_data(n_samples=100):
    """Create simple sample data without numpy/pandas"""
    
    print(f"📊 Creating {n_samples} sample data points...")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Define data columns
    columns = [
        'sample_id', 'cs_br_concentration', 'pb_br2_concentration', 'temperature',
        'solvent_type', 'oa_concentration', 'oam_concentration', 'reaction_time',
        'phase_label', 'bandgap', 'plqy', 'emission_peak', 'emission_fwhm',
        'particle_size', 'size_distribution_width', 'lifetime', 'stability_score', 'phase_purity'
    ]
    
    # Create sample data
    data_rows = []
    
    for i in range(n_samples):
        # Random synthesis parameters
        cs_conc = round(random.uniform(0.5, 2.5), 2)
        pb_conc = round(random.uniform(0.5, 2.0), 2)
        temp = round(random.uniform(100, 200), 1)
        solvent = random.randint(0, 4)  # 0=DMSO, 1=DMF, 2=Water, 3=Toluene, 4=Octadecene
        oa_conc = round(random.uniform(0, 0.5), 3)
        oam_conc = round(random.uniform(0, 0.5), 3)
        time = round(random.uniform(5, 30), 1)
        
        # Simple logic for phase determination
        cs_pb_ratio = cs_conc / pb_conc
        if cs_pb_ratio > 1.2 and temp < 150:
            phase = 1  # 0D phase
        elif cs_pb_ratio < 0.8 and temp > 150:
            phase = 2  # 2D phase
        elif temp < 120 or cs_conc < 0.5:
            phase = 4  # Failed
        elif random.random() < 0.1:
            phase = 3  # Mixed
        else:
            phase = 0  # 3D phase (most common)
        
        # Properties based on phase
        if phase == 0:  # 3D CsPbBr3
            bandgap = round(2.3 + random.gauss(0, 0.1), 2)
            plqy = round(0.8 + random.gauss(0, 0.1), 3)
            emission = round(520 + random.gauss(0, 10), 1)
            size = round(10 + random.gauss(0, 2), 1)
        elif phase == 1:  # 0D Cs4PbBr6
            bandgap = round(3.9 + random.gauss(0, 0.2), 2)
            plqy = round(0.15 + random.gauss(0, 0.05), 3)
            emission = round(410 + random.gauss(0, 15), 1)
            size = round(5 + random.gauss(0, 1), 1)
        elif phase == 2:  # 2D CsPb2Br5
            bandgap = round(2.9 + random.gauss(0, 0.15), 2)
            plqy = round(0.45 + random.gauss(0, 0.1), 3)
            emission = round(460 + random.gauss(0, 12), 1)
            size = round(8 + random.gauss(0, 1.5), 1)
        elif phase == 3:  # Mixed
            bandgap = round(2.6 + random.gauss(0, 0.3), 2)
            plqy = round(0.35 + random.gauss(0, 0.15), 3)
            emission = round(480 + random.gauss(0, 20), 1)
            size = round(9 + random.gauss(0, 2), 1)
        else:  # Failed
            bandgap = 0.0
            plqy = 0.0
            emission = 0.0
            size = 0.0
        
        # Ensure positive values
        bandgap = max(0.5, bandgap) if phase != 4 else 0.0
        plqy = max(0.0, min(1.0, plqy))
        emission = max(300, emission) if phase != 4 else 0.0
        size = max(1.0, size) if phase != 4 else 0.0
        
        # Additional properties
        fwhm = round(20 + random.gauss(0, 3), 1) if phase != 4 else 0.0
        width = round(0.2 + random.uniform(0, 0.3), 2)
        lifetime = round(15 + random.gauss(0, 5), 1) if phase == 0 else round(5 + random.gauss(0, 2), 1)
        lifetime = max(0.5, lifetime) if phase != 4 else 0.0
        stability = round(0.7 + random.uniform(-0.2, 0.3), 2)
        stability = max(0.0, min(1.0, stability))
        purity = round(0.9 + random.uniform(-0.1, 0.1), 2) if phase != 3 else round(0.6, 2)
        purity = max(0.0, min(1.0, purity))
        
        row = [
            i, cs_conc, pb_conc, temp, solvent, oa_conc, oam_conc, time,
            phase, bandgap, plqy, emission, fwhm, size, width, lifetime, stability, purity
        ]
        data_rows.append(row)
    
    # Write to CSV
    csv_path = "data/sample_training_data.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(data_rows)
    
    print(f"✅ Sample data saved to: {csv_path}")
    
    # Print summary
    phase_counts = {}
    for row in data_rows:
        phase = row[8]  # phase_label column
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    print("\n📈 Dataset Summary:")
    print(f"   Total samples: {len(data_rows)}")
    print("   Phase distribution:")
    phase_names = {0: 'CsPbBr3_3D', 1: 'Cs4PbBr6_0D', 2: 'CsPb2Br5_2D', 3: 'Mixed', 4: 'Failed'}
    for phase, count in sorted(phase_counts.items()):
        percentage = count / len(data_rows) * 100
        print(f"     {phase_names[phase]}: {count} ({percentage:.1f}%)")
    
    return csv_path

def main():
    """Main function"""
    print("🧪 CsPbBr3 Digital Twin - Simple Sample Data Creator")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create datasets
    datasets = [
        (200, "Small dataset"),
        (1000, "Medium dataset")
    ]
    
    for n_samples, description in datasets:
        print(f"\n🎲 Creating {description} ({n_samples} samples)...")
        csv_path = create_simple_sample_data(n_samples)
        
        # Rename for specific size
        new_path = f"data/sample_data_{n_samples}.csv"
        os.rename(csv_path, new_path)
        print(f"   Saved as: {new_path}")
    
    print("\n🎉 Sample data creation complete!")
    print("\n📝 Data columns created:")
    columns = [
        'sample_id', 'cs_br_concentration', 'pb_br2_concentration', 'temperature',
        'solvent_type', 'oa_concentration', 'oam_concentration', 'reaction_time',
        'phase_label', 'bandgap', 'plqy', 'emission_peak', 'emission_fwhm',
        'particle_size', 'size_distribution_width', 'lifetime', 'stability_score', 'phase_purity'
    ]
    for i, col in enumerate(columns):
        print(f"   {i+1:2d}. {col}")
    
    print("\n📚 Next steps:")
    print("1. Install dependencies: ./setup_environment.sh")
    print("2. Train model: python train_pytorch_models.py --data-file data/sample_data_1000.csv")
    print("3. Test predictions: python quick_start_example.py")

if __name__ == "__main__":
    main()
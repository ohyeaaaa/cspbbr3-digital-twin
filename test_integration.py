#!/usr/bin/env python3
"""
Integration Test for CsPbBr₃ Digital Twin
Tests the complete pipeline from parameters to predictions
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Test imports
try:
    from synthesis import (
        create_complete_digital_twin,
        quick_prediction,
        SynthesisParameters,
        SolventType,
        PhaseType
    )
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_quick_prediction():
    """Test quick prediction functionality"""
    print("\n🧪 Testing quick prediction...")
    
    try:
        result = quick_prediction(
            cs_concentration=1.0,
            pb_concentration=1.0,
            temperature=150.0,
            solvent="DMSO"
        )
        
        print(f"✅ Prediction successful!")
        print(f"   Primary phase: {result.primary_phase.name}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Processing time: {result.processing_time_ms:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick prediction failed: {e}")
        return False

def test_digital_twin_creation():
    """Test digital twin creation and detailed prediction"""
    print("\n🏗️ Testing digital twin creation...")
    
    try:
        # Create digital twin
        twin = create_complete_digital_twin()
        print("✅ Digital twin created successfully")
        
        # Test health check
        health = twin.health_check()
        print(f"✅ Health check: {health['status']}")
        
        # Create detailed parameters
        params = SynthesisParameters(
            cs_br_concentration=1.5,
            pb_br2_concentration=1.0,
            temperature=160.0,
            solvent_type=SolventType.DMSO,
            oa_concentration=0.1,
            oam_concentration=0.1,
            reaction_time=15.0
        )
        
        # Make prediction
        result = twin.predict(params, return_features=True, enable_uncertainty=True)
        
        print("✅ Detailed prediction successful!")
        print(f"   Primary phase: {result.primary_phase.name}")
        print(f"   Phase probabilities:")
        for phase, prob in result.phase_probabilities.items():
            print(f"     {phase.name}: {prob:.3f}")
        
        print(f"   Material properties:")
        print(f"     Bandgap: {result.properties.bandgap:.2f} eV")
        print(f"     PLQY: {result.properties.plqy:.3f}")
        print(f"     Particle size: {result.properties.particle_size:.1f} nm")
        
        print(f"   Physics features:")
        print(f"     Supersaturation: {result.physics_features.supersaturation:.2f}")
        print(f"     Nucleation rate (3D): {result.physics_features.nucleation_rate_3d:.2e}")
        
        print(f"   Uncertainty: {result.phase_uncertainty:.3f}")
        print(f"   Physics consistency: {result.physics_consistency:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Digital twin test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_prediction():
    """Test batch prediction functionality"""
    print("\n📊 Testing batch prediction...")
    
    try:
        twin = create_complete_digital_twin()
        
        # Create multiple parameter sets
        param_sets = [
            SynthesisParameters(
                cs_br_concentration=1.0,
                pb_br2_concentration=1.0,
                temperature=150.0,
                solvent_type=SolventType.DMSO
            ),
            SynthesisParameters(
                cs_br_concentration=2.0,
                pb_br2_concentration=1.5,
                temperature=180.0,
                solvent_type=SolventType.DMF
            ),
            SynthesisParameters(
                cs_br_concentration=0.5,
                pb_br2_concentration=0.8,
                temperature=120.0,
                solvent_type=SolventType.TOLUENE
            )
        ]
        
        # Batch prediction
        results = twin.batch_predict(param_sets, batch_size=2)
        
        print(f"✅ Batch prediction successful for {len(results)} samples!")
        for i, result in enumerate(results):
            print(f"   Sample {i+1}: {result.primary_phase.name} (confidence: {result.confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        return False

def test_performance_stats():
    """Test performance monitoring"""
    print("\n📈 Testing performance monitoring...")
    
    try:
        twin = create_complete_digital_twin()
        
        # Make a few predictions
        for i in range(3):
            quick_prediction(1.0 + i*0.5, 1.0, 150.0 + i*10, "DMSO")
        
        # Get performance stats
        stats = twin.get_performance_stats()
        
        print("✅ Performance stats retrieved!")
        print(f"   Total predictions: {stats['total_predictions']}")
        print(f"   Average processing time: {stats['average_processing_time_ms']:.1f}ms")
        print(f"   Device: {stats['device']}")
        print(f"   Models loaded: {stats['model_loaded']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance stats test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 CsPbBr₃ Digital Twin Integration Test")
    print("=" * 50)
    
    # Suppress excessive logging for cleaner output
    logging.getLogger().setLevel(logging.WARNING)
    
    tests = [
        test_quick_prediction,
        test_digital_twin_creation,
        test_batch_prediction,
        test_performance_stats
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Digital twin integration is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Quick test to verify the simulation works"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'link'))

def test_imports():
    """Test that all modules can be imported"""
    try:
        import waveforms
        print("✓ waveforms module imported")
        
        import devices  
        print("✓ devices module imported")
        
        import channel
        print("✓ channel module imported")
        
        import link_budget
        print("✓ link_budget module imported")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_simulation():
    """Test basic simulation components"""
    try:
        from waveforms import WaveformConfig, generate_prbs
        from devices import MZMConfig, MachZehnderModulator
        
        # Test PRBS generation
        prbs = generate_prbs(10, seed=42)
        print(f"✓ Generated PRBS pattern of length {len(prbs)}")
        
        # Test MZM
        mzm_config = MZMConfig()
        mzm = MachZehnderModulator(mzm_config)
        print(f"✓ Created MZM with Vπ = {mzm.vpi} V")
        
        return True
    except Exception as e:
        print(f"✗ Simulation error: {e}")
        return False

def main():
    print("Silicon Photonics Transceiver - Installation Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\nPlease install required packages:")
        print("  pip install numpy matplotlib pandas scipy")
        return 1
    
    # Test basic functionality
    if not test_basic_simulation():
        return 1
    
    print("\n✓ All tests passed! The simulation is ready to run.")
    print("\nTo run the demo:")
    print("  cd src/link")
    print("  python3 sim_link_nrz_pam4.py --bitrate 25 --modulation NRZ --smoke")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
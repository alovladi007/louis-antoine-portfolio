#!/usr/bin/env python3
"""
Test script to verify project structure and basic functionality
This script tests the project without requiring external packages
"""

import os
import sys
import json
from pathlib import Path

def test_project_structure():
    """Test that all required directories and files exist"""
    project_root = Path(__file__).parent
    
    required_dirs = [
        'scripts',
        'config', 
        'examples',
        'docs',
        'results'
    ]
    
    required_files = [
        'README.md',
        'requirements.txt',
        'LICENSE',
        '.gitignore',
        'scripts/optimize_modulator.lsf',
        'scripts/optimize_pd.lsf', 
        'scripts/build_interconnect.icp',
        'scripts/analysis_visualization.py',
        'scripts/run_all_simulations.py',
        'config/simulation_config.json',
        'examples/run_example.py'
    ]
    
    print("🔍 Testing project structure...")
    print("-" * 40)
    
    # Test directories
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ Directory: {dir_name}")
        else:
            print(f"❌ Missing directory: {dir_name}")
            missing_dirs.append(dir_name)
    
    # Test files
    missing_files = []
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists() and file_path.is_file():
            print(f"✅ File: {file_name}")
        else:
            print(f"❌ Missing file: {file_name}")
            missing_files.append(file_name)
    
    return len(missing_dirs) == 0 and len(missing_files) == 0

def test_config_file():
    """Test that configuration file is valid JSON"""
    print("\n🔧 Testing configuration file...")
    print("-" * 35)
    
    config_path = Path(__file__).parent / 'config' / 'simulation_config.json'
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_sections = ['simulation_parameters', 'file_paths', 'analysis_settings']
        
        for section in required_sections:
            if section in config:
                print(f"✅ Config section: {section}")
            else:
                print(f"❌ Missing config section: {section}")
                return False
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        return False
    except FileNotFoundError:
        print("❌ Config file not found")
        return False

def test_script_syntax():
    """Test that Python scripts have valid syntax"""
    print("\n🐍 Testing Python script syntax...")
    print("-" * 37)
    
    project_root = Path(__file__).parent
    python_scripts = [
        'scripts/analysis_visualization.py',
        'scripts/run_all_simulations.py',
        'examples/run_example.py'
    ]
    
    all_valid = True
    for script_path in python_scripts:
        full_path = project_root / script_path
        try:
            with open(full_path, 'r') as f:
                code = f.read()
            
            compile(code, full_path, 'exec')
            print(f"✅ Syntax valid: {script_path}")
            
        except SyntaxError as e:
            print(f"❌ Syntax error in {script_path}: {e}")
            all_valid = False
        except FileNotFoundError:
            print(f"❌ File not found: {script_path}")
            all_valid = False
    
    return all_valid

def test_lumerical_scripts():
    """Test that Lumerical scripts exist and have content"""
    print("\n📡 Testing Lumerical scripts...")
    print("-" * 32)
    
    project_root = Path(__file__).parent
    lumerical_scripts = [
        ('scripts/optimize_modulator.lsf', 'Ring modulator FDTD script'),
        ('scripts/optimize_pd.lsf', 'Photodetector FDTD script'),
        ('scripts/build_interconnect.icp', 'INTERCONNECT system script')
    ]
    
    all_valid = True
    for script_path, description in lumerical_scripts:
        full_path = project_root / script_path
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            
            if len(content.strip()) > 100:  # Basic content check
                print(f"✅ {description}: {len(content)} chars")
            else:
                print(f"❌ {description}: Too short or empty")
                all_valid = False
                
        except FileNotFoundError:
            print(f"❌ {description}: File not found")
            all_valid = False
    
    return all_valid

def generate_test_report():
    """Generate a test report"""
    print("\n" + "=" * 60)
    print("SILICON PHOTONICS TRANSCEIVER PROJECT TEST REPORT")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Configuration File", test_config_file),
        ("Python Script Syntax", test_script_syntax),
        ("Lumerical Scripts", test_lumerical_scripts)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error running {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n📊 TEST SUMMARY")
    print("-" * 15)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Project structure is correct.")
        print("📁 Ready to run simulations!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = generate_test_report()
    sys.exit(0 if success else 1)
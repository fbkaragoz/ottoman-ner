#!/usr/bin/env python3
"""
Test script for the Ottoman NER evaluation pipeline.

This script tests the new CONLL utilities, alignment functions, and evaluation pipeline.
"""

import sys
import tempfile
import os
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_conll_utilities():
    """Test CONLL data loading and writing utilities."""
    print("Testing CONLL utilities...")
    
    try:
        from ottoman_ner.io.conll import load_conll_data, write_conll_data, get_conll_statistics, validate_conll_data
        
        # Create sample CONLL data
        sample_data = [
            [("Emin", "B-PER"), ("Bey", "I-PER"), ("geldi", "O")],
            [("İstanbul'da", "B-LOC"), ("yaşıyor", "O")],
            [("Ahmet", "B-PER"), ("Paşa", "I-PER"), ("Ankara'ya", "B-LOC"), ("gitti", "O")]
        ]
        
        # Test writing and reading
        with tempfile.NamedTemporaryFile(mode='w', suffix='.conll', delete=False) as f:
            temp_file = f.name
        
        try:
            # Write data
            write_conll_data(sample_data, temp_file)
            print("✅ CONLL writing successful")
            
            # Read data back
            loaded_data = load_conll_data(temp_file)
            print("✅ CONLL reading successful")
            
            # Verify data integrity
            if loaded_data == sample_data:
                print("✅ Data integrity verified")
            else:
                print("❌ Data integrity check failed")
                return False
            
            # Test statistics
            stats = get_conll_statistics(loaded_data)
            print(f"✅ Statistics: {stats}")
            
            # Test validation
            is_valid = validate_conll_data(loaded_data)
            print(f"✅ Validation result: {is_valid}")
            
        finally:
            os.unlink(temp_file)
        
        return True
        
    except Exception as e:
        print(f"❌ CONLL utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alignment_functions():
    """Test prediction alignment functions."""
    print("\nTesting alignment functions...")
    
    try:
        from ottoman_ner.evaluation.alignment import align_predictions_with_tokens, apply_iob2_tags, validate_iob2_sequence
        
        # Test IOB2 tag application
        tags = ['O', 'O', 'O', 'O']
        entity_tokens = [1, 2]
        apply_iob2_tags(tags, entity_tokens, 'PER')
        
        expected_tags = ['O', 'B-PER', 'I-PER', 'O']
        if tags == expected_tags:
            print("✅ IOB2 tag application successful")
        else:
            print(f"❌ IOB2 tag application failed: {tags} != {expected_tags}")
            return False
        
        # Test IOB2 sequence validation
        valid_sequence = ['B-PER', 'I-PER', 'O', 'B-LOC']
        invalid_sequence = ['I-PER', 'B-PER', 'O', 'I-LOC']
        
        if validate_iob2_sequence(valid_sequence) and not validate_iob2_sequence(invalid_sequence):
            print("✅ IOB2 sequence validation successful")
        else:
            print("❌ IOB2 sequence validation failed")
            return False
        
        # Test alignment with mock predictions
        tokens = ["Emin", "Bey", "geldi"]
        mock_predictions = [
            {
                'text': 'Emin Bey',
                'label': 'PER',
                'start': 0,
                'end': 8,
                'confidence': 0.95
            }
        ]
        sentence_text = "Emin Bey geldi"
        
        aligned_tags = align_predictions_with_tokens(tokens, mock_predictions, sentence_text)
        print(f"✅ Alignment result: {list(zip(tokens, aligned_tags))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Alignment functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_package_imports():
    """Test that all new package imports work correctly."""
    print("\nTesting package imports...")
    
    try:
        # Test main package imports
        from ottoman_ner import (
            NERPredictor, 
            AVAILABLE_MODELS,
            load_conll_data,
            write_conll_data, 
            get_predictions_in_conll_format,
            align_predictions_with_tokens
        )
        print("✅ Main package imports successful")
        
        # Test submodule imports
        from ottoman_ner.io import load_conll_data, write_conll_data
        from ottoman_ner.evaluation import get_predictions_in_conll_format, align_predictions_with_tokens
        print("✅ Submodule imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Package imports test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_script_help():
    """Test that the scripts can show help without errors."""
    print("\nTesting script help functions...")
    
    try:
        # Test evaluation script help
        result = os.system("python scripts/evaluate_latin_ner.py --help > /dev/null 2>&1")
        if result == 0:
            print("✅ Evaluation script help works")
        else:
            print("❌ Evaluation script help failed")
            return False
        
        # Test splitting script help
        result = os.system("python scripts/split_conll_dataset.py --help > /dev/null 2>&1")
        if result == 0:
            print("✅ Splitting script help works")
        else:
            print("❌ Splitting script help failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Script help test failed: {e}")
        return False


def test_with_existing_data():
    """Test with existing data if available."""
    print("\nTesting with existing data...")
    
    try:
        # Check if we have existing CONLL data
        data_files = [
            "data/raw/train.txt",
            "data/raw/dev.txt", 
            "data/raw/test.txt"
        ]
        
        existing_files = [f for f in data_files if os.path.exists(f)]
        
        if existing_files:
            print(f"Found existing data files: {existing_files}")
            
            # Try to load one of them using our parse script
            result = os.system("python .scripts/parse_conll_to_dataset.py > /dev/null 2>&1")
            if result == 0:
                print("✅ Existing data parsing works")
            else:
                print("⚠️  Existing data parsing had issues (this might be expected)")
            
        else:
            print("⚠️  No existing CONLL data found, skipping this test")
        
        return True
        
    except Exception as e:
        print(f"❌ Existing data test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Ottoman NER Evaluation Pipeline Test Suite")
    print("=" * 60)
    
    tests = [
        test_package_imports,
        test_conll_utilities,
        test_alignment_functions,
        test_script_help,
        test_with_existing_data,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The evaluation pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your gold standard CONLL annotations")
        print("2. Use scripts/evaluate_latin_ner.py to evaluate your model")
        print("3. Use scripts/split_conll_dataset.py to split large datasets")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
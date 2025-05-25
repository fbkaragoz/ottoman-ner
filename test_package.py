#!/usr/bin/env python3
"""
Test script for Ottoman NER package.
"""

import sys
import os
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_package_import():
    """Test basic package import."""
    print("Testing package import...")
    try:
        from ottoman_ner import NERPredictor, AVAILABLE_MODELS
        print("✅ Package import successful")
        print(f"Available models: {list(AVAILABLE_MODELS.keys())}")
        return True
    except ImportError as e:
        print(f"❌ Package import failed: {e}")
        return False

def test_model_info():
    """Test model information functions."""
    print("\nTesting model information...")
    try:
        from ottoman_ner.model_config import list_available_models, get_model_path
        
        models_info = list_available_models()
        print("✅ Model info retrieval successful")
        print(f"Remote models: {list(models_info['remote_models'].keys())}")
        print(f"Local models: {list(models_info['local_models'].keys())}")
        
        # Test model path resolution
        try:
            latin_path = get_model_path("latin", use_local=True)
            print(f"✅ Local Latin model path: {latin_path}")
        except ValueError as e:
            print(f"⚠️  Local model not found: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Model info test failed: {e}")
        return False

def test_predictor_with_dummy():
    """Test predictor with a dummy/mock setup."""
    print("\nTesting NERPredictor initialization...")
    try:
        # Check if we have a local model
        model_path = "./models/ottoman_ner_latin"
        if os.path.exists(model_path):
            print(f"Found local model at: {model_path}")
            from ottoman_ner import NERPredictor
            
            predictor = NERPredictor(
                model_name_or_path="latin",
                use_local=True
            )
            print("✅ NERPredictor initialization successful with local model")
            
            # Test prediction
            test_text = "Emin Bey'in kuklaları Tepebaşı'nda oynuyor."
            print(f"Testing prediction with: '{test_text}'")
            
            result = predictor.predict(test_text)
            print(f"✅ Prediction successful: {len(result)} entities found")
            
            for entity in result:
                print(f"  - {entity['text']} -> {entity['label']} (confidence: {entity.get('confidence', 0):.3f})")
            
            return True
        else:
            print(f"⚠️  No local model found at {model_path}")
            print("Skipping predictor test (model not available)")
            return True
            
    except Exception as e:
        print(f"❌ Predictor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_import():
    """Test CLI module import."""
    print("\nTesting CLI import...")
    try:
        from ottoman_ner.cli import main, setup_parser
        parser = setup_parser()
        print("✅ CLI import and parser setup successful")
        return True
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    try:
        from ottoman_ner.utils import clean_text, split_sentences, group_entities_by_type
        
        # Test text cleaning
        dirty_text = "  Bu   bir    test   metnidir.  "
        clean = clean_text(dirty_text)
        print(f"✅ Text cleaning: '{dirty_text}' -> '{clean}'")
        
        # Test sentence splitting
        text = "Bu birinci cümle. Bu ikinci cümle. Bu üçüncü cümle."
        sentences = split_sentences(text)
        print(f"✅ Sentence splitting: {len(sentences)} sentences found")
        
        # Test entity grouping
        entities = [
            {"text": "Ahmet", "label": "PER"},
            {"text": "İstanbul", "label": "LOC"},
            {"text": "Mehmet", "label": "PER"}
        ]
        grouped = group_entities_by_type(entities)
        print(f"✅ Entity grouping: {len(grouped)} groups created")
        
        return True
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Ottoman NER Package Test Suite")
    print("=" * 40)
    
    tests = [
        test_package_import,
        test_model_info,
        test_cli_import,
        test_utils,
        test_predictor_with_dummy,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed or were skipped")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
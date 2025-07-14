#!/usr/bin/env python3
"""
Test script for the numerical question functionality.
This script tests the end-to-end pipeline for numerical (GSM8K) questions.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.numerical_panchavakya_generator import generate_numerical_panchavakya_arguments
from src.numerical_evaluator import run_numerical_evaluation, analyze_numerical_results
from src.config import MODELS_TO_TEST

def test_numerical_pipeline():
    """Test the numerical question pipeline with a small sample"""
    print("=== Testing Numerical Question Pipeline ===")
    
    # Step 1: Generate Panchavakya for first 3 numerical questions
    print("\n--- Step 1: Generating Panchavakya for Numerical Questions ---")
    df = generate_numerical_panchavakya_arguments(num_samples=3)
    
    if df is None:
        print("❌ Failed to generate Panchavakya arguments")
        return False
    
    print(f"✅ Successfully generated Panchavakya for {len(df)} questions")
    
    # Step 2: Test evaluation with first model
    test_model = MODELS_TO_TEST[0]  # Use first model for testing
    print(f"\n--- Step 2: Testing Evaluation with {test_model} ---")
    
    try:
        # Test COT evaluation
        print("Testing Chain-of-Thought evaluation...")
        run_numerical_evaluation(test_model, 'cot', num_samples=3)
        
        # Test Zero-shot evaluation
        print("Testing Zero-shot evaluation...")
        run_numerical_evaluation(test_model, 'zero_shot', num_samples=3)
        
        print("✅ Successfully completed evaluations")
        
        # Step 3: Analyze results
        print(f"\n--- Step 3: Analyzing Results ---")
        analyze_numerical_results(test_model, 'cot')
        analyze_numerical_results(test_model, 'zero_shot')
        
        print("✅ Successfully analyzed results")
        return True
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return False

if __name__ == "__main__":
    success = test_numerical_pipeline()
    if success:
        print("\n🎉 Numerical pipeline test completed successfully!")
        print("You can now run the full pipeline with main.py")
    else:
        print("\n❌ Numerical pipeline test failed. Please check the errors above.")

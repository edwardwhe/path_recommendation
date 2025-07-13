#!/usr/bin/env python3
"""
Simple test runner for the Markov-based Educational Path Recommendation System.

This script sets up the environment and runs all tests automatically.
"""

import os
import sys
import subprocess

def main():
    """Main function to run all tests."""
    print("Markov Educational Path Recommendation System - Test Runner")
    print("=" * 60)
    
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Check if sample data exists, if not create it
    print("\nStep 1: Checking/Creating sample data...")
    try:
        import create_sample_data
        if not create_sample_data.check_existing_data():
            print("Creating sample data...")
            create_sample_data.create_all_sample_data()
    except Exception as e:
        print(f"Error creating sample data: {e}")
        return
    
    # Run the function tests
    print("\nStep 2: Running function tests...")
    try:
        import test_functions
        test_runner = test_functions.TestRunner()
        test_runner.run_all_tests()
    except Exception as e:
        print(f"Error running tests: {e}")
        return
    
    print("\nStep 3: Running quick functionality demo...")
    try:
        # Quick demo of the system
        from markov_trainer_offline import MarkovTrainerOffline
        from markov_predictor import MarkovPredictor
        
        print("Testing basic functionality...")
        trainer = MarkovTrainerOffline()
        
        # Quick test
        questions, path = trainer.get_assessment_question_by_unit(3)
        print(f"✅ Found {len(questions)} questions for unit 3")
        
        students = trainer.get_students(questions[:5])  # Use first 5 questions
        print(f"✅ Found {len(students)} students")
        
        if students:
            valid_students = trainer.filter_valid_students(students)
            print(f"✅ Filtered to {len(valid_students)} valid students")
        
        print("✅ Basic functionality test completed!")
        
    except Exception as e:
        print(f"Error in functionality demo: {e}")
    
    print("\n" + "=" * 60)
    print("Test run completed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Demo script for the get_next_question function.

This script demonstrates how to use the get_next_question function to get
personalized next question recommendations for students.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from markov_trainer_offline import MarkovTrainerOffline
from markov_predictor import MarkovPredictor
import json

def demo_get_next_question():
    """
    Demonstrates the get_next_question function with real data.
    """
    print("="*60)
    print("DEMO: get_next_question Function")
    print("="*60)
    
    # Initialize trainer and predictor
    print("1. Initializing trainer and predictor...")
    trainer = MarkovTrainerOffline()
    predictor = MarkovPredictor()
    
    # Train a model for unit 3
    print("2. Training model for unit 3...")
    try:
        transition_matrix, unit_questions, index_to_state, state_to_index = trainer.train_transition_matrix(
            unit=3,
            encoding_type='ordinal',
            cluster_index=None,
            include_exam=False
        )
        print(f"   ✓ Model trained successfully!")
        print(f"   ✓ Transition matrix shape: {transition_matrix.shape}")
        print(f"   ✓ Number of questions: {len(unit_questions)}")
    except Exception as e:
        print(f"   ✗ Training failed: {e}")
        return
    
    # Get some sample students
    print("3. Getting sample students...")
    students = trainer.get_students(unit_questions, include_exam=False)
    valid_students = trainer.filter_valid_students(students)
    print(f"   ✓ Found {len(valid_students)} valid students")
    
    if not valid_students:
        print("   ✗ No valid students found!")
        return
    
    # Demonstrate the get_next_question function
    print("4. Demonstrating get_next_question function...")
    
    # Test with multiple students
    sample_students = valid_students[:5]  # First 5 students
    
    print(f"\n{'Student ID':<15} {'Next Question ID':<40} {'Encoding'}")
    print("-" * 65)
    
    for i, student in enumerate(sample_students):
        student_id = student['user_id']
        
        # Get next question with ordinal encoding
        try:
            next_question_id = predictor.get_next_question(
                student_id=student_id,
                transition_matrix=transition_matrix,
                unit_questions=unit_questions,
                unit=3,
                encoding_type='ordinal',
                index_to_state=index_to_state,
                state_to_index=state_to_index
            )
            print(f"{student_id:<15} {next_question_id:<40} {'ordinal'}")
        except Exception as e:
            print(f"{student_id:<15} {'ERROR: ' + str(e):<40} {'ordinal'}")
    
    # Test with one_hot encoding
    print("\n5. Testing with one_hot encoding...")
    try:
        transition_matrix_oh, unit_questions_oh, index_to_state_oh, state_to_index_oh = trainer.train_transition_matrix(
            unit=3,
            encoding_type='one_hot',
            cluster_index=None,
            include_exam=False
        )
        
        print(f"   ✓ One-hot model trained! Shape: {transition_matrix_oh.shape}")
        
        # Test with first student
        student_id = sample_students[0]['user_id']
        next_question_oh = predictor.get_next_question(
            student_id=student_id,
            transition_matrix=transition_matrix_oh,
            unit_questions=unit_questions_oh,
            unit=3,
            encoding_type='one_hot',
            index_to_state=index_to_state_oh,
            state_to_index=state_to_index_oh
        )
        print(f"   ✓ One-hot prediction for {student_id}: {next_question_oh}")
        
    except Exception as e:
        print(f"   ✗ One-hot encoding failed: {e}")
    
    # Test with new student (no history) - loads from database only
    print("\n6. Testing with new student (no history) - database lookup...")
    try:
        new_student_id = "new_student_12345"
        next_question_new = predictor.get_next_question(
            student_id=new_student_id,
            transition_matrix=transition_matrix,
            unit_questions=unit_questions,
            unit=3,
            encoding_type='ordinal',
            index_to_state=index_to_state,
            state_to_index=state_to_index
        )
        print(f"   ✓ New student prediction: {next_question_new}")
    except Exception as e:
        print(f"   ✗ New student prediction failed: {e}")
        print(f"   Note: This is expected if student doesn't exist in database")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED!")
    print("="*60)
    
    # Show function signature
    print("\nFunction Signature:")
    print("get_next_question(student_id, transition_matrix, unit_questions, unit, encoding_type, index_to_state, state_to_index)")
    print("\nParameters:")
    print("- student_id: String ID of the student")
    print("- transition_matrix: Trained Markov transition matrix")
    print("- unit_questions: List of questions for the unit")
    print("- unit: Unit number")
    print("- encoding_type: 'ordinal' or 'one_hot'")
    print("- index_to_state: State mapping (can be None for ordinal)")
    print("- state_to_index: State mapping (can be None for ordinal)")
    print("\nReturns:")
    print("- question_id: String ID of the next recommended question")

if __name__ == "__main__":
    demo_get_next_question() 
#!/usr/bin/env python3
"""
Example usage script for the Markov-based Educational Path Recommendation System.

This script demonstrates various ways to use the system for training models
and making predictions.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import PathRecommendationSystem
from markov_trainer_offline import MarkovTrainerOffline
from markov_predictor import MarkovPredictor

def example_basic_usage():
    """
    Example 1: Basic training and prediction workflow
    """
    print("=== Example 1: Basic Usage ===")
    
    try:
        # Initialize the system
        system = PathRecommendationSystem()
        
        # Train a model for unit 3 using ordinal encoding
        print("Training model...")
        model_info = system.train_model(
            unit=3,
            encoding_type='ordinal',
            cluster_index=None,  # Use all clusters
            include_exam=False   # Exclude exam questions
        )
        print(f"Training completed: {model_info}")
        
        # Load the model for prediction
        print("\nLoading model...")
        system.load_model(unit=3, encoding_type='ordinal')
        
        # Make predictions with a sample sequence
        sample_sequence = [0, 2, 5]  # Student has attempted questions 0, 2, 5
        print(f"\nSample student sequence: {sample_sequence}")
        
        # Get single prediction
        next_question = system.predict_next_question(sample_sequence)
        print(f"Next recommended question: {next_question}")
        
        # Get multiple predictions
        top_predictions = system.get_multiple_predictions(sample_sequence, n_predictions=3)
        print(f"Top 3 recommendations: {top_predictions}")
        
        # Analyze the sequence
        analysis = system.analyze_student_sequence(sample_sequence)
        print(f"Sequence analysis: {analysis}")
        
    except Exception as e:
        print(f"Error in basic usage example: {e}")

def example_cluster_based_training():
    """
    Example 2: Training models for specific student clusters
    """
    print("\n=== Example 2: Cluster-based Training ===")
    
    try:
        system = PathRecommendationSystem()
        
        # Train models for each cluster
        for cluster_id in [0, 1, 2]:
            print(f"\nTraining model for cluster {cluster_id}...")
            model_info = system.train_model(
                unit=3,
                encoding_type='ordinal',
                cluster_index=cluster_id,
                include_exam=False
            )
            print(f"Cluster {cluster_id} model: {model_info['transition_matrix_shape']}")
        
        # Load and test cluster 0 model
        print("\nTesting cluster 0 model...")
        system.load_model(unit=3, encoding_type='ordinal', cluster_index=0)
        
        # Test with same sequence as before
        sample_sequence = [0, 2, 5]
        next_question = system.predict_next_question(sample_sequence)
        print(f"Cluster 0 recommendation: {next_question}")
        
    except Exception as e:
        print(f"Error in cluster-based training example: {e}")

def example_one_hot_encoding():
    """
    Example 3: Using one-hot encoding
    """
    print("\n=== Example 3: One-Hot Encoding ===")
    
    try:
        system = PathRecommendationSystem()
        
        # Train model with one-hot encoding
        print("Training one-hot model...")
        model_info = system.train_model(
            unit=3,
            encoding_type='one_hot',
            cluster_index=None,
            include_exam=False
        )
        print(f"One-hot model trained: {model_info}")
        
        # Load the one-hot model
        system.load_model(unit=3, encoding_type='one_hot')
        
        # Create a one-hot sequence (simplified example with 10 questions)
        num_questions = min(10, model_info['num_questions'])
        one_hot_sequence = [
            [1 if i == 0 else 0 for i in range(num_questions)],  # Only question 0
            [1 if i in [0, 2] else 0 for i in range(num_questions)],  # Questions 0, 2
            [1 if i in [0, 2, 5] else 0 for i in range(num_questions)]  # Questions 0, 2, 5
        ]
        
        print(f"One-hot sequence: {one_hot_sequence}")
        
        # Make prediction
        next_question = system.predict_next_question(one_hot_sequence)
        print(f"One-hot prediction: {next_question}")
        
    except Exception as e:
        print(f"Error in one-hot encoding example: {e}")

def example_advanced_prediction_methods():
    """
    Example 4: Different prediction methods
    """
    print("\n=== Example 4: Advanced Prediction Methods ===")
    
    try:
        system = PathRecommendationSystem()
        
        # Load an existing model (assumes one was trained previously)
        try:
            system.load_model(unit=3, encoding_type='ordinal')
        except:
            # If no model exists, train one quickly
            print("No existing model found, training a new one...")
            system.train_model(unit=3, encoding_type='ordinal', cluster_index=None)
            system.load_model(unit=3, encoding_type='ordinal')
        
        sample_sequence = [0, 2, 5]
        
        # Method 1: Maximum probability (deterministic)
        print("Method 1: Maximum probability")
        next_q_max = system.predict_next_question(
            sample_sequence, 
            method='max_probability'
        )
        print(f"Max probability prediction: {next_q_max}")
        
        # Method 2: Weighted random (stochastic)
        print("\nMethod 2: Weighted random (run multiple times)")
        random_predictions = []
        for i in range(5):
            next_q_random = system.predict_next_question(
                sample_sequence, 
                method='weighted_random'
            )
            random_predictions.append(next_q_random)
        print(f"Random predictions: {random_predictions}")
        
        # Get multiple predictions with probabilities
        print("\nMultiple predictions with probabilities:")
        multiple_preds = system.get_multiple_predictions(
            sample_sequence, 
            n_predictions=5
        )
        for i, (question_id, prob) in enumerate(multiple_preds):
            print(f"  {i+1}. {question_id} (probability: {prob:.4f})")
        
    except Exception as e:
        print(f"Error in advanced prediction methods example: {e}")

def example_direct_api_usage():
    """
    Example 5: Using the trainer and predictor classes directly
    """
    print("\n=== Example 5: Direct API Usage ===")
    
    try:
        # Initialize trainer with custom configuration (if needed)
        trainer = MarkovTrainer()
        
        # Get questions for unit 3
        questions, path = trainer.get_assessment_question_by_unit(3)
        print(f"Found {len(questions)} questions for unit 3")
        
        # Get students data
        students = trainer.get_students(questions, include_exam=False)
        print(f"Found {len(students)} students")
        
        # Filter valid students
        valid_students = trainer.filter_valid_students(students)
        print(f"Filtered to {len(valid_students)} valid students")
        
        # Train the model directly
        transition_matrix, unit_questions, index_to_state, state_to_index = trainer.train_transition_matrix(
            unit=3,
            encoding_type='ordinal',
            cluster_index=None,
            include_exam=False
        )
        
        print(f"Transition matrix shape: {transition_matrix.shape}")
        print(f"Number of unit questions: {len(unit_questions)}")
        
        # Save the model
        trainer.save_model(
            transition_matrix,
            unit_questions,
            index_to_state,
            state_to_index,
            unit=3,
            encoding_type='ordinal'
        )
        
        # Use predictor directly
        predictor = MarkovPredictor()
        predictor.load_model(unit=3, encoding_type='ordinal')
        
        # Make predictions
        sample_sequence = [0, 1, 3]
        prediction = predictor.predict_next_question(sample_sequence)
        question_id = predictor.get_question_id(prediction)
        
        print(f"Direct prediction: index {prediction} -> question {question_id}")
        
    except Exception as e:
        print(f"Error in direct API usage example: {e}")

def example_error_handling():
    """
    Example 6: Error handling and edge cases
    """
    print("\n=== Example 6: Error Handling ===")
    
    system = PathRecommendationSystem()
    
    # Try to predict without loading a model
    print("Testing prediction without model...")
    try:
        system.predict_next_question([0, 1, 2])
    except ValueError as e:
        print(f"Expected error: {e}")
    
    # Try to load non-existent model
    print("\nTesting loading non-existent model...")
    try:
        system.load_model(unit=999, encoding_type='ordinal')
    except FileNotFoundError as e:
        print(f"Expected error: {e}")
    
    # Try prediction with empty sequence
    print("\nTesting prediction with empty sequence...")
    try:
        # First train and load a model
        system.train_model(unit=3, encoding_type='ordinal')
        system.load_model(unit=3, encoding_type='ordinal')
        
        # Then try empty sequence
        prediction = system.predict_next_question([])
        print(f"Empty sequence prediction: {prediction}")
        
    except Exception as e:
        print(f"Error with empty sequence: {e}")

def example_get_next_question():
    """
    Example 7: Using get_next_question function
    
    This example demonstrates how to use the get_next_question function
    to get the next recommended question for a specific student.
    """
    print("="*60)
    print("EXAMPLE 7: Using get_next_question function")
    print("="*60)
    
    try:
        # Initialize trainer and predictor
        trainer = MarkovTrainerOffline()
        predictor = MarkovPredictor()
        
        # Train a model for unit 3
        print("Training model for unit 3...")
        transition_matrix, unit_questions, index_to_state, state_to_index = trainer.train_transition_matrix(
            unit=3,
            encoding_type='ordinal',
            cluster_index=None,
            include_exam=False
        )
        
        print(f"Model trained successfully!")
        print(f"Transition matrix shape: {transition_matrix.shape}")
        print(f"Number of questions: {len(unit_questions)}")
        
        # Get a sample student
        students = trainer.get_students(unit_questions, include_exam=False)
        valid_students = trainer.filter_valid_students(students)
        
        if valid_students:
            # Use a real student
            student_id = valid_students[0]['user_id']
            print(f"\nGetting next question for student: {student_id}")
        else:
            # Use a dummy student ID
            student_id = "sample_student_123"
            print(f"\nGetting next question for sample student: {student_id}")
        
        # Get next question recommendation
        next_question_id = predictor.get_next_question(
            student_id=student_id,
            transition_matrix=transition_matrix,
            unit_questions=unit_questions,
            unit=3,
            encoding_type='ordinal',
            index_to_state=index_to_state,
            state_to_index=state_to_index
        )
        
        print(f"Recommended next question: {next_question_id}")
        
        # Test with one_hot encoding
        print("\nTesting with one_hot encoding...")
        
        # Train with one_hot encoding
        transition_matrix_oh, unit_questions_oh, index_to_state_oh, state_to_index_oh = trainer.train_transition_matrix(
            unit=3,
            encoding_type='one_hot',
            cluster_index=None,
            include_exam=False
        )
        
        next_question_id_oh = predictor.get_next_question(
            student_id=student_id,
            transition_matrix=transition_matrix_oh,
            unit_questions=unit_questions_oh,
            unit=3,
            encoding_type='one_hot',
            index_to_state=index_to_state_oh,
            state_to_index=state_to_index_oh
        )
        
        print(f"Recommended next question (one_hot): {next_question_id_oh}")
        
        # Test with different students
        print("\nTesting with multiple students...")
        for i, student in enumerate(valid_students[:3]):  # Test first 3 students
            student_id = student['user_id']
            next_q = predictor.get_next_question(
                student_id=student_id,
                transition_matrix=transition_matrix,
                unit_questions=unit_questions,
                unit=3,
                encoding_type='ordinal',
                index_to_state=index_to_state,
                state_to_index=state_to_index
            )
            print(f"Student {i+1} ({student_id}): {next_q}")
        
    except Exception as e:
        print(f"Error in get_next_question example: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Run all examples
    """
    print("Markov-based Educational Path Recommendation System - Examples")
    print("=" * 65)
    
    # Check if we can connect to database
    try:
        trainer = MarkovTrainer()
        print("Database connection successful!")
    except Exception as e:
        print(f"Database connection failed: {e}")
        print("Please check your MongoDB configuration and try again.")
        return
    
    # Run examples
    examples = [
        example_basic_usage,
        example_cluster_based_training,
        example_one_hot_encoding,
        example_advanced_prediction_methods,
        example_direct_api_usage,
        example_error_handling,
        example_get_next_question
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Example failed: {e}")
        print()  # Add spacing between examples

if __name__ == "__main__":
    main() 
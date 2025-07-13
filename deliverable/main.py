#!/usr/bin/env python3
"""
Main interface for the Markov-based Educational Path Recommendation System.

This file provides a unified interface for both training Markov models and making predictions.
It demonstrates how to use the MarkovTrainer and MarkovPredictor classes.
"""

import argparse
import sys
import os
from typing import List, Union, Optional
import json

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from markov_trainer import MarkovTrainer
from markov_predictor import MarkovPredictor

class PathRecommendationSystem:
    """
    A complete path recommendation system that handles both training and prediction.
    """
    
    def __init__(self, db_config: Optional[dict] = None):
        """
        Initialize the path recommendation system.
        
        Args:
            db_config: Database configuration dictionary. If None, uses default config.
        """
        self.trainer = MarkovTrainer(db_config)
        self.predictor = MarkovPredictor()
        self.models = {}  # Cache for loaded models
    
    def train_model(self, unit: int, encoding_type: str = 'ordinal', 
                   cluster_index: Optional[int] = None, include_exam: bool = False,
                   save_dir: str = '.') -> dict:
        """
        Train a Markov model for the specified unit.
        
        Args:
            unit: The unit number to train for.
            encoding_type: 'ordinal' or 'one_hot'.
            cluster_index: Specific cluster to use (0, 1, 2) or None for all clusters.
            include_exam: Whether to include exam questions in training.
            save_dir: Directory to save the trained model.
            
        Returns:
            Dictionary containing training results and model information.
        """
        print(f"Starting training for unit {unit}...")
        
        try:
            # Train the model
            transition_matrix, unit_questions, index_to_state, state_to_index = self.trainer.train_transition_matrix(
                unit=unit,
                encoding_type=encoding_type,
                cluster_index=cluster_index,
                include_exam=include_exam
            )
            
            # Save the model
            self.trainer.save_model(
                transition_matrix=transition_matrix,
                unit_questions=unit_questions,
                index_to_state=index_to_state,
                state_to_index=state_to_index,
                unit=unit,
                encoding_type=encoding_type,
                cluster_index=cluster_index
            )
            
            # Store model info
            model_key = f"unit_{unit}_{encoding_type}"
            if cluster_index is not None:
                model_key += f"_cluster_{cluster_index}"
            
            model_info = {
                'unit': unit,
                'encoding_type': encoding_type,
                'cluster_index': cluster_index,
                'include_exam': include_exam,
                'transition_matrix_shape': transition_matrix.shape,
                'num_questions': len(unit_questions),
                'model_key': model_key
            }
            
            self.models[model_key] = model_info
            
            print(f"Training completed successfully!")
            print(f"Model key: {model_key}")
            print(f"Transition matrix shape: {transition_matrix.shape}")
            print(f"Number of questions: {len(unit_questions)}")
            
            return model_info
            
        except Exception as e:
            print(f"Training failed: {e}")
            raise
    
    def load_model(self, unit: int, encoding_type: str = 'ordinal', 
                   cluster_index: Optional[int] = None, model_dir: str = '.') -> str:
        """
        Load a trained model for prediction.
        
        Args:
            unit: The unit number.
            encoding_type: 'ordinal' or 'one_hot'.
            cluster_index: Cluster index if model was trained on specific cluster.
            model_dir: Directory containing the model files.
            
        Returns:
            Model key for the loaded model.
        """
        model_key = f"unit_{unit}_{encoding_type}"
        if cluster_index is not None:
            model_key += f"_cluster_{cluster_index}"
        
        print(f"Loading model: {model_key}")
        
        try:
            self.predictor.load_model(
                unit=unit,
                encoding_type=encoding_type,
                cluster_index=cluster_index,
                model_dir=model_dir
            )
            
            # Cache model info
            self.models[model_key] = {
                'unit': unit,
                'encoding_type': encoding_type,
                'cluster_index': cluster_index,
                'loaded': True
            }
            
            print(f"Model loaded successfully: {model_key}")
            return model_key
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def predict_next_question(self, current_sequence: List[Union[int, List[int]]], 
                             method: str = 'max_probability', 
                             return_question_id: bool = True) -> Union[int, str, tuple]:
        """
        Predict the next question for a student.
        
        Args:
            current_sequence: Current student's question sequence.
            method: Prediction method ('max_probability', 'weighted_random').
            return_question_id: If True, return question ID; if False, return index.
            
        Returns:
            Next question index or ID, depending on return_question_id parameter.
        """
        if self.predictor.transition_matrix is None:
            raise ValueError("No model loaded. Load a model first using load_model().")
        
        try:
            question_index = self.predictor.predict_next_question(current_sequence, method)
            
            if return_question_id:
                question_id = self.predictor.get_question_id(question_index)
                return question_id
            else:
                return question_index
                
        except Exception as e:
            print(f"Prediction failed: {e}")
            raise
    
    def get_multiple_predictions(self, current_sequence: List[Union[int, List[int]]], 
                               n_predictions: int = 5, 
                               return_question_ids: bool = True) -> List[tuple]:
        """
        Get multiple next question predictions.
        
        Args:
            current_sequence: Current student's question sequence.
            n_predictions: Number of predictions to return.
            return_question_ids: If True, return question IDs; if False, return indices.
            
        Returns:
            List of (question_id/index, probability) tuples.
        """
        if self.predictor.transition_matrix is None:
            raise ValueError("No model loaded. Load a model first using load_model().")
        
        try:
            predictions = self.predictor.get_multiple_predictions(current_sequence, n_predictions)
            
            if return_question_ids:
                result = []
                for question_index, probability in predictions:
                    try:
                        question_id = self.predictor.get_question_id(question_index)
                        result.append((question_id, probability))
                    except ValueError:
                        # Skip invalid indices
                        continue
                return result
            else:
                return predictions
                
        except Exception as e:
            print(f"Multiple predictions failed: {e}")
            raise
    
    def analyze_student_sequence(self, sequence: List[Union[int, List[int]]]) -> dict:
        """
        Analyze a student's learning sequence.
        
        Args:
            sequence: Student's question sequence.
            
        Returns:
            Dictionary containing analysis results.
        """
        if self.predictor.transition_matrix is None:
            raise ValueError("No model loaded. Load a model first using load_model().")
        
        try:
            return self.predictor.analyze_sequence(sequence)
        except Exception as e:
            print(f"Sequence analysis failed: {e}")
            raise
    
    def demo_workflow(self, unit: int = 3, encoding_type: str = 'ordinal'):
        """
        Demonstrate a complete workflow from training to prediction.
        
        Args:
            unit: Unit number to demonstrate with.
            encoding_type: Encoding type to use.
        """
        print(f"=== Demo Workflow for Unit {unit} with {encoding_type} encoding ===")
        
        try:
            # Step 1: Train a model
            print("\n1. Training model...")
            model_info = self.train_model(
                unit=unit,
                encoding_type=encoding_type,
                cluster_index=None,  # Use all clusters
                include_exam=False
            )
            
            # Step 2: Load the model for prediction
            print("\n2. Loading model for prediction...")
            model_key = self.load_model(
                unit=unit,
                encoding_type=encoding_type,
                cluster_index=None
            )
            
            # Step 3: Make some sample predictions
            print("\n3. Making sample predictions...")
            
            if encoding_type == 'ordinal':
                # Sample ordinal sequence
                sample_sequence = [0, 2, 5]
                print(f"Sample sequence (ordinal): {sample_sequence}")
            else:  # one_hot
                # Sample one-hot sequence (simplified for demo)
                num_questions = model_info['num_questions']
                sample_sequence = [
                    [1 if i == 0 else 0 for i in range(min(10, num_questions))],
                    [1 if i in [0, 2] else 0 for i in range(min(10, num_questions))]
                ]
                print(f"Sample sequence (one-hot): {sample_sequence}")
            
            # Single prediction
            next_question = self.predict_next_question(sample_sequence)
            print(f"Next question prediction: {next_question}")
            
            # Multiple predictions
            multiple_predictions = self.get_multiple_predictions(sample_sequence, n_predictions=3)
            print(f"Top 3 predictions: {multiple_predictions}")
            
            # Sequence analysis
            analysis = self.analyze_student_sequence(sample_sequence)
            print(f"Sequence analysis: {analysis}")
            
            print("\n=== Demo completed successfully! ===")
            
        except Exception as e:
            print(f"Demo failed: {e}")
            raise


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Markov-based Educational Path Recommendation System')
    parser.add_argument('action', choices=['train', 'predict', 'demo'], 
                       help='Action to perform: train a model, make predictions, or run demo')
    parser.add_argument('--unit', type=int, default=3, help='Unit number (default: 3)')
    parser.add_argument('--encoding', choices=['ordinal', 'one_hot'], default='ordinal',
                       help='Encoding type (default: ordinal)')
    parser.add_argument('--cluster', type=int, default=None, 
                       help='Cluster index (0, 1, 2) or None for all clusters')
    parser.add_argument('--include-exam', action='store_true', 
                       help='Include exam questions in training')
    parser.add_argument('--sequence', type=str, default=None,
                       help='JSON string of current sequence for prediction')
    parser.add_argument('--model-dir', type=str, default='.',
                       help='Directory containing model files (default: current directory)')
    
    args = parser.parse_args()
    
    # Initialize the system
    system = PathRecommendationSystem()
    
    try:
        if args.action == 'train':
            print(f"Training model for unit {args.unit}...")
            model_info = system.train_model(
                unit=args.unit,
                encoding_type=args.encoding,
                cluster_index=args.cluster,
                include_exam=args.include_exam
            )
            print(f"Training completed. Model info: {model_info}")
            
        elif args.action == 'predict':
            if args.sequence is None:
                print("Error: --sequence is required for prediction")
                sys.exit(1)
            
            # Load model
            model_key = system.load_model(
                unit=args.unit,
                encoding_type=args.encoding,
                cluster_index=args.cluster,
                model_dir=args.model_dir
            )
            
            # Parse sequence
            try:
                sequence = json.loads(args.sequence)
            except json.JSONDecodeError:
                print("Error: Invalid JSON format for sequence")
                sys.exit(1)
            
            # Make prediction
            next_question = system.predict_next_question(sequence)
            multiple_predictions = system.get_multiple_predictions(sequence, n_predictions=5)
            analysis = system.analyze_student_sequence(sequence)
            
            print(f"Current sequence: {sequence}")
            print(f"Next question: {next_question}")
            print(f"Top 5 predictions: {multiple_predictions}")
            print(f"Sequence analysis: {analysis}")
            
        elif args.action == 'demo':
            system.demo_workflow(unit=args.unit, encoding_type=args.encoding)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
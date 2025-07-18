import json
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any, Union
import os
from pymongo import MongoClient
from datetime import datetime
from question_cluster import calculate_similarity

class MarkovPredictor:
    """
    A Markov model predictor for educational path recommendation.
    """
    
    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize the Markov predictor.
        
        Args:
            db_config: Database configuration dictionary. If None, uses default config.
        """
        self.transition_matrix = None
        self.unit_questions = None
        self.index_to_state = None
        self.state_to_index = None
        self.encoding_type = None
        
        # Database configuration
        self.db_config = db_config or {
            'password': 'hf6Wbg3dm8',
            'url': 'mongodb://root:hf6Wbg3dm8@dds-3ns35de5eee23e941756-pub.mongodb.rds.aliyuncs.com:3717,dds-3ns35de5eee23e942366-pub.mongodb.rds.aliyuncs.com:3717/admin?replicaSet=mgset-33008719'
        }
        self.client = None
        self.db = None
        
        # Time filtering thresholds (in milliseconds)
        self.min_duration = 2000  # 2 seconds
        self.max_duration = 93500  # 93.5 seconds
        
    def _connect_to_database(self):
        """Establish connection to MongoDB database."""
        if self.client is None:
            try:
                self.client = MongoClient(self.db_config['url'])
                self.db = self.client['pro_mathaday_assessment']
                print("Connected to MongoDB successfully")
            except Exception as e:
                print(f"Failed to connect to database: {e}")
                raise
    
    def get_next_question(self, student_id: str, transition_matrix: np.ndarray, 
                         unit_questions: List[str], unit: int, encoding_type: str,
                         index_to_state: Optional[Any] = None, 
                         state_to_index: Optional[Any] = None) -> str:
        """
        Get the next question for a student based on their latest question in the unit.
        Loads student data from database only (online mode).
        
        Args:
            student_id: String ID of the student.
            transition_matrix: Markov transition matrix.
            unit_questions: List of questions used to encode the state order by ID.
            unit: The unit number for prediction.
            encoding_type: 'ordinal' or 'one_hot'.
            index_to_state: Mapping from index to state (for one_hot encoding).
            state_to_index: Mapping from state to index (for one_hot encoding).
            
        Returns:
            question_id: The ID of the next question.
        """
        # Set the model parameters
        self.transition_matrix = transition_matrix
        self.unit_questions = unit_questions
        self.index_to_state = index_to_state
        self.state_to_index = state_to_index
        self.encoding_type = encoding_type
        
        # Get student's current sequence in the unit
        student_sequence = self._get_student_sequence(student_id, unit, unit_questions)
        
        if not student_sequence:
            # If no sequence found, return first question or random question
            if unit_questions:
                return unit_questions[0]
            else:
                raise ValueError(f"No questions available for unit {unit}")
        
        # Predict next question index
        next_question_index = self.predict_next_question(student_sequence, method='max_probability')
        
        # Convert index to question ID
        if 0 <= next_question_index < len(unit_questions):
            return unit_questions[next_question_index]
        else:
            # Fallback: return random question from unit
            return random.choice(unit_questions)
    
    
    def get_next_question_by_similarity(self, student_id: str, transition_matrix: np.ndarray, 
                         unit_questions: List[str], unit: int, encoding_type: str,
                         index_to_state: Optional[Any] = None, 
                         state_to_index: Optional[Any] = None) -> str:
        
        """
        Get the next question for a student based on similarity of questions.
        Loads student data from database only (online mode).
        
        Args:
            student_id: String ID of the student.
            transition_matrix: Markov transition matrix.
            unit_questions: List of questions used to encode the state order by ID.
            unit: The unit number for prediction.
            encoding_type: 'ordinal' or 'one_hot'.
            index_to_state: Mapping from index to state (for one_hot encoding).
            state_to_index: Mapping from state to index (for one_hot encoding).
            
        Returns:
            question_id: The ID of the next question.
        """
        
        # Set the model parameters
        self.transition_matrix = transition_matrix
        self.unit_questions = unit_questions
        self.index_to_state = index_to_state
        self.state_to_index = state_to_index
        self.encoding_type = encoding_type
        
        # Get student's current sequence in the unit
        student_sequence = self._get_student_sequence(student_id, unit, unit_questions)

        if not student_sequence:
            # If no sequence found, return first question or random question
            if unit_questions:
                return unit_questions[0]
            else:
                raise ValueError(f"No questions available for unit {unit}")
            
        # predict next question by similarity
        done_indices = set(student_sequence)
        all_indices = set(range(len(unit_questions)))
        undone_indices = list(all_indices-done_indices)
        last_done = student_sequence[-1]
        if not undone_indices:
            return '-1'

        best_idx = calculate_similarity(last_done, undone_indices,unit_questions)
        return unit_questions[best_idx]

    def _get_student_sequence(self, student_id: str, unit: int, unit_questions: List[str]) -> List[Union[int, List[int]]]:
        """
        Get the student's sequence of questions in the specified unit.
        Loads student data from database only (online mode).
        
        Args:
            student_id: Student ID.
            unit: Unit number.
            unit_questions: List of questions in the unit.
            
        Returns:
            Student's sequence in the appropriate encoding format.
        """
        try:
            # Load student data from database (online only)
            student_data = self._load_student_data_database(student_id, unit, unit_questions)
        except Exception as e:
            print(f"Failed to load student data from database: {e}")
            return []
        
        if not student_data or not student_data.get('assessments'):
            return []
        
        # Filter valid assessments
        valid_assessments = []
        for assessment in student_data['assessments']:
            duration = assessment.get('duration')
            if (duration is not None and 
                self.min_duration <= duration <= self.max_duration and
                assessment.get('score') is not None):
                valid_assessments.append(assessment)
        
        if not valid_assessments:
            return []
        
        # Sort by start time
        valid_assessments.sort(key=lambda x: x.get('start_time', 0))
        
        # Encode sequence based on encoding type
        if self.encoding_type == 'ordinal':
            sequence = []
            for assessment in valid_assessments:
                if assessment['question'] in unit_questions:
                    question_index = unit_questions.index(assessment['question'])
                    sequence.append(question_index)
            return sequence
        
        elif self.encoding_type == 'one_hot':
            sequence = []
            tracking_vector = [0] * len(unit_questions)
            
            for assessment in valid_assessments:
                if assessment['question'] in unit_questions:
                    question_index = unit_questions.index(assessment['question'])
                    tracking_vector[question_index] = 1
                    sequence.append(tracking_vector.copy())
            
            return sequence
        
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
    
    def _load_student_data_offline(self, student_id: str, unit: int, unit_questions: List[str]) -> Dict:
        """
        Load student data from offline JSON files.
        
        Args:
            student_id: Student ID.
            unit: Unit number.
            unit_questions: List of questions in the unit.
            
        Returns:
            Student data dictionary.
        """
        # Try to load from users.json
        users_file = "./data/users.json"
        if not os.path.exists(users_file):
            raise FileNotFoundError(f"Users file not found: {users_file}")
        
        with open(users_file, 'r') as f:
            users = json.load(f)
        
        # Find the student
        student_data = None
        for user in users:
            if user.get('user_id') == student_id:
                student_data = user
                break
        
        if not student_data:
            raise ValueError(f"Student {student_id} not found in offline data")
        
        # Filter assessments to only include those in the target unit questions
        filtered_assessments = []
        for assessment in student_data.get('assessments', []):
            if assessment.get('question') in unit_questions:
                filtered_assessments.append(assessment)
        
        return {
            'user_id': student_id,
            'assessments': filtered_assessments
        }
    
    def _load_student_data_database(self, student_id: str, unit: int, unit_questions: List[str]) -> Dict:
        """
        Load student data from MongoDB database.
        
        Args:
            student_id: Student ID.
            unit: Unit number.
            unit_questions: List of questions in the unit.
            
        Returns:
            Student data dictionary.
        """
        self._connect_to_database()
        
        # Get assessment info for filtering
        assessments_collection = self.db['assessments']
        assessment_info = {}
        
        for assessment in assessments_collection.find():
            assessment_info[assessment["id"]] = {
                "type": assessment["type"],
                "syllabi": [syllable.get("unitId") for syllable in assessment.get("syllabi", [])]
            }
        
        # Get student's question records
        question_records_collection = self.db['assessment_question_records']
        user_data = question_records_collection.find({"userId": student_id})
        
        assessments = []
        for record in user_data:
            # Skip if question not in unit questions
            if record.get("assessmentQuestionId") not in unit_questions:
                continue
            
            # Get assessment info
            assessment = assessment_info.get(record["assessmentId"])
            if not assessment:
                continue
            
            # Exclude exam types (5 and 7)
            if assessment.get("type") in [5, 7]:
                continue
            
            assessment_data = {
                "start_time": record["createdAt"],
                "duration": record.get("totalTimeSpent"),
                "score": record.get("score"),
                "question": record.get("assessmentQuestionId"),
                "assessment_id": record.get("assessmentId"),
                "assessment_type": assessment.get("type"),
                "unit": assessment.get("syllabi")
            }
            assessments.append(assessment_data)
        
        return {
            'user_id': student_id,
            'assessments': assessments
        }

    def load_model(self, unit: int, encoding_type: str = 'ordinal', 
                   cluster_index: Optional[int] = None, model_dir: str = '.'):
        """
        Load a trained Markov model.
        
        Args:
            unit: The unit number.
            encoding_type: 'ordinal' or 'one_hot'.
            cluster_index: Cluster index if model was trained on specific cluster.
            model_dir: Directory containing the model files.
        """
        # Create filename suffix
        suffix = f"unit_{unit}_{encoding_type}"
        if cluster_index is not None:
            suffix += f"_cluster_{cluster_index}"
        
        # Load transition matrix
        transition_matrix_path = os.path.join(model_dir, f'transition_matrix_{suffix}.npy')
        if not os.path.exists(transition_matrix_path):
            raise FileNotFoundError(f"Transition matrix not found: {transition_matrix_path}")
        
        self.transition_matrix = np.load(transition_matrix_path)
        
        # Load questions
        questions_path = os.path.join(model_dir, f'unit_questions_{suffix}.json')
        if not os.path.exists(questions_path):
            raise FileNotFoundError(f"Unit questions not found: {questions_path}")
        
        with open(questions_path, 'r') as f:
            self.unit_questions = json.load(f)
        
        # Load state mappings for one-hot encoding
        if encoding_type == 'one_hot':
            index_to_state_path = os.path.join(model_dir, f'index_to_state_{suffix}.npy')
            state_to_index_path = os.path.join(model_dir, f'state_to_index_{suffix}.json')
            
            if os.path.exists(index_to_state_path):
                self.index_to_state = np.load(index_to_state_path, allow_pickle=True).tolist()
            
            if os.path.exists(state_to_index_path):
                with open(state_to_index_path, 'r') as f:
                    json_data = json.load(f)
                    # Convert string keys back to tuples
                    self.state_to_index = {eval(k): v for k, v in json_data.items()}
        
        self.encoding_type = encoding_type
        print(f"Model loaded successfully: {suffix}")
        print(f"Transition matrix shape: {self.transition_matrix.shape}")
        print(f"Number of questions: {len(self.unit_questions)}")
    
    def predict_next_question(self, current_sequence: List[Union[int, List[int]]], 
                             method: str = 'max_probability') -> Union[int, str]:
        """
        Predict the next question based on the current student sequence.
        
        Args:
            current_sequence: Current student's question sequence.
                            For ordinal: List of question indices.
                            For one_hot: List of state vectors.
            method: Prediction method ('max_probability', 'weighted_random').
            
        Returns:
            predicted_question_index: Index of the predicted next question.
        """
        if self.transition_matrix is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        if self.encoding_type == 'ordinal':
            return self._predict_ordinal(current_sequence, method)
        elif self.encoding_type == 'one_hot':
            return self._predict_one_hot(current_sequence, method)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
    
    def _predict_ordinal(self, current_sequence: List[int], method: str) -> int:
        """
        Predict next question for ordinal encoding.
        
        Args:
            current_sequence: List of question indices.
            method: Prediction method.
            
        Returns:
            Index of predicted next question.
        """
        if not current_sequence:
            # If no sequence, return random question or use initial distribution
            if method == 'max_probability':
                # Use the state with highest total outgoing probability
                total_probs = np.sum(self.transition_matrix, axis=1)
                return int(np.argmax(total_probs))
            else:  # weighted_random
                return random.randint(0, len(self.transition_matrix) - 1)
        
        current_state = current_sequence[-1]  # Last question in sequence
        
        if current_state >= len(self.transition_matrix):
            raise ValueError(f"Current state {current_state} is out of bounds for transition matrix")
        
        # Get transition probabilities from current state
        transition_probs = self.transition_matrix[current_state]
        
        if method == 'max_probability':
            # Return the most likely next state
            predicted_next_state = int(np.argmax(transition_probs))
        elif method == 'weighted_random':
            # Sample from the probability distribution
            if np.sum(transition_probs) > 0:
                # Normalize probabilities
                probs = transition_probs / np.sum(transition_probs)
                predicted_next_state = np.random.choice(len(probs), p=probs)
            else:
                # If no valid transitions, return random
                predicted_next_state = random.randint(0, len(self.transition_matrix) - 1)
        else:
            raise ValueError(f"Unknown prediction method: {method}")
        
        return predicted_next_state
    
    def _predict_one_hot(self, current_sequence: List[List[int]], method: str) -> int:
        """
        Predict next question for one-hot encoding.
        
        Args:
            current_sequence: List of state vectors.
            method: Prediction method.
            
        Returns:
            Index of predicted next question.
        """
        if not current_sequence:
            # If no sequence, return random question
            return random.randint(0, len(self.unit_questions) - 1)
        
        current_state_vector = current_sequence[-1]  # Last state vector
        current_state_tuple = tuple(current_state_vector)
        
        # Find current state in the model
        if current_state_tuple not in self.state_to_index:
            print(f"Current state {current_state_tuple} not found in state_to_index")
            # If state not found, find the closest state by similarity
            max_similarity = -1
            closest_states = []
            
            for state_tuple, state_index in self.state_to_index.items():
                state_vector = np.array(state_tuple)
                current_vector = np.array(current_state_vector)
                similarity = np.dot(state_vector, current_vector)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    closest_states = [state_index]
                elif similarity == max_similarity:
                    closest_states.append(state_index)
            
            # Randomly select among the closest states
            if closest_states:
                current_state_index = random.choice(closest_states)
            else:
                # If no similar states found, return random
                return random.randint(0, len(self.unit_questions) - 1)
        else:
            current_state_index = self.state_to_index[current_state_tuple]
        
        if current_state_index >= len(self.transition_matrix):
            raise ValueError(f"Current state index {current_state_index} is out of bounds")
        
        # Get transition probabilities from current state
        transition_probs = self.transition_matrix[current_state_index]
        
        # Select next state based on method
        if method == 'max_probability':
            predicted_next_state_index = int(np.argmax(transition_probs))
        elif method == 'weighted_random':
            if np.sum(transition_probs) > 0:
                probs = transition_probs / np.sum(transition_probs)
                predicted_next_state_index = np.random.choice(len(probs), p=probs)
            else:
                return random.randint(0, len(self.unit_questions) - 1)
        else:
            raise ValueError(f"Unknown prediction method: {method}")
        
        if predicted_next_state_index >= len(self.index_to_state):
            raise ValueError(f"Predicted state index {predicted_next_state_index} is out of bounds")
        
        predicted_next_state_vector = self.index_to_state[predicted_next_state_index]
        
        # Find questions that were newly attempted (changed from 0 to 1)
        valid_next_questions = []
        previous_state_vector = current_sequence[-1]
        for i, (prev, curr) in enumerate(zip(previous_state_vector, predicted_next_state_vector)):
            if prev == 0 and curr == 1:
                valid_next_questions.append(i)
        
        # Select from valid next questions
        if valid_next_questions:
            return random.choice(valid_next_questions)
        # If no valid next questions found, select from undone questions
        current_state_vector = current_sequence[-1]
        undone_questions = [i for i, done in enumerate(current_state_vector) if done == 0]
        
        if undone_questions:
            return random.choice(undone_questions)
        
        # If all questions are done, return -1 to indicate no more questions available
        return -1
    
    def get_question_id(self, question_index: int) -> str:
        """
        Get the actual question ID from the question index.
        
        Args:
            question_index: Index in the unit_questions list.
            
        Returns:
            Question ID string.
        """
        if self.unit_questions is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        if 0 <= question_index < len(self.unit_questions):
            return self.unit_questions[question_index]
        else:
            raise ValueError(f"Question index {question_index} is out of bounds")
    
    def get_multiple_predictions(self, current_sequence: List[Union[int, List[int]]], 
                               n_predictions: int = 5, method: str = 'max_probability') -> List[Tuple[int, float]]:
        """
        Get multiple next question predictions with their probabilities.
        
        Args:
            current_sequence: Current student's question sequence.
            n_predictions: Number of predictions to return.
            method: Currently only supports 'max_probability'.
            
        Returns:
            List of (question_index, probability) tuples, sorted by probability (descending).
        """
        if self.transition_matrix is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        if self.encoding_type == 'ordinal':
            return self._get_multiple_predictions_ordinal(current_sequence, n_predictions)
        elif self.encoding_type == 'one_hot':
            return self._get_multiple_predictions_one_hot(current_sequence, n_predictions)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
    
    def _get_multiple_predictions_ordinal(self, current_sequence: List[int], 
                                        n_predictions: int) -> List[Tuple[int, float]]:
        """Get multiple predictions for ordinal encoding."""
        if not current_sequence:
            # If no sequence, return top states by overall probability
            total_probs = np.sum(self.transition_matrix, axis=1)
            top_indices = np.argsort(total_probs)[-n_predictions:][::-1]
            return [(int(idx), float(total_probs[idx])) for idx in top_indices]
        
        current_state = current_sequence[-1]
        
        if current_state >= len(self.transition_matrix):
            return [(random.randint(0, len(self.transition_matrix) - 1), 0.0)]
        
        # Get transition probabilities from current state
        transition_probs = self.transition_matrix[current_state]
        
        # Get top n predictions
        top_indices = np.argsort(transition_probs)[-n_predictions:][::-1]
        predictions = [(int(idx), float(transition_probs[idx])) for idx in top_indices]
        
        return predictions
    
    def _get_multiple_predictions_one_hot(self, current_sequence: List[List[int]], 
                                        n_predictions: int) -> List[Tuple[int, float]]:
        """Get multiple predictions for one-hot encoding."""
        if not current_sequence:
            return [(random.randint(0, len(self.unit_questions) - 1), 0.0) for _ in range(n_predictions)]
        
        current_state_vector = current_sequence[-1]
        current_state_tuple = tuple(current_state_vector)
        
        # Find current state in the model (similar to _predict_one_hot)
        if current_state_tuple not in self.state_to_index:
            return [(random.randint(0, len(self.unit_questions) - 1), 0.0) for _ in range(n_predictions)]
        
        current_state_index = self.state_to_index[current_state_tuple]
        
        if current_state_index >= len(self.transition_matrix):
            return [(random.randint(0, len(self.unit_questions) - 1), 0.0) for _ in range(n_predictions)]
        
        # Get transition probabilities
        transition_probs = self.transition_matrix[current_state_index]
        
        # Get top next states
        top_state_indices = np.argsort(transition_probs)[-n_predictions:][::-1]
        
        predictions = []
        for state_idx in top_state_indices:
            if state_idx < len(self.index_to_state):
                predicted_state_vector = self.index_to_state[state_idx]
                
                # Find newly attempted questions
                valid_questions = []
                if len(current_sequence) > 1:
                    previous_state_vector = current_sequence[-2]
                    for i, (prev, curr) in enumerate(zip(previous_state_vector, predicted_state_vector)):
                        if prev == 0 and curr == 1:
                            valid_questions.append(i)
                else:
                    for i, val in enumerate(predicted_state_vector):
                        if val == 1:
                            valid_questions.append(i)
                
                # Add predictions for valid questions
                prob = float(transition_probs[state_idx])
                for q_idx in valid_questions:
                    predictions.append((q_idx, prob / len(valid_questions) if valid_questions else prob))
        
        # Sort by probability and return top n
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_predictions]
    
    def analyze_sequence(self, sequence: List[Union[int, List[int]]]) -> Dict[str, Any]:
        """
        Analyze a student sequence and provide insights.
        
        Args:
            sequence: Student's question sequence.
            
        Returns:
            Dictionary containing analysis results.
        """
        if self.transition_matrix is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        analysis = {
            'sequence_length': len(sequence),
            'unique_questions': None,
            'transition_probabilities': [],
            'average_transition_probability': 0.0,
            'next_predictions': []
        }
        
        if self.encoding_type == 'ordinal':
            analysis['unique_questions'] = len(set(sequence))
            
            # Calculate transition probabilities for the sequence
            transition_probs = []
            for i in range(len(sequence) - 1):
                current_state = sequence[i]
                next_state = sequence[i + 1]
                
                if (current_state < len(self.transition_matrix) and 
                    next_state < len(self.transition_matrix[current_state])):
                    prob = self.transition_matrix[current_state][next_state]
                    transition_probs.append(prob)
            
            analysis['transition_probabilities'] = transition_probs
            analysis['average_transition_probability'] = np.mean(transition_probs) if transition_probs else 0.0
        
        elif self.encoding_type == 'one_hot':
            # For one-hot, count unique question attempts
            attempted_questions = set()
            for state_vector in sequence:
                for i, attempted in enumerate(state_vector):
                    if attempted == 1:
                        attempted_questions.add(i)
            analysis['unique_questions'] = len(attempted_questions)
        
        # Get next predictions
        try:
            analysis['next_predictions'] = self.get_multiple_predictions(sequence, n_predictions=3)
        except:
            analysis['next_predictions'] = []
        
        return analysis

# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = MarkovPredictor()
    
    try:
        # Load a model (assuming it exists)
        predictor.load_model(unit=3, encoding_type='ordinal')
        
        # Test prediction with a simple sequence
        test_sequence = [0, 1, 2]  # Example ordinal sequence
        
        # Get single prediction
        next_question_idx = predictor.predict_next_question(test_sequence)
        next_question_id = predictor.get_question_id(next_question_idx)
        
        print(f"Current sequence: {test_sequence}")
        print(f"Predicted next question index: {next_question_idx}")
        print(f"Predicted next question ID: {next_question_id}")
        
        # Get multiple predictions
        multiple_predictions = predictor.get_multiple_predictions(test_sequence, n_predictions=5)
        print(f"Top 5 predictions: {multiple_predictions}")
        
        # Analyze sequence
        analysis = predictor.analyze_sequence(test_sequence)
        print(f"Sequence analysis: {analysis}")
        
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("Please train a model first using MarkovTrainer")
    except Exception as e:
        print(f"Error: {e}") 
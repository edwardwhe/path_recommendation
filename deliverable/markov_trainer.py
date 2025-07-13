import json
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.cluster import KMeans
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import os
import tqdm

class MarkovTrainer:
    """
    A comprehensive Markov model trainer for educational path recommendation.
    """
    
    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize the Markov trainer with database configuration.
        
        Args:
            db_config: Database configuration dictionary. If None, uses default config.
        """
        self.db_config = db_config or {
            'password': 'hf6Wbg3dm8',
            'url': 'mongodb://root:hf6Wbg3dm8@dds-3ns35de5eee23e941756-pub.mongodb.rds.aliyuncs.com:3717,dds-3ns35de5eee23e942366-pub.mongodb.rds.aliyuncs.com:3717/admin?replicaSet=mgset-33008719'
        }
        self.client = None
        self.db = None
        self._connect_to_database()
        
        # Time filtering thresholds (in milliseconds)
        self.min_duration = 2000  # 2 seconds
        self.max_duration = 93500  # 93.5 seconds
        
    def _connect_to_database(self):
        """Establish connection to MongoDB database."""
        try:
            self.client = MongoClient(self.db_config['url'])
            self.db = self.client['pro_mathaday_assessment']
            print("Connected to MongoDB successfully")
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            raise
    
    def get_assessment_question_by_unit(self, unit: int) -> Tuple[List[str], str]:
        """
        Get all assessment questions in the target unit from the database.
        
        Args:
            unit: The unit number that this transition matrix will be used for prediction.
            
        Returns:
            questions: All the assessment questions in this unit.
            path_to_all_questions: The path to folders that hold all the questions.
        """
        print(f"Getting assessment questions for unit {unit}")
        
        # First try to load from existing JSON file
        questions_file_path = f"./data/assessment_questions/{unit}.json"
        path_to_all_questions = "./data/assessment_questions/"
        
        if os.path.exists(questions_file_path):
            print(f"Loading questions from existing file: {questions_file_path}")
            with open(questions_file_path, 'r') as file:
                questions = json.load(file)
            return questions, path_to_all_questions
        
        # If file doesn't exist, query from database
        print(f"File not found, querying database for unit {unit} questions...")
        
        # Load objective to unit mapping
        obj_to_unit_path = "./data/obj_to_unit.json"
        if not os.path.exists(obj_to_unit_path):
            raise FileNotFoundError(f"obj_to_unit.json not found at {obj_to_unit_path}")
            
        with open(obj_to_unit_path, 'r') as file:
            obj_to_unit = json.load(file)
        
        # Query database for questions
        collection = self.db['assessment_questions']
        questions = []
        
        for item in collection.find():
            item_id = item['itemId']
            math_objectives = item.get('mathObjectives', [])
            
            for obj in math_objectives:
                if obj in obj_to_unit:
                    mapped_units = obj_to_unit[obj]
                    # Handle both single unit and list of units
                    if isinstance(mapped_units, (list, tuple)):
                        if unit in mapped_units and item_id not in questions:
                            questions.append(item_id)
                    else:
                        if mapped_units == unit and item_id not in questions:
                            questions.append(item_id)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(questions_file_path), exist_ok=True)
        
        # Save questions to file for future use
        with open(questions_file_path, 'w') as file:
            json.dump(questions, file)
        
        print(f"Found {len(questions)} questions for unit {unit}")
        return questions, path_to_all_questions
    
    def get_students(self, unit_questions: List[str], include_exam: bool = False, 
                    path_to_all_questions: Optional[str] = None, unit: Optional[int] = None) -> List[Dict]:
        """
        Get all students information from the database filtered by unit questions.
        
        Args:
            unit_questions: All the assessment questions of a unit.
            include_exam: Boolean, true means assessment questions in exams will be included.
            path_to_all_questions: The path to folders that hold all the questions (can be null).
            unit: Target unit number (can be null).
            
        Returns:
            students: A list of dictionaries containing student information.
        """
        print(f"Getting students data with {len(unit_questions)} unit questions")
        
        # If unit_questions is empty but path and unit are provided, load questions
        if not unit_questions and path_to_all_questions and unit:
            questions_file = os.path.join(path_to_all_questions, f"{unit}.json")
            with open(questions_file, 'r') as file:
                unit_questions = json.load(file)
        
        if not unit_questions:
            raise ValueError("unit_questions cannot be empty when path_to_all_questions and unit are not provided")
        
        # Get assessment info for filtering exam types
        assessments_collection = self.db['assessments']
        assessment_info = {}
        
        print("Loading assessment information...")
        for assessment in tqdm.tqdm(assessments_collection.find()):
            assessment_info[assessment["id"]] = {
                "type": assessment["type"],
                "syllabi": [syllable.get("unitId") for syllable in assessment.get("syllabi", [])]
            }
        
        # Get question records from database
        question_records_collection = self.db['assessment_question_records']
        user_ids = question_records_collection.distinct('userId')
        
        students = []
        print(f"Processing {len(user_ids)} students...")
        
        for user_id in tqdm.tqdm(user_ids):
            user_data = question_records_collection.find({"userId": user_id})
            assessments = []
            
            for record in user_data:
                # Skip if question not in unit questions
                if record.get("assessmentQuestionId") not in unit_questions:
                    continue
                
                # Get assessment info
                assessment = assessment_info.get(record["assessmentId"])
                if not assessment:
                    continue
                
                # Filter exam types if include_exam is False
                if not include_exam:
                    # Exclude exam types: 5 (exam) and 7 (other exam type)
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
            
            if assessments:
                # Sort assessments by start time
                assessments.sort(key=lambda x: x["start_time"])
                
                student = {
                    "user_id": user_id,
                    "assessments": assessments
                }
                students.append(student)
        
        print(f"Found {len(students)} students with relevant assessments")
        return students
    
    def filter_valid_students(self, students: List[Dict]) -> List[Dict]:
        """
        Filter students based on validity criteria (duration thresholds, etc.).
        
        Args:
            students: A list of dictionaries containing student information.
            
        Returns:
            students: Filtered list of valid students.
        """
        print("Filtering valid students...")
        valid_students = []
        
        for student in students:
            valid_assessments = []
            
            for assessment in student["assessments"]:
                # Filter by duration (exclude very short or very long attempts)
                duration = assessment.get("duration")
                if duration is None or duration < self.min_duration or duration > self.max_duration:
                    continue
                
                # Additional validity checks can be added here
                # For example: score validity, question validity, etc.
                if assessment.get("score") is None:
                    continue
                
                valid_assessments.append(assessment)
            
            # Only keep students with sufficient valid assessments
            if len(valid_assessments) >= 2:  # Need at least 2 assessments for transitions
                student["assessments"] = valid_assessments
                valid_students.append(student)
        
        print(f"Filtered to {len(valid_students)} valid students")
        return valid_students
    
    def _cluster_students(self, students: List[Dict], n_clusters: int = 3) -> pd.DataFrame:
        """
        Cluster students based on their learning behavior patterns.
        
        Args:
            students: List of student dictionaries.
            n_clusters: Number of clusters to create.
            
        Returns:
            DataFrame with student clusters.
        """
        print("Clustering students based on learning patterns...")
        
        # Load assessment question info for difficulty weighting
        try:
            with open("./data/assessment_question_info.json", 'r') as file:
                question_info = json.load(file)
        except FileNotFoundError:
            # If file doesn't exist, create simplified difficulty mapping
            question_info = {}
            for student in students:
                for assessment in student["assessments"]:
                    if assessment["question"] not in question_info:
                        question_info[assessment["question"]] = {"difficulty": 2}  # Default medium
        
        difficulty_mapping = {1: 1, 2: 1.2, 3: 2}
        student_features = []
        
        for student in students:
            # Calculate level (weighted average score)
            total_weighted_score = 0
            for assessment in student["assessments"]:
                difficulty = question_info.get(assessment["question"], {}).get("difficulty", 2)
                score = assessment.get("score", 0)
                total_weighted_score += score * difficulty_mapping.get(difficulty, 1.2)
            
            level = total_weighted_score / len(student["assessments"]) if student["assessments"] else 0
            
            # Calculate intensity and regularity
            start_times = [datetime.fromtimestamp(assessment["start_time"]) for assessment in student["assessments"]]
            week_numbers = [time.isocalendar()[1] for time in start_times]
            week_counts = {}
            for week in week_numbers:
                week_counts[week] = week_counts.get(week, 0) + 1
            
            intensity = sum(week_counts.values()) / len(week_counts) if week_counts else 0
            regularity = len(week_counts)
            
            student_features.append({
                "user_id": student["user_id"],
                "level": level,
                "intensity": intensity,
                "regularity": regularity
            })
        
        # Create DataFrame and normalize features
        df = pd.DataFrame(student_features)
        normalized_df = df.copy()
        
        for column in ['level', 'intensity', 'regularity']:
            if df[column].std() > 0:
                mean = df[column].mean()
                std = df[column].std()
                normalized_df[column] = (df[column] - mean) / std
            else:
                normalized_df[column] = 0
        
        # Perform K-means clustering
        if len(df) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            clusters = kmeans.fit_predict(normalized_df[['level', 'intensity', 'regularity']])
            df['cluster'] = clusters
        else:
            # If not enough students, assign all to cluster 0
            df['cluster'] = 0
        
        print(f"Created {n_clusters} clusters from {len(df)} students")
        return df
    
    def _encode_sequences(self, students: List[Dict], unit_questions: List[str], 
                         encoding_type: str = 'ordinal') -> List[List]:
        """
        Encode student sequences for Markov training.
        
        Args:
            students: List of student dictionaries.
            unit_questions: List of questions for encoding.
            encoding_type: 'ordinal' or 'one_hot'.
            
        Returns:
            List of encoded sequences.
        """
        print(f"Encoding sequences using {encoding_type} encoding...")
        
        if encoding_type == 'ordinal':
            return self._encode_ordinal(students, unit_questions)
        elif encoding_type == 'one_hot':
            return self._encode_one_hot(students, unit_questions)
        else:
            raise ValueError("encoding_type must be either 'ordinal' or 'one_hot'")
    
    def _encode_ordinal(self, students: List[Dict], unit_questions: List[str]) -> List[List[int]]:
        """Encode sequences as ordinal (question indices)."""
        sequences = []
        for student in students:
            sequence = []
            for assessment in student["assessments"]:
                if assessment["question"] in unit_questions:
                    question_index = unit_questions.index(assessment["question"])
                    sequence.append(question_index)
            if sequence:
                sequences.append(sequence)
        return sequences
    
    def _encode_one_hot(self, students: List[Dict], unit_questions: List[str]) -> List[List[List[int]]]:
        """Encode sequences as one-hot vectors."""
        sequences = []
        for student in students:
            sequence = []
            tracking_vector = [0] * len(unit_questions)
            
            for assessment in student["assessments"]:
                if assessment["question"] in unit_questions:
                    question_index = unit_questions.index(assessment["question"])
                    tracking_vector[question_index] = 1
                    sequence.append(tracking_vector.copy())
            
            if sequence:
                sequences.append(sequence)
        return sequences
    
    def _train_markov_ordinal(self, sequences: List[List[int]], num_states: int) -> Tuple[np.ndarray, List[int]]:
        """Train Markov model with ordinal encoding."""
        print(f"Training Markov model (ordinal) with {len(sequences)} sequences and {num_states} states")
        
        # Initialize transition count matrix
        transition_counts = np.zeros((num_states, num_states))
        
        # Count transitions
        for i, sequence in enumerate(sequences):
            for j in range(len(sequence) - 1):
                current_state = sequence[j]
                next_state = sequence[j + 1]
                if 0 <= current_state < num_states and 0 <= next_state < num_states:
                    transition_counts[current_state][next_state] += 1
        
        # Create transition probability matrix
        transition_matrix = np.zeros((num_states, num_states))
        for i in range(num_states):
            total_transitions = np.sum(transition_counts[i])
            if total_transitions > 0:
                transition_matrix[i] = transition_counts[i] / total_transitions
        
        state_indices = list(range(num_states))
        return transition_matrix, state_indices
    
    def _train_markov_one_hot(self, sequences: List[List[List[int]]]) -> Tuple[np.ndarray, List[Tuple], Dict]:
        """Train Markov model with one-hot encoding."""
        print(f"Training Markov model (one-hot) with {len(sequences)} sequences")
        
        # Define unique states
        state_to_index = {}
        index_to_state = []
        
        # Collect all unique states
        for sequence in sequences:
            for state in sequence:
                state_tuple = tuple(state)
                if state_tuple not in state_to_index:
                    state_to_index[state_tuple] = len(state_to_index)
                    index_to_state.append(state_tuple)
        
        num_states = len(index_to_state)
        print(f"Total unique states found: {num_states}")
        
        # Initialize transition count matrix
        transition_counts = np.zeros((num_states, num_states))
        
        # Count transitions
        for sequence in sequences:
            for j in range(len(sequence) - 1):
                current_state = tuple(sequence[j])
                next_state = tuple(sequence[j + 1])
                
                current_index = state_to_index[current_state]
                next_index = state_to_index[next_state]
                
                transition_counts[current_index][next_index] += 1
        
        # Create transition probability matrix
        transition_matrix = np.zeros((num_states, num_states))
        for i in range(num_states):
            total_transitions = np.sum(transition_counts[i])
            if total_transitions > 0:
                transition_matrix[i] = transition_counts[i] / total_transitions
        
        return transition_matrix, index_to_state, state_to_index
    
    def train_transition_matrix(self, unit: int, encoding_type: str = 'ordinal', 
                              cluster_index: Optional[int] = None, include_exam: bool = False) -> Tuple[np.ndarray, List[str], Any, Any]:
        """
        Train and return the Markov transition matrix.
        
        Args:
            unit: The unit number that this transition matrix will be used for prediction.
            encoding_type: 'ordinal' or 'one_hot'.
            cluster_index: Specific cluster to use (0, 1, 2) or None to combine all clusters.
            include_exam: Whether to include exam questions in training.
            
        Returns:
            transition_matrix: Markov transition matrix (N * N).
            unit_questions: A list of questions (N) used to encode the state order by id.
            index_to_state: Mapping from index to state.
            state_to_index: Mapping from state to index.
        """
        print(f"Training transition matrix for unit {unit} with {encoding_type} encoding")
        
        # Step 1: Obtain questions from database by unit
        unit_questions, _ = self.get_assessment_question_by_unit(unit)
        
        # Step 2: Get corresponding students
        students = self.get_students(unit_questions, include_exam)
        
        # Step 3: Filter valid students
        students = self.filter_valid_students(students)
        
        if not students:
            raise ValueError(f"No valid students found for unit {unit}")
        
        # Step 4: Cluster students and filter by cluster if specified
        if cluster_index is not None:
            cluster_df = self._cluster_students(students)
            if cluster_index not in cluster_df['cluster'].values:
                raise ValueError(f"Cluster {cluster_index} not found")
            
            # Filter students by cluster
            cluster_user_ids = cluster_df[cluster_df['cluster'] == cluster_index]['user_id'].tolist()
            students = [s for s in students if s['user_id'] in cluster_user_ids]
            print(f"Using cluster {cluster_index} with {len(students)} students")
        
        # Step 5: Encode sequences
        sequences = self._encode_sequences(students, unit_questions, encoding_type)
        
        if not sequences:
            raise ValueError("No valid sequences found for training")
        
        # Step 6: Train Markov model
        if encoding_type == 'ordinal':
            transition_matrix, index_to_state = self._train_markov_ordinal(sequences, len(unit_questions))
            state_to_index = None
        elif encoding_type == 'one_hot':
            transition_matrix, index_to_state, state_to_index = self._train_markov_one_hot(sequences)
        else:
            raise ValueError("encoding_type must be either 'ordinal' or 'one_hot'")
        
        print(f"Training completed. Transition matrix shape: {transition_matrix.shape}")
        return transition_matrix, unit_questions, index_to_state, state_to_index
    
    def save_model(self, transition_matrix: np.ndarray, unit_questions: List[str], 
                   index_to_state: Any, state_to_index: Any, 
                   unit: int, encoding_type: str, cluster_index: Optional[int] = None):
        """
        Save the trained model to files.
        
        Args:
            transition_matrix: The trained transition matrix.
            unit_questions: List of questions.
            index_to_state: State index mapping.
            state_to_index: State to index mapping.
            unit: Unit number.
            encoding_type: Encoding type used.
            cluster_index: Cluster index if used.
        """
        # Create filename suffix
        suffix = f"unit_{unit}_{encoding_type}"
        if cluster_index is not None:
            suffix += f"_cluster_{cluster_index}"
        
        # Save transition matrix
        np.save(f'transition_matrix_{suffix}.npy', transition_matrix)
        
        # Save questions
        with open(f'unit_questions_{suffix}.json', 'w') as f:
            json.dump(unit_questions, f)
        
        # Save state mappings
        if encoding_type == 'one_hot':
            np.save(f'index_to_state_{suffix}.npy', np.array(index_to_state, dtype=object))
            with open(f'state_to_index_{suffix}.json', 'w') as f:
                # Convert tuple keys to strings for JSON serialization
                json_compatible = {str(k): v for k, v in state_to_index.items()}
                json.dump(json_compatible, f)
        
        print(f"Model saved with suffix: {suffix}")
    
    def __del__(self):
        """Close database connection on destruction."""
        if self.client:
            self.client.close()

# Example usage and testing
if __name__ == "__main__":
    # Initialize trainer
    trainer = MarkovTrainer()
    
    # Train a model for unit 3
    try:
        transition_matrix, unit_questions, index_to_state, state_to_index = trainer.train_transition_matrix(
            unit=3, 
            encoding_type='ordinal', 
            cluster_index=None,
            include_exam=False
        )
        
        # Save the model
        trainer.save_model(
            transition_matrix, 
            unit_questions, 
            index_to_state, 
            state_to_index,
            unit=3,
            encoding_type='ordinal'
        )
        
        print("Training completed successfully!")
        print(f"Transition matrix shape: {transition_matrix.shape}")
        print(f"Number of questions: {len(unit_questions)}")
        
    except Exception as e:
        print(f"Training failed: {e}") 
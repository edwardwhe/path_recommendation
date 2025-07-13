# Markov-based Educational Path Recommendation System

A comprehensive system for training and using Markov models to recommend educational paths based on student learning patterns.

## Features

- **Training**: Train Markov models from student assessment data
- **Prediction**: Predict next questions based on student sequences
- **Multiple Encoding Types**: Support for ordinal and one-hot encoding
- **Student Clustering**: Group students by learning patterns
- **Database Integration**: Direct MongoDB connectivity
- **Command Line Interface**: Easy-to-use CLI for all operations

## Installation

### Prerequisites

- Python 3.7+
- MongoDB database access
- Required Python packages (see requirements.txt)

### Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure your MongoDB database is accessible and contains the required collections:
   - `assessment_question_records`
   - `assessments` 
   - `assessment_questions`

3. Make sure you have the required data files in the parent directory:
   - `../data/obj_to_unit.json`
   - `../data/assessment_questions/{unit}.json` (optional, will be generated if missing)

## Usage

### Command Line Interface

The system provides three main modes of operation:

#### 1. Training a Model

Train a Markov model for a specific unit:

```bash
python main.py train --unit 3 --encoding ordinal
```

Options:
- `--unit`: Unit number (default: 3)
- `--encoding`: Encoding type (`ordinal` or `one_hot`, default: `ordinal`)
- `--cluster`: Specific cluster index (0, 1, 2) or None for all clusters
- `--include-exam`: Include exam questions in training

#### 2. Making Predictions

Make predictions using a trained model:

```bash
python main.py predict --unit 3 --encoding ordinal --sequence "[0, 2, 5]"
```

Options:
- `--sequence`: JSON string of current student sequence
- `--model-dir`: Directory containing model files (default: current directory)

#### 3. Demo Mode

Run a complete demonstration workflow:

```bash
python main.py demo --unit 3 --encoding ordinal
```

### Python API Usage

#### Basic Training and Prediction

```python
from main import PathRecommendationSystem

# Initialize the system
system = PathRecommendationSystem()

# Train a model
model_info = system.train_model(
    unit=3,
    encoding_type='ordinal',
    cluster_index=None,  # Use all clusters
    include_exam=False
)

# Load the model for prediction
system.load_model(unit=3, encoding_type='ordinal')

# Make predictions
current_sequence = [0, 2, 5]  # Student's question sequence
next_question = system.predict_next_question(current_sequence)
print(f"Next recommended question: {next_question}")

# Get multiple predictions
top_predictions = system.get_multiple_predictions(current_sequence, n_predictions=5)
print(f"Top 5 recommendations: {top_predictions}")

# Analyze student sequence
analysis = system.analyze_student_sequence(current_sequence)
print(f"Sequence analysis: {analysis}")
```

#### Advanced Usage with Custom Database Config

```python
from markov_trainer import MarkovTrainer
from markov_predictor import MarkovPredictor

# Custom database configuration
db_config = {
    'password': 'your_password',
    'url': 'your_mongodb_url'
}

# Initialize trainer with custom config
trainer = MarkovTrainer(db_config)

# Train with specific parameters
transition_matrix, unit_questions, index_to_state, state_to_index = trainer.train_transition_matrix(
    unit=3,
    encoding_type='one_hot',
    cluster_index=1,  # Use only cluster 1
    include_exam=True
)

# Save the model
trainer.save_model(
    transition_matrix, 
    unit_questions, 
    index_to_state, 
    state_to_index,
    unit=3,
    encoding_type='one_hot',
    cluster_index=1
)
```

## API Reference

### PathRecommendationSystem

Main interface class that combines training and prediction functionality.

#### Methods

- `train_model(unit, encoding_type='ordinal', cluster_index=None, include_exam=False)`: Train a new model
- `load_model(unit, encoding_type='ordinal', cluster_index=None, model_dir='.')`: Load a trained model
- `predict_next_question(current_sequence, method='max_probability', return_question_id=True)`: Predict next question
- `get_multiple_predictions(current_sequence, n_predictions=5, return_question_ids=True)`: Get multiple predictions
- `analyze_student_sequence(sequence)`: Analyze a student's learning sequence

### MarkovTrainer

Handles the training of Markov models from database data.

#### Key Methods

- `get_assessment_question_by_unit(unit)`: Get questions for a specific unit
- `get_students(unit_questions, include_exam=False)`: Get student data filtered by unit questions
- `filter_valid_students(students)`: Filter students based on validity criteria
- `train_transition_matrix(unit, encoding_type='ordinal', cluster_index=None, include_exam=False)`: Train the main model

### MarkovPredictor

Handles prediction using trained Markov models.

#### Key Methods

- `load_model(unit, encoding_type='ordinal', cluster_index=None, model_dir='.')`: Load a trained model
- `predict_next_question(current_sequence, method='max_probability')`: Single prediction
- `get_multiple_predictions(current_sequence, n_predictions=5)`: Multiple predictions
- `analyze_sequence(sequence)`: Sequence analysis

## Data Formats

### Encoding Types

#### Ordinal Encoding
Student sequences are represented as lists of question indices:
```python
sequence = [0, 2, 5, 1]  # Question indices in order attempted
```

#### One-Hot Encoding
Student sequences are represented as lists of binary vectors:
```python
sequence = [
    [1, 0, 0, 0],  # Only question 0 attempted
    [1, 0, 1, 0],  # Questions 0 and 2 attempted
    [1, 0, 1, 1]   # Questions 0, 2, and 3 attempted
]
```

### Model Files

Trained models are saved as:
- `transition_matrix_{suffix}.npy`: Transition probability matrix
- `unit_questions_{suffix}.json`: List of questions in the unit
- `index_to_state_{suffix}.npy`: State index mapping (one-hot only)
- `state_to_index_{suffix}.json`: State to index mapping (one-hot only)

Where `{suffix}` follows the pattern: `unit_{unit}_{encoding_type}[_cluster_{cluster_index}]`

## Examples

### Training Different Model Types

```python
system = PathRecommendationSystem()

# Train ordinal model for all students
system.train_model(unit=3, encoding_type='ordinal')

# Train one-hot model for cluster 0 only
system.train_model(unit=3, encoding_type='one_hot', cluster_index=0)

# Train model including exam questions
system.train_model(unit=3, encoding_type='ordinal', include_exam=True)
```

### Making Predictions with Different Methods

```python
# Load model
system.load_model(unit=3, encoding_type='ordinal')

# Get most probable next question
next_q = system.predict_next_question([0, 2, 5], method='max_probability')

# Get weighted random prediction
next_q_random = system.predict_next_question([0, 2, 5], method='weighted_random')

# Get top 3 recommendations with probabilities
top_3 = system.get_multiple_predictions([0, 2, 5], n_predictions=3)
```

## Error Handling

The system includes comprehensive error handling for common issues:

- **Database Connection**: Clear error messages for connection failures
- **Missing Files**: Automatic file generation where possible
- **Invalid Sequences**: Graceful handling of out-of-bounds sequences
- **Model Loading**: Informative errors for missing model files

## Performance Considerations

- **Database Queries**: Student data is cached during training to minimize database calls
- **Memory Usage**: Large transition matrices are handled efficiently with NumPy
- **File I/O**: Models are saved in compressed NumPy format for fast loading

## Troubleshooting

### Common Issues

1. **"No valid students found"**: Check that your unit has sufficient assessment data
2. **"Transition matrix not found"**: Ensure you've trained a model before trying to load it
3. **"Database connection failed"**: Verify your MongoDB credentials and network access
4. **"Question index out of bounds"**: Check that your sequence uses valid question indices

### Debug Mode

For debugging, you can access internal components:

```python
system = PathRecommendationSystem()

# Access trainer directly
trainer = system.trainer
questions, path = trainer.get_assessment_question_by_unit(3)
print(f"Found {len(questions)} questions")

# Access predictor directly
predictor = system.predictor
# ... after loading a model
analysis = predictor.analyze_sequence([0, 1, 2])
```

## Contributing

When contributing to this project:

1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings to new functions
3. Include error handling for edge cases
4. Test with both encoding types (ordinal and one-hot)
5. Verify database connectivity works with your changes

## License

This project is part of an educational research system. Please refer to your institution's guidelines for usage and distribution. 
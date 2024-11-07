import pandas as pd
import numpy as np

# Sample dataset: A sequence of topic interactions (replace with your actual data)
# Assume the data is in the format: student_id, topic, next_topic
data = pd.DataFrame({
    'student_id': [1, 1, 1, 2, 2, 2, 3, 3],
    'topic': ['A', 'B', 'C', 'A', 'C', 'B', 'A', 'B'],
    'next_topic': ['B', 'C', 'A', 'C', 'B', 'A', 'B', 'C']
})

# Calculate transition probabilities
# Create a transition matrix with topics as both rows and columns
topics = data['topic'].unique()
transition_matrix = pd.DataFrame(0, index=topics, columns=topics, dtype=float)

# Count the transitions from each topic to the next
for _, row in data.iterrows():
    current_topic = row['topic']
    next_topic = row['next_topic']
    transition_matrix.loc[current_topic, next_topic] += 1

# Normalize each row to get probabilities
transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)

print("Transition Matrix:")
print(transition_matrix)

# Function to recommend the next topic based on the current topic
def recommend_next_topic(current_topic, transition_matrix):
    if current_topic not in transition_matrix.index:
        print(f"Topic '{current_topic}' not found in transition matrix.")
        return None
    next_topic_probs = transition_matrix.loc[current_topic]
    recommended_topic = np.random.choice(next_topic_probs.index, p=next_topic_probs.values)
    return recommended_topic

# Example usage: Get the next recommended topic from a given current topic
current_topic = 'A'
recommended_topic = recommend_next_topic(current_topic, transition_matrix)
print(f"Recommended next topic after '{current_topic}': {recommended_topic}")

# To create a learning path, start from an initial topic and keep recommending until desired length
def generate_learning_path(start_topic, transition_matrix, path_length=5):
    learning_path = [start_topic]
    current_topic = start_topic
    for _ in range(path_length - 1):
        next_topic = recommend_next_topic(current_topic, transition_matrix)
        if next_topic is None:
            break  # No further recommendation if the topic doesn't exist in matrix
        learning_path.append(next_topic)
        current_topic = next_topic
    return learning_path

# Example: Generate a learning path starting from topic 'A'
start_topic = 'A'
learning_path = generate_learning_path(start_topic, transition_matrix, path_length=5)
print(f"Generated learning path: {learning_path}")

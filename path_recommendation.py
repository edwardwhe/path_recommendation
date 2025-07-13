import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
# from data_preprocessing import DataPreprocessing
from data_analysis import DataAnalysis

def group_students(student_attributes_list, n_clusters = 3):
    # Function: group students into three clusters: regular learning, intensive learning and, advanced learning.
    # Input: [[1, 2, 2, 1], [2, 3, 1, 3], [5, 3, 1, 2]], where each sublist is a student, and each student has four attributes (learning duration, learning frequency, learning intensity, learning profciency)
    data = np.array(student_attributes_list)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    # get labels
    labels = kmeans.labels_
    df = pd.DataFrame(data)
    print(labels)
    df['Cluster'] = labels

    # calculate the means and std of each cluster
    means = df.groupby('Cluster').mean()
    stds = df.groupby('Cluster').std()

    # plot
    plt.figure(figsize=(12, 8))
    num_dimensions = df.shape[1] - 1

    for i in range(n_clusters):
        cluster_data = df[df['Cluster'] == i]
        for dim in range(num_dimensions):
            sns.kdeplot(cluster_data[dim], label=f'Cluster {i + 1} - Dimension {dim + 1}', fill=True, alpha=0.5)

    plt.title('Distribution of Dimensions by Cluster')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    return means, stds, labels

def train_markov_base(student_states_list):
    # Function: group students into three clusters: regular learning, intensive learning and, advanced learning.
    # Input: [(student1)[[0, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 0]], (student2)[[1, 1, 1, 0], [1, 1, 1, 1]]], where each sublist is a student, and subsublist is the one-hot encoding of students' questions
    # Note: we don't consider repeated questions
    
    print(f"Training Markov model with {len(student_states_list)} students")
    
    # Step 1: Define the unique states
    state_to_index = {}
    index_to_state = []
    
    # Collect all unique states
    for student_states in student_states_list:
        for state in student_states:
            state_tuple = tuple(state)
            if state_tuple not in state_to_index:
                state_to_index[state_tuple] = len(state_to_index)
                index_to_state.append(state_tuple)
    
    num_states = len(index_to_state)
    print(f"Total unique states found: {num_states}")
    
    # Step 2: Initialize the transition count matrix
    transition_counts = np.zeros((num_states, num_states))
    
    # Count transitions
    for i, student_states in enumerate(student_states_list):
        print(f"Student {i+1}: {len(student_states)} states")
        for j in range(len(student_states) - 1):
            current_state = tuple(student_states[j])
            next_state = tuple(student_states[j + 1])
            
            current_index = state_to_index[current_state]
            next_index = state_to_index[next_state]
            
            transition_counts[current_index][next_index] += 1
    
    # Step 3: Create the transition probability matrix
    transition_probabilities = np.zeros((num_states, num_states))
    
    for i in range(num_states):
        total_transitions = np.sum(transition_counts[i])
        if total_transitions > 0:
            transition_probabilities[i] = transition_counts[i] / total_transitions
    
    return transition_probabilities, index_to_state, state_to_index

def train_markov_ordinal(student_states_list, num_states):
    # Function: Train Markov model where each state is the index of the problem attempted (ordinal-based)
    # Input: student_states_list: list of lists, each sublist is a sequence of problem indices (ints)
    #        num_states: total number of unique problems (size of question pool)
    print(f"Training Markov model (ordinal-based) with {len(student_states_list)} students and {num_states} states")

    # Step 1: Initialize the transition count matrix
    transition_counts = np.zeros((num_states, num_states))

    # Step 2: Count transitions
    for i, student_states in enumerate(student_states_list):
        print(f"Student {i+1}: {len(student_states)} states")
        for j in range(len(student_states) - 1):
            current_index = student_states[j]
            next_index = student_states[j + 1]
            transition_counts[current_index][next_index] += 1

    # Step 3: Create the transition probability matrix
    transition_probabilities = np.zeros((num_states, num_states))
    for i in range(num_states):
        total_transitions = np.sum(transition_counts[i])
        if total_transitions > 0:
            transition_probabilities[i] = transition_counts[i] / total_transitions

    # The states are simply 0 to num_states-1
    state_indices = list(range(num_states))
    return transition_probabilities, state_indices

def train_markov(encoding_type='one_hot', cluster_index=None):
    # Function: Unified Markov training function that combines training set generation and Markov training
    # Input: encoding_type: 'one_hot' or 'ordinal'
    #        cluster_index: specific cluster to use (0, 1, 2) or None to combine all clusters
    # Returns: transition_matrix, states
    
    dataAnalysis = DataAnalysis()
    
    # Get training data with specified encoding
    cluster_data = dataAnalysis.training_set(encoding_type)
    
    # Combine all clusters if no specific cluster is specified
    if cluster_index is None:
        combined_data = []
        for cluster in cluster_data:
            combined_data.extend(cluster)
        print(f"Combined {len(combined_data)} students from all clusters")
    else:
        if cluster_index < 0 or cluster_index >= len(cluster_data):
            raise ValueError(f"cluster_index must be between 0 and {len(cluster_data)-1}")
        combined_data = cluster_data[cluster_index]
        print(f"Using cluster {cluster_index} with {len(combined_data)} students")
    
    index_to_state = []
    state_to_index = {}
    # Train Markov model based on encoding type
    if encoding_type == 'one_hot':
        transition_matrix, index_to_state, state_to_index = train_markov_base(combined_data)
    elif encoding_type == 'ordinal':
        # Get the number of states from the question pool
        assessment_questions_3 = dataAnalysis.util.load_json(dataAnalysis.util, "data/assessment_questions/3.json")
        num_states = len(assessment_questions_3)
        transition_matrix, index_to_state = train_markov_ordinal(combined_data, num_states)
    else:
        raise ValueError("encoding_type must be either 'one_hot' or 'ordinal'")
    
    return transition_matrix, index_to_state, state_to_index

def predict_next_question(transition_matrix, encoding_type, current_sequence, question_ids=None, state_to_index=None, index_to_state=None):
    """
    Predict the next question based on the current student sequence and transition matrix.
    
    Args:
        transition_matrix: The trained transition probability matrix
        encoding_type: 'one_hot' or 'ordinal'
        current_sequence: Current student's question sequence
        question_ids: List of question IDs (needed for ordinal encoding)
        state_to_index: Mapping from state to index (needed for one_hot encoding)
        index_to_state: Mapping from index to state (needed for one_hot encoding)
    
    Returns:
        predicted_question_index: Index of the predicted next question
    """
    
    if encoding_type == 'ordinal':
        # For ordinal encoding, current_sequence is a list of question indices
        if not current_sequence:
            # If no sequence, return random question based on initial probabilities
            return random.randint(0, len(transition_matrix) - 1)
        
        current_state = current_sequence[-1]  # Last question in sequence
        if current_state >= len(transition_matrix):
            raise ValueError(f"Current state {current_state} is out of bounds for transition matrix")
        
        # Get transition probabilities from current state
        transition_probs = transition_matrix[current_state]
        
        # Find the most likely next state
        predicted_next_state = np.argmax(transition_probs)
        
        return predicted_next_state
        
    elif encoding_type == 'one_hot':
        # For one_hot encoding, current_sequence is a list of state vectors
        if not current_sequence:
            # If no sequence, return random question based on initial probabilities
            return random.randint(0, len(question_ids) - 1)
        
        current_state_vector = current_sequence[-1]  # Last state vector
        current_state_tuple = tuple(current_state_vector)
        
        # If current state not found, find the closest state by vector multiplication
        if current_state_tuple not in state_to_index:
            max_similarity = -1
            closest_states = []
            
            for state_tuple, state_index in state_to_index.items():
                state_vector = np.array(state_tuple)
                current_vector = np.array(current_state_vector)
                similarity = np.dot(state_vector, current_vector)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    closest_states = [state_index]
                elif similarity == max_similarity:
                    closest_states.append(state_index)
            
            # Randomly select among the closest states
            current_state_index = random.choice(closest_states)
        else:
            current_state_index = state_to_index[current_state_tuple]
        
        if current_state_index >= len(transition_matrix):
            raise ValueError(f"Current state index {current_state_index} is out of bounds for transition matrix")
        
        # Get transition probabilities from current state
        transition_probs = transition_matrix[current_state_index]
        
        # Find the most likely next state
        predicted_next_state_index = np.argmax(transition_probs)
        
        if predicted_next_state_index >= len(index_to_state):
            raise ValueError(f"Predicted state index {predicted_next_state_index} is out of bounds for index_to_state")
        
        predicted_next_state_vector = index_to_state[predicted_next_state_index]
        
        # Find all questions that were newly attempted (changed from 0 to 1)
        valid_next_questions = []
        if len(current_sequence) > 1:
            previous_state_vector = current_sequence[-2]
            for i, (prev, curr) in enumerate(zip(previous_state_vector, predicted_next_state_vector)):
                if prev == 0 and curr == 1:
                    valid_next_questions.append(i)
        else:
            # If this is the first question, find all 1s in the predicted state
            for i, val in enumerate(predicted_next_state_vector):
                if val == 1:
                    valid_next_questions.append(i)
        
        # Randomly select among valid next questions
        if valid_next_questions:
            return random.choice(valid_next_questions)
        
        # If no valid next questions found, randomly select one question not done by student
        current_state_vector = current_sequence[-1]
        undone_questions = [i for i, done in enumerate(current_state_vector) if done == 0]
        
        if undone_questions:
            return random.choice(undone_questions)
        
        # If all questions are done, return random
        return random.randint(0, len(question_ids) - 1)
    
    else:
        raise ValueError("encoding_type must be either 'one_hot' or 'ordinal'")

if __name__ == "__main__":
    # student_states_list = [
    #     [[0, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 0]],
    #     [[1, 1, 1, 0], [1, 1, 1, 1]],
    #     [[0, 0, 1, 0], [0, 0, 1, 1]],
    # ]
    dataAnalysis = DataAnalysis()
    cluster_0_matrix, cluster_1_matrix, cluster_2_matrix = dataAnalysis.training_set()

    transition_matrix, states = train_markov_base(cluster_0_matrix)
    
    # Save transition matrix to file
    np.save('transition_matrix.npy', transition_matrix)
    np.save('states.npy', np.array(states, dtype=object))
    
    # Output the transition matrix and states
    # print("States:", states)
    # print("Transition Matrix:\n", transition_matrix)
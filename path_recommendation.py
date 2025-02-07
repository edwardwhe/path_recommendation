import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
    
    # Step 2: Initialize the transition count matrix
    transition_counts = np.zeros((num_states, num_states))
    
    # Count transitions
    for student_states in student_states_list:
        for i in range(len(student_states) - 1):
            current_state = tuple(student_states[i])
            next_state = tuple(student_states[i + 1])
            
            current_index = state_to_index[current_state]
            next_index = state_to_index[next_state]
            
            transition_counts[current_index][next_index] += 1
    
    # Step 3: Create the transition probability matrix
    transition_probabilities = np.zeros((num_states, num_states))
    
    for i in range(num_states):
        total_transitions = np.sum(transition_counts[i])
        if total_transitions > 0:
            transition_probabilities[i] = transition_counts[i] / total_transitions
    
    return transition_probabilities, index_to_state

def train_markov_advanced(student_states_list):
    # Function: group students into three clusters: regular learning, intensive learning and, advanced learning.
    # Input: [(student1)[[0, 0, 1, 0,// 0, 0, 5, 0, // 0, 0, 2, 0], ...], (student2)[[...]]], where each sublist is a student, and subsublist is the one-hot encoding of students' questions
    # Note: consider more information, like time, question type, ...

    transition_probabilities, index_to_state = train_markov_base(student_states_list)
    # Todo: combine



if __name__ == "__main__":
    student_states_list = [
        [[0, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 0]],
        [[1, 1, 1, 0], [1, 1, 1, 1]],
        [[0, 0, 1, 0], [0, 0, 1, 1]],
    ]
    dataAnalysis = DataAnalysis()
    cluster_0_matrix, cluster_1_matrix, cluster_2_matrix = dataAnalysis.training_set()

    transition_matrix, states = train_markov_base(cluster_0_matrix)
    
    # Output the transition matrix and states
    print("States:", states)
    print("Transition Matrix:\n", transition_matrix)
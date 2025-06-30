import numpy as np
import pandas as pd

def convert_transition_matrix_to_csv():
    # Load the transition matrix and states
    transition_matrix = np.load('transition_matrix.npy')
    states = np.load('states.npy', allow_pickle=True)
    
    # Create a DataFrame with states as row and column labels
    df = pd.DataFrame(transition_matrix, 
                     index=[f'State_{i}' for i in range(len(states))],
                     columns=[f'State_{i}' for i in range(len(states))])
    
    # Save to CSV
    df.to_csv('transition_matrix.csv')
    print(f"Transition matrix saved to transition_matrix.csv")
    print(f"Matrix shape: {transition_matrix.shape}")
    
    # Also save states mapping to a separate CSV for reference
    states_df = pd.DataFrame({
        'State_Index': range(len(states)),
        'State_Vector': [str(state) for state in states]
    })
    states_df.to_csv('states_mapping.csv', index=False)
    print(f"States mapping saved to states_mapping.csv")

if __name__ == "__main__":
    convert_transition_matrix_to_csv() 
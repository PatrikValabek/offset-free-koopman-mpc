import numpy as np

def generate_steps(step_time: float, how_many: int, constraints: list, seed: int = 42):
    """
    Get the time points where the range changes. Range should be a numpy array representing a range of values. In left column min, in right column max.
    """
    np.random.seed(seed)
    
    step_changes = np.zeros((how_many*step_time, constraints.shape[0]))
    for i in range(how_many):
        for j in range(constraints.shape[0]):
            step_changes[i*step_time:(i+1)*step_time,j] = np.random.uniform(low=constraints[j,0], high=constraints[j,1])
        
    return step_changes
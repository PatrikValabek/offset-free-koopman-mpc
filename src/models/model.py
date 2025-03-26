
class Model:
    def __init__(self, model_name = "Generic Model"):
        self.model_name = model_name
        
    def describe(self):
        print("Model Name: ", self.model_name)
    
    def log_state(self, state: dict):
        self.history.append(state.copy())
        
    def get_history(self):
        return self.history
    
    def reset_history(self):
        self.history = []
        
    
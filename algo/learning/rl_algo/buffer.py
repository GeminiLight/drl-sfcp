class RolloutBuffer:
    
    def __init__(self):
        self.actions = []
        self.obervations = []
        self.logprobs = []
        self.rewards = []
        self.not_dones = []
    
    def clear(self):
        del self.actions[:]
        del self.obervations[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.not_dones[:]

    def size(self):
        return len(self.actions)
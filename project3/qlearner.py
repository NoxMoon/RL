import numpy as np

class QLearner():
    
    class agent():
        def __init__(self, nS, na):
            self.Q = np.ones([nS, na]) # nstate, my action
        
        def select_action(self, s):
            max_idx = np.where(np.abs(self.Q[s]-np.max(self.Q[s]))<1e-8)
            return np.random.choice(max_idx[0])  
        
        def update_Q(self, s, next_s, a, r, alpha, gamma):
            self.Q[s][a] = (1-alpha)*self.Q[s][a] + alpha*((1-gamma)*r+gamma*np.max(self.Q[next_s]))
            
    def __init__(self, nS, nA):
        self.agents = [self.agent(nS, na) for na in nA]
        
    def select_action(self, s):
        return tuple([agent.select_action(s) for agent in self.agents])
    
    def update(self, s, next_s, action, reward, done, alpha, gamma):
        
        for i, (a,r) in enumerate(zip(action, reward)):
            self.agents[i].update_Q(s, next_s, a, r, alpha, gamma)
    
    def log_value(self):
        # player A taking action S(4)
        # agent1.Q[s][4]
        return self.agents[0].Q[71][4]
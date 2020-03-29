import numpy as np

# class QLearner():
    
#     class agent():
#         def __init__(self, nS, nA):
#             self.nS = nS
#             self.nA = nA
#             self.Q = np.ones([nS, nA]) # nstate, my action
        
#         def select_action(self, s):
#             return np.argmax(self.Q[s])  
        
#         def update_Q(self, s, next_s, a, r, alpha, gamma):
#             self.Q[s][a] = (1-alpha)*self.Q[s][a] + alpha*(r+gamma*np.max(self.Q[next_s]))
            
#     def __init__(self, nS, nA):
#         self.agent1 = self.agent(nS, nA)
#         self.agent2 = self.agent(nS, nA)
        
#     def select_action(self, s):
#         return self.agent1.select_action(s), self.agent2.select_action(s)
    
#     def update(self, s, next_s, action, reward, done, alpha, gamma):
#         a1, a2 = action
#         r1, r2 = reward
        
#         self.agent1.update_Q(s, next_s, a1, r1, alpha, gamma)
#         self.agent2.update_Q(s, next_s, a2, r2, alpha, gamma)
        
#     def log_value(self):
#         return self.agent1.Q[71][4]
    
    
class QLearner():
    
    class agent():
        def __init__(self, nS, na):
            self.Q = np.ones([nS, na]) # nstate, my action
        
        def select_action(self, s):
            return np.argmax(self.Q[s])  
        
        def update_Q(self, s, next_s, a, r, alpha, gamma):
            self.Q[s][a] = (1-alpha)*self.Q[s][a] + alpha*(r+gamma*np.max(self.Q[next_s]))
            
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
        return self.agents[0].Q[71][4]+self.agents[0].Q[71][3]
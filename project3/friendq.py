import numpy as np
    
class FriendQ():
    class agent():
        def __init__(self, nS, nA, i):
            self.i = i
            self.Q = np.zeros([nS]+list(nA)) # nstate, player1 action, player2 action ...
        
        def select_action(self, s):
            max_idx = np.where(np.abs(self.Q[s]-np.max(self.Q[s]))<1e-8)
            return np.random.choice(max_idx[self.i])
        
        def update_Q(self, s, next_s, a, r, alpha, gamma):
            self.Q[s][a] = (1-alpha)*self.Q[s][a] + alpha*((1-gamma)*r+gamma*np.max(self.Q[next_s]))
            
    def __init__(self, nS, nA):
        self.agents = [self.agent(nS, nA, i) for i in range(len(nA))]
        
    def select_action(self, s):
        return tuple([agent.select_action(s) for agent in self.agents])
    
    def update(self, s, next_s, action, reward, done, alpha, gamma):
        for r, agent in zip(reward, self.agents):
            agent.update_Q(s, next_s, action, r, alpha, gamma)
        
    def log_value(self):
        # player A taking action S(4) and player B stick(0)
        return self.agents[0].Q[71][4][0]+self.agents[0].Q[71][4][3]
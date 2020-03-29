import numpy as np

# class FriendQ():
#     class agent():
#         def __init__(self, nS, nA):
#             self.nS = nS
#             self.nA = nA
#             self.Q = np.zeros([nS, nA, nA]) # nstate, other player action, my action
        
#         def select_action(self, s):
#             return np.argmax(self.Q[s]) % self.Q[s].shape[1]  
        
#         def update_Q(self, s, next_s, a1, a2, r, done, alpha, gamma):
# #             if done:
# #                 assert np.max(self.Q[next_s])==0, print(next_s, np.max(self.Q[next_s]))
# #                 self.Q[s][a1][a2] = (1-alpha)*self.Q[s][a1][a2] + alpha*r
# #             else:
#             self.Q[s][a1][a2] = (1-alpha)*self.Q[s][a1][a2] + alpha*(r+gamma*np.max(self.Q[next_s]))
            
#     def __init__(self, nS, nA):
#         self.agent1 = self.agent(nS, nA)
#         self.agent2 = self.agent(nS, nA)
        
#     def select_action(self, s):
#         return self.agent1.select_action(s), self.agent2.select_action(s)

    
#     def update(self, s, next_s, action, reward, done, alpha, gamma):
#         a1, a2 = action
#         r1, r2 = reward
        
#         self.agent1.update_Q(s, next_s, a2, a1, r1, done, alpha, gamma)
#         self.agent2.update_Q(s, next_s, a1, a2, r2, done, alpha, gamma)
        
#     def log_value(self):
#         return self.agent1.Q[71][0][4]
    
    
class FriendQ():
    class agent():
        def __init__(self, nS, nA, i):
            self.i = i
            self.Q = np.zeros([nS]+list(nA)) # nstate, player1 action, player2 action ...
        
        def select_action(self, s):
            max_idx = np.where(self.Q[s] == np.max(self.Q[s]))
            return max_idx[self.i][0]
        
        def update_Q(self, s, next_s, a, r, alpha, gamma):
            self.Q[s][a] = (1-alpha)*self.Q[s][a] + alpha*(r+gamma*np.max(self.Q[next_s]))
            
    def __init__(self, nS, nA):
        self.agents = [self.agent(nS, nA, i) for i in range(len(nA))]
        
    def select_action(self, s):
        return tuple([agent.select_action(s) for agent in self.agents])
    
    def update(self, s, next_s, action, reward, done, alpha, gamma):
        for r, agent in zip(reward, self.agents):
            agent.update_Q(s, next_s, action, r, alpha, gamma)
        
    def log_value(self):
        # player A taking action S(4) and player B stick(0)
        # agent1.Q[s][4][0] or agent2.Q[s][4][0]
        return self.agents[0].Q[71][0][4]
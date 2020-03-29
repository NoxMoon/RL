import numpy as np
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False
solvers.options['glpk'] = {'tm_lim' : 1000}

class FoeQ():
    
    class agent():
        def __init__(self, nS, na1, na2):
            self.nS = nS
            self.na1, self.na2 = na1, na2
            self.p = np.ones([nS, na1])/na1
            self.Q = np.ones([nS, na2, na1]) # nstate, other player action, my action
            self.V = np.zeros([nS])
        
        def select_action(self, s):
            return np.random.choice(range(len(self.p[s])), p=self.p[s])   
         
            
        def minmax(self, s, solver="glpk"):
            c = matrix(np.array([-1]+[0]*self.na1, dtype='float'))
            # constraints G*x <= h
            G = np.vstack([-self.Q[s], np.eye(self.na1) * -1]) # > 0 constraint for all vars
            new_col = [1 for i in range(self.na2)] + [0 for i in range(self.na1)]
            G = np.insert(G, 0, new_col, axis=1) # insert utility column
            G = matrix(G)
            #print("G",G)
            h = matrix(np.array([0 for i in range(self.na1+self.na2)], dtype='float'))
            #print("h",h)
            # contraints Ax = b
            A = matrix(np.matrix([0] + [1 for i in range(self.na2)], dtype="float"))
            b = matrix(np.matrix(1, dtype="float"))
            sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
            if not sol["x"]:
                print("fail in LP")
                return
            
            val, p = sol["x"][0], np.abs(np.array(sol["x"][1:])).flatten() #/np.sum(np.abs(sol["x"][1:]))
            p=p/sum(p)

            #print(val, p)
            self.p[s] = p
            self.V[s] = val
        
        def update_Q(self, s, next_s, a1, a2, r, alpha, gamma):
            self.Q[s][a1][a2] = (1-alpha)*self.Q[s][a1][a2] + alpha*(r+gamma*self.V[next_s])
            
    def __init__(self, nS, nA):
        assert len(nA)==2, "foeQ only works for two agent..."
        self.agent1 = self.agent(nS, nA[0], nA[1])
        self.agent2 = self.agent(nS, nA[1], nA[0])
        
    def select_action(self, s):
        return self.agent1.select_action(s), self.agent2.select_action(s)
    
    def update(self, s, next_s, action, reward, done, alpha, gamma):
        a1, a2 = action
        r1, r2 = reward
        
        self.agent1.update_Q(s, next_s, a2, a1, r1, alpha, gamma)
        self.agent1.minmax(s)
        
        self.agent2.update_Q(s, next_s, a1, a2, r2, alpha, gamma)
        self.agent2.minmax(s)
        
    def log_value(self):
        # player A taking action S(4) and player B stick(0)
        # agent1.Q[s][0][4] or agent2.Q[s][4][0]
        return self.agent1.Q[71][0][4]+self.agent1.Q[71][3][4]
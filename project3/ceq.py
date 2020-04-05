import numpy as np
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False
solvers.options['glpk'] = {'tm_lim' : 1000}



def put_by_axis(arr, val, inds, axis):
    """
        Assign value to arr at inds, axis=axis
        for example arr = array(3,4,5), val = array(3,5)
        put_by_axis(arr, val, ind=2, axis=1) will do arr[:,2,:] = val
    """ 
    sl = [slice(None)] * arr.ndim
    sl[axis] = inds
    #print(tuple(sl))
    arr[tuple(sl)] = val

class ceQ():
    class agent():
        def __init__(self, nS, nA, i):
            self.i = i
            self.nA = nA
            self.na = nA[i]
            #self.Q = np.random.sample([nS]+list(nA)) # nstate, player1 action, player2 action ...
            self.Q = np.ones([nS]+list(nA)) * (1-2*i) # for two-agent zero sum game we enforce Q1=-Q2, for mult-agent general sum game we can use the random initialization above
            self.V = np.zeros([nS])
            
        def get_rational_constraint(self, s):
            #print("agent",self.i)

            nc = self.na*(self.na-1) # number of rational constraint for each agent

            rational_c = np.zeros([nc]+list(self.nA))

            for a in range(self.na):
                Q0 = np.take(self.Q[s], a, axis=self.i) # Q0 = Q[:,:..a..:]
                Q0 = np.expand_dims(Q0, axis=self.i)
                constraint = np.delete(self.Q[s] - Q0, a, axis=self.i)# for agent i, action a, compute Q[:,:,...:]-Q[:,:..a..:], only for a'!=a, a' in action of agent i.
                constraint = np.moveaxis(constraint, self.i, 0) # reshape to [(na_i-1),na_1, .. na_i-1,na_i+1, .. na_N]
                put_by_axis(rational_c[(self.na-1)*a:(self.na-1)*(a+1)], constraint, a, axis=self.i+1)
            
            rational_c = rational_c.reshape(nc, -1) #reshape to [nc, prod(na_i)]
            return rational_c
        
        def update_Q(self, s, next_s, a, r, done, alpha, gamma):
            self.Q[s][a] = (1-alpha)*self.Q[s][a] + alpha*((1-gamma)*r+gamma*self.V[next_s])
            
    def __init__(self, nS, nA):
        self.nA = nA
        self.NA = np.prod(self.nA)
        self.agents = [self.agent(nS, nA, i) for i in range(len(nA))]
        self.p = np.ones([nS, self.NA])/self.NA
        
    def solve(self, s, solver="coneqp"):
        G = np.vstack([agent.get_rational_constraint(s) for agent in self.agents])
        G = np.vstack([G, -np.eye(self.NA)])
        h = np.array([0.0]*G.shape[0])
        c = -sum(agent.Q[s] for agent in self.agents).flatten()
        G, h, c = matrix(G), matrix(h), matrix(c)
        A = matrix(np.ones((1, self.NA)))
        b = matrix(1.0)
        
        sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
        if not sol["x"]:
            print("fail in LP")
            return
            
        self.p[s] = np.abs(np.array(sol["x"])).flatten()
        self.p[s] = self.p[s]/sum(self.p[s])
            
        for agent in self.agents:
            agent.V[s] = np.dot(agent.Q[s].flatten(), self.p[s])

            
    def select_action(self, s):
        idx = np.random.choice(range(len(self.p[s])), p=self.p[s])
        return np.unravel_index(idx, self.nA)
    
    def update(self, s, next_s, action, reward, done, alpha, gamma):
        for r, agent in zip(reward, self.agents):
            agent.update_Q(s, next_s, action, r, done, alpha, gamma)
            
        self.solve(s)
        
    def log_value(self):
        # player A taking action S(4) and player B stick(0)
        return self.agents[0].Q[71][4][0]+self.agents[0].Q[71][4][3]
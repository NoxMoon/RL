import numpy as np

class player():
    def __init__(self, pos, hasball):
        self.pos = pos
        self.hasball = hasball

class soccer_game():
    def __init__(self):
        # 0 1 2 3
        # 4 5 6 7
        self.nA = (5,5)
        self.nS = 112
        self.state_idx = {}
        k = 0
        for i in range(8):
            for j in range(8):
                if i==j:
                    continue
                self.state_idx[(i,j,True)] = k
                self.state_idx[(i,j,False)] = k+56
                k = k+1
        
        self.reset()
        
    def reset(self):
        self.player1 = player(2, False)
        self.player2 = player(1, True)
        self.total_step = 0
        return self.getstate()
        
    def getstate(self):
        return self.state_idx[(self.player1.pos, self.player2.pos, self.player1.hasball)]
        
    def move(self, pos, a):
        if a==1 and pos not in (0,4): # Left
            return pos-1
        elif a==2 and pos not in (3,7): # right
            return pos+1
        elif a==3 and pos>=4: #up
            return pos-4
        elif a==4 and pos<=3: #down
            return pos+4
        else: # stick or hit the boundry
            return pos
        
    def execute_action(self, player1, player2, a1, a2):
        
        new_pos1 = self.move(player1.pos, a1)
        
        if new_pos1 != player2.pos:
            player1.pos = new_pos1
        elif player1.hasball:
            player1.hasball = False
            player2.hasball = True
            
        new_pos2 = self.move(player2.pos, a2)
        
        if new_pos2 != player1.pos:
            player2.pos = new_pos2
        elif player2.hasball:
            player2.hasball = False
            player1.hasball = True      
        
    def step(self, action):
        
        self.total_step += 1
        
        if np.random.randint(2)==0:
            self.execute_action(self.player1, self.player2, action[0], action[1])
        else:
            self.execute_action(self.player2, self.player1, action[1], action[0])
            
        reward, done, info = (0,0), False, ""
        
        if self.player1.hasball and self.player1.pos in (0,4):
            reward = (100, -100)
            done = True
        if self.player1.hasball and self.player1.pos in (3,7):
            reward = (-100, 100)
            done = True
        if self.player2.hasball and self.player2.pos in (0,4):
            reward = (100, -100)
            done = True
        if self.player2.hasball and self.player2.pos in (3,7):
            reward = (-100, 100)
            done = True
            
        if self.total_step>200:
            done = True
            info = "episode too long"
            
        return self.getstate(), reward, done, info
    
    def random_action(self):
        return np.random.randint(self.nA[0]), np.random.randint(self.nA[1])
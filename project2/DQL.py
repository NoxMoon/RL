import numpy as np
import gym
import random
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.models import load_model
import json

#env = gym.make('LunarLander-v2')


def Q_model(state_dim, n_action, lr=0.001):
    model = Sequential()

    model.add(Dense(64, input_dim=state_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_action, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model


        
class DQL_agent():
    def __init__(self, dS, nA, batch_size, gamma, beta, epsilon, epsilon_decay, lr, max_episodes=10000, min_episodes=1000, epochs=1):
        self.dS = dS
        self.nA = nA
        self.max_episodes = max_episodes
        self.min_episodes = min_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.beta = beta
        #self.update_freq = update_freq
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.curr_epsilon = epsilon
        self.epochs = epochs
        
        self.model = Q_model(dS, nA, lr)
        self.target_model = Q_model(dS, nA, lr)  
        self.memory = deque(maxlen=self.batch_size * 100)
        self.episode_rewards = []
        
        self.copy_weight()
        
    def epsilon_greedy(self, s, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(0, self.nA)
        else:
            return np.argmax(self.model.predict(np.expand_dims(s, axis=0))[0])
        
    def update(self):
        batch = np.array(random.sample(self.memory, self.batch_size))
        s, next_s = np.vstack(batch[:,0]), np.vstack(batch[:,3])
        a, r, done = np.array(batch[:,1], dtype='int'), np.array(batch[:,2], dtype='float'), np.array(batch[:,4], dtype='bool')
        Q = self.model.predict(s) #current prediction of Q values (n_sample * n_action)
        new_Q = r #target prediction of Q values
        not_done_idx = np.where(~done)[0]
        new_Q[not_done_idx] += self.gamma * np.max(self.target_model.predict(next_s[not_done_idx, :]), axis=1)
        Q[range(Q.shape[0]), a] = new_Q #update Q of a to new_Q
        self.model.fit(x=s, y=Q, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        
    def update_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.beta * weights[i] + (1 - self.beta) * target_weights[i]
        self.target_model.set_weights(target_weights)
        
    def copy_weight(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def Q_learning(self, env, warm_start=False):

        i = 0
        if not warm_start:
            self.episode_rewards = []
            self.curr_epsilon = self.epsilon

        for n in range(self.max_episodes):
            s = env.reset()
            
            done = False
            episode_reward = 0
            while not done:
                a = self.epsilon_greedy(s, self.curr_epsilon)
                next_s, r, done, info = env.step(a)
                i += 1
                self.memory.append([s, int(a), r, next_s, done])
                if i>=self.batch_size:
                    self.update()
                    self.update_target()
                #if i%(self.update_freq)==0:
                #   self.copy_weight()
                    
                s = next_s
                episode_reward += r
                
            self.episode_rewards.append(episode_reward)
            self.curr_epsilon *= self.epsilon_decay
            print(f"\repisode {n}, reward {np.mean(self.episode_rewards[-100:])}", end="")
            if n%100 == 0:
                if np.mean(self.episode_rewards[-100:]) > 200 and n>=self.min_episodes:
                    print("average rewards reached 200 in last 100 episodes")
                    return
                print("")
                
    def save_model(self, f):
        self.model.save(f)
        
    def load_model(self, f):
        self.model = load_model(f)
        self.copy_weight()
        
        
def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n

def run_experiment(params, f):
    results = {}
    
    env = gym.make('LunarLander-v2')
    dS = env.observation_space.shape[0]
    nA = env.action_space.n
    agent = DQL_agent(dS, nA, 
                      batch_size = params['batch_size'], 
                      gamma = params['gamma'], 
                      beta = params['beta'],
                      epsilon = params['epsilon'], 
                      epsilon_decay = params['epsilon_decay'],
                      lr = params['lr']
                     )

    agent.Q_learning(env)
    
    y = moving_average(agent.episode_rewards, n=100)
    results['params'] = params
    results['rewards'] = agent.episode_rewards
    results['moving_avg'] = y
    
    json.dump(result, open(f,'w'))
import numpy as np
import gym
import random
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU
from keras.optimizers import Adam
from keras.models import load_model
import tensorflow as tf
import json
import os, sys

#env = gym.make('LunarLander-v2')


def Q_model(state_dim, n_action, lr):
    model = Sequential()

    model.add(Dense(64, input_dim=state_dim))
    model.add(LeakyReLU())
    model.add(Dense(32))
    model.add(LeakyReLU())
    model.add(Dense(n_action, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model


        
class DQL_agent():
    def __init__(self, dS, nA, batch_size, gamma, tau, epsilon, epsilon_decay, lr, max_episodes=1000, dumpfile=None, epochs=1):
        self.dS = dS
        self.nA = nA
        self.max_episodes = max_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        #self.update_freq = update_freq
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.curr_epsilon = epsilon
        self.epochs = epochs
        self.dumpfile = dumpfile
        
        self.model = Q_model(dS, nA, lr)
        self.target_model = Q_model(dS, nA, lr)  
        self.memory = deque(maxlen=batch_size*300)
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
        target_Q = r #target prediction of Q values
        not_done_idx = np.where(~done)[0]
        
        #target_Q[not_done_idx] += self.gamma * np.max(self.target_model.predict(next_s[not_done_idx, :]), axis=1)
        
        next_a = np.argmax(self.model.predict(next_s[not_done_idx]), axis=1)
        target_Q[not_done_idx] += self.gamma * self.target_model.predict(next_s)[not_done_idx, next_a]
        
        Q[range(self.batch_size), a] = target_Q  # for action=a, set Q to target_Q
        
        self.model.fit(x=s, y=Q, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        
    def update_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
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
            avg_reward = np.mean(self.episode_rewards[-100:])
            print(f"\repisode {n}, reward {avg_reward}", end="")
            if n%100 == 0:
                print("")
                sys.stdout.flush()
            if avg_reward > 200 and self.dumpfile!=None:
                self.save_model(f"{self.dumpfile}_ep{n}.h5")
                
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
    
    np.random.seed(123)
    env.seed(123)
    tf.random.set_seed(123)

    agent = DQL_agent(dS, nA, 
                      batch_size = params['batch_size'], 
                      gamma = params['gamma'], 
                      tau = params['tau'],
                      epsilon = params['epsilon'], 
                      epsilon_decay = params['epsilon_decay'],
                      lr = params['lr']
                     )

    agent.Q_learning(env)
    
    y = moving_average(agent.episode_rewards, n=100)
    results['params'] = params
    results['episode_rewards'] = agent.episode_rewards
    results['moving_avg'] = list(y)
    
    json.dump(results, open(f,'w'))
    
    
def test_agent(agent, env):
    rewards = []

    for _ in range(100):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.epsilon_greedy(s, 0)
            next_s, r, done, _ = env.step(a)
            s = next_s
            episode_reward += r
        rewards.append(episode_reward)

    return np.mean(rewards), np.std(rewards), rewards

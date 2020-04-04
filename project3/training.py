import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def epsilon_greedy(learner, env, s, epsilon):
    if np.random.rand()<epsilon:
        return env.random_action()
    else:
        return learner.select_action(s)
    
def learning(learner, env, 
             epsilon=1, epsilon_decay=0.9995, epsilon_min=0.1, 
             alpha=1, alpha_decay=0.9995, alpha_min=0.001, 
             gamma = 0.9,
             max_episode=10000):
    
    np.random.seed(10)
    
    hist = []
    for i in tqdm(range(max_episode)):
        s = env.reset()
        done = False
        while not done:
            action = epsilon_greedy(learner, env, s, epsilon)
            next_s, r, done, info = env.step(action)
            learner.update(s, next_s, action, r, done, alpha, gamma)
            s = next_s
            
        epsilon = max(epsilon*epsilon_decay, epsilon_min)
        alpha = max(alpha*alpha_decay, alpha_min)
        
        hist.append(learner.log_value())
        
    return hist


def learning_2005(learner, env, 
             epsilon=1, 
             gamma = 0.9,
             max_episode=10000):
# following Greenwald 2005 paper LR decaying schedule: 
# decay by 1/n(s,a) n(s,a) is # of visit to state-action pair

    np.random.seed(10)
    
    a = np.ones([env.nS]+list(env.nA))
    c = np.zeros([env.nS]+list(env.nA))
    
    hist = []
    for i in tqdm(range(max_episode)):
        s = env.reset()
        done = False
        while not done:
            action = epsilon_greedy(learner, env, s, epsilon)
            next_s, r, done, info = env.step(action)
            learner.update(s, next_s, action, r, done, a[s][action], gamma)
            
            c[s][action] += 1
            a[s][action] = 1/c[s][action]
         
            s = next_s
        
        hist.append(learner.log_value())
        
    return hist

def plot_error(hist, f):
    errors = np.abs(np.array(hist[1:])-np.array(hist[:-1]))
    plt.figure(figsize=(7,5))
    plt.plot(errors)
    plt.ylim(0,0.5)
    plt.xlabel("episode")
    plt.ylabel("Q value change")
    plt.savefig(f)
    plt.show()
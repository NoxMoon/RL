from DQL import *
#env = gym.make('LunarLander-v2')

for i in range(1,4):
    params = {
        'batch_size': 32,
        'gamma': 1,
        'tau': 0.001,
        'epsilon': 1,
        'epsilon_decay': 0.995,
        'lr': 0.0005,
    }
    print("\n",params,"\n")
    run_experiment(params, f'experiments/lr0.0005_e0.995_tau0.001_gamma1_{i}.txt')

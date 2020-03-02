from DQL import *
#env = gym.make('LunarLander-v2')

for tau in [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
    params = {
        'batch_size': 32,
        'gamma': 0.99,
        'tau': tau,
        'epsilon': 1,
        'epsilon_decay': 0.995,
        'lr': 0.0005,
    }
    print("\n",params,"\n")
    run_experiment(params, f'experiments/tau_{tau}_2.txt')

from DQL import *
#env = gym.make('LunarLander-v2')

for epsilon_decay in [0.999, 0.995, 0.99, 0.95]:
    params = {
        'batch_size': 32,
        'gamma': 0.99,
        'tau': 0.001,
        'epsilon': 1,
        'epsilon_decay': epsilon_decay,
        'lr': 0.0005,
    }
    print("\n",params,"\n")
    run_experiment(params, f'experiments/epsilon_decay_{epsilon_decay}.txt')
from DQL import *
#env = gym.make('LunarLander-v2')

for lr in [0.005, 0.001, 0.0005, 0.0001]:
    params = {
        'batch_size': 32,
        'gamma': 0.99,
        'tau': 0.001,
        'epsilon': 1,
        'epsilon_decay': 0.995,
        'lr': lr,
    }
    print("\n",params,"\n")
    run_experiment(params, f'experiments/lr_{lr}.txt')

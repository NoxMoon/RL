I use Keras/Tensorflow for building/training Neural network, Opengym for RL environment. Please see Pipfile (under main directory) for detailed environment.

Tha main DDQL algorithm is implemented in DQL.py.

In sample_run.ipynb, I have one agent trained to achieve +200 test rewards. You can visualize the agent behavior in visualization.ipynb.

param_lr.py, param_gamma.py, param_edecay.py, param_tau.py generate multiple runs to study the effect of hyperparameters, the results are collected in hyperparameter.ipynb.

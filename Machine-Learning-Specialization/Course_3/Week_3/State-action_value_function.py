import numpy as np
from Package_1 import utils

# Do not modify
num_states = 6
num_actions = 2

terminal_left_reward = 100
terminal_right_reward = 40
each_step_reward = 0

# Discount factor
gamma = 0.9

# Probability of going in the wrong direction
misstep_prob = 0.2

utils.generate_visualization(terminal_left_reward, terminal_right_reward, each_step_reward, gamma, misstep_prob)

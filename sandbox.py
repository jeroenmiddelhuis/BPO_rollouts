import rollouts
import smdp
import mdp
import policy_learner
import rollouts
from crn import CRN
from tqdm import tqdm
import os
import pickle
import numpy as np
import smdp_composite
import mdp_composite
from heuristic_policies import random_policy, greedy_policy
from collections import Counter

if __name__ == '__main__':
    env = smdp_composite.SMDP_composite(2500, 'composite', track_cycle_times=False)
    states = rollouts.random_states(env, random_policy, 5000)
    state_actions = []
    means = []
    for i, state in enumerate(states):
        print(i)
        env.set_state(state)
        state_action, mean = rollouts.find_learning_sample(env, greedy_policy, nr_rollouts_per_action=100, 
                                                            nr_steps_per_rollout=100, only_statistically_significant=False)
        state_actions.append(state_action)
        means.append(mean)

    # Save the state_actions list to a file
    with open('./data/state_actions_composite.pkl', 'wb') as f:
        pickle.dump(state_actions, f)

    print("state_actions list has been saved to 'state_actions_composite.pkl'")
    
    # Save the means to disk
    with open('./data/means_composite.pkl', 'wb') as f:
        pickle.dump(means, f)



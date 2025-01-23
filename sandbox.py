import rollouts
import smdp
import mdp
import policy_learner
import rollouts
from crn import CRN
from tqdm import tqdm
import os
import pickle

if __name__ == '__main__':

    average_step_time_smdp = {
        'slow_server': 0.6700290812922858,
        'low_utilization': 0.6667579762550317,
        'high_utilization': 0.6695359811479276,
        'n_system': 1.0013236383088864,
        'parallel': 0.6680392125284501,
        'down_stream': 0.6681612256898539,
        'single_activity': 1.0042253394472318
    }

    config_type = 'slow_server'
    nr_steps_per_rollout = 50
    tau = average_step_time_smdp[config_type] * 0.5
    mdp_steps = int(nr_steps_per_rollout * average_step_time_smdp[config_type] / tau)

    env = mdp.MDP(2500, config_type='slow_server', tau=tau)
    greedy_policy = smdp.greedy_policy

    learning_samples_X, learning_samples_y = rollouts.learn_iteration(env, 
                                                                      greedy_policy, 
                                                                      nr_states_to_explore=5000, 
                                                                      nr_rollouts_per_action_per_state=100,
                                                                      nr_steps_per_rollout=mdp_steps, 
                                                                      only_statistically_significant=True,
                                                                      return_learning_samples=True)

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Save learning samples
    with open('data/mdp_learning_samples.pkl', 'wb') as f:
        pickle.dump({
            'X': learning_samples_X,
            'y': learning_samples_y
        }, f)





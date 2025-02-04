from collections import deque
from subprocess import call
import gymnasium as gym
import os
import numpy as np
from gym_env import Environment
import sys
import smdp, mdp
import rollouts
import policy_learner

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from callbacks import SaveOnBestTrainingRewardCallback, EvalPolicyCallback
from callbacks import custom_schedule, linear_schedule

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Input parameters
nr_layers = 2
nr_neurons = 128
clip_range = 0.2
batch_size = 256
lr = 3e-05

n_steps = 25600 # Number of steps per update
time_steps = 1e7 # Total timesteps for training
#config_type = ['n_system', 'slow_server', 'low_utilization', 'high_utilization', 'parallel', 'down_stream', 'single_activity']
config_type = sys.argv[1] if len(sys.argv) > 1 else 'slow_server'
env_type = sys.argv[2] if len(sys.argv) > 2 else 'smdp'
test_mode = sys.argv[3] if len(sys.argv) > 3 else False
reward_function = sys.argv[4] if len(sys.argv) > 4 else 'inverse_case_cycle_time'

net_arch = dict(pi=[nr_neurons for _ in range(nr_layers)], vf=[nr_neurons for _ in range(nr_layers)])

class CustomPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=net_arch)

def evaluate_policy(filename, config_type, episode_length=10, nr_rollouts=100, results_dir=None, env_type='smdp'):
    env = smdp.SMDP(episode_length, config_type)
    pl = policy_learner.PolicyLearner.load(filename)

    rewards = rollouts.evaluate_policy(env, pl.policy, nr_rollouts, ppo=True)
    average_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    if not os.path.exists(results_dir):
        with open(results_dir, 'w') as results_file:
            results_file.write("mean,std\n")
    # Write average reward and standard deviation to file
    with open(results_dir, 'a') as results_file:
        results_file.write(f"{average_reward},{std_reward}\n")

    print(average_reward, std_reward)  

if __name__ == '__main__':
    if test_mode == False or test_mode == "False":
        # Create log dir
        log_dir = f"./models/ppo_test/{env_type}/{config_type}/{reward_function}/" # Logging training results
        os.makedirs(log_dir, exist_ok=True)

        average_step_time_smdp = {
            'slow_server': 0.6700290812922858,
            'low_utilization': 0.6667579762550317,
            'high_utilization': 0.6695359811479276,
            'n_system': 1.0014597219587165,
            'parallel': 0.6680392125284501,
            'down_stream': 0.6681612256898539,
            'single_activity': 1.0042253394472318
        }
        tau = average_step_time_smdp[config_type] / 2

        if env_type == 'mdp':
            env = mdp.MDP(2500, config_type, tau, reward_function=reward_function)
        elif env_type == 'smdp':
            env = smdp.SMDP(2500, config_type, reward_function=reward_function)

        print(f'Training agent for {config_type} with {time_steps} timesteps and reward function {reward_function} in updates of {n_steps} steps.')
        
        # Training environment
        # Create and wrap the environment
        gym_env = Environment(env)  # Initialize env
        gym_env = Monitor(gym_env, log_dir)

        if reward_function == 'inverse_case_cycle_time':
            gamma = 0.999
        else:
            gamma = 1
            
        # Create the model
        model = MaskablePPO(CustomPolicy, 
                            gym_env, 
                            clip_range=clip_range, 
                            learning_rate=linear_schedule(lr), 
                            n_steps=int(n_steps), 
                            batch_size=batch_size, 
                            gamma=gamma, 
                            verbose=1) #

        #Logging to tensorboard. To access tensorboard, open a bash terminal in the projects directory, activate the environment (where tensorflow should be installed) and run the command in the following line
        # tensorboard --logdir ./tmp/
        # then, in a browser page, access localhost:6006 to see the board
        model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

        # Evaluation environment
        if env_type == 'mdp':
            eval_env = mdp.MDP(2500, config_type, tau)
        elif env_type == 'smdp':
            eval_env = smdp.SMDP(2500, config_type)
            
        gym_env_eval = Environment(eval_env)  # Initialize env
        gym_env_eval = Monitor(gym_env_eval, log_dir)
        eval_callback = EvalPolicyCallback(check_freq=10*int(n_steps), nr_evaluations=10, log_dir=log_dir, eval_env=gym_env_eval)
        best_reward_callback = SaveOnBestTrainingRewardCallback(check_freq=int(n_steps), log_dir=log_dir)

        model.learn(total_timesteps=int(time_steps), callback=eval_callback)#

        # For episode rewards, use env.get_episode_rewards()®
        # env.get_episode_times() returns the wall clock time in seconds of each episode (since start)
        # env.rewards returns a list of ALL rewards. Of current episode?
        # env.episode_lengths returns the number of timesteps per episode
        # print(env.get_episode_rewards())
        #     print(env.get_episode_times())

        model.save(f'{log_dir}/final_model')

        #import matplotlib.pyplot as plt
        #plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, f"{model_name}")
        #plt.show()
    else:
        """
        Evaluation of the learned policies
        """

        print(config_type, env_type)

        filename = f"./models/ppo//{env_type}/{config_type}/{reward_function}/best_model.zip"
        results_dir = f"./results/ppo_{env_type}_{reward_function}_{config_type}.txt"

        evaluate_policy(filename, config_type, episode_length=2500, nr_rollouts=300, 
                        results_dir=results_dir, env_type=env_type)
        print('\n')        

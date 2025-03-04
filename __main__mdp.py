import mdp, mdp_composite, smdp, smdp_composite
import rollouts
import policy_learner
import sys, re
import numpy as np
import itertools
import os
from crn import CRN
import torch
from heuristic_policies import fifo_policy, random_policy, greedy_policy, threshold_policy

def learn(config_type, 
          bootstrap_policy, 
          filename_without_extension, 
          learning_iterations=3, 
          episode_length=10, 
          nr_states_to_explore=1000, 
          nr_rollouts=100, 
          nr_steps_per_rollout=100,
          model_type = 'neural_network', 
          only_statistically_significant=False,
          tau=0.5):
    if config_type == 'composite':
        env = mdp_composite.MDP_composite(episode_length, config_type, tau=tau, crn=CRN(), track_cycle_times=False)
        evaluation_env = smdp_composite.SMDP_composite(episode_length, config_type)
    else:
        env = mdp.MDP(episode_length, config_type, tau=tau, crn=CRN(), track_cycle_times=False)  
        evaluation_env = smdp.SMDP(episode_length, config_type)  
    extension = '.pth'
    pl = rollouts.learn_iteration(env, bootstrap_policy, nr_states_to_explore, nr_rollouts, nr_steps_per_rollout, model_type=model_type)
    pl.save(filename_without_extension + config_type + ".v1" + extension)
    pl.save(filename_without_extension + config_type + ".best_policy" + extension)  
    pl = policy_learner.PolicyLearner.load(filename_without_extension + config_type + ".v1" + extension)
    pl.cache = fill_cache(env, pl)

    print('Evaluating policy 1..')
    rewards, _ = rollouts.evaluate_policy(evaluation_env, pl.policy, nr_rollouts=100, nr_arrivals=2500, parallel=False)
    best_policy_reward = np.mean(rewards)
    print('Reward of policy version 1:', best_policy_reward)
    best_policy_v = 1
    print('\n')
    for i in range(2, learning_iterations+1):
        pl = rollouts.learn_iteration(env, pl.policy, nr_states_to_explore, nr_rollouts, nr_steps_per_rollout, pl)
        pl.save(filename_without_extension + config_type + ".v" + str(i) + extension)
        pl = policy_learner.PolicyLearner.load(filename_without_extension + config_type + ".v" + str(i) + extension)
        pl.cache = fill_cache(env, pl)
        
        print(f'Evaluating new policy {i}..')
        rewards, _ = rollouts.evaluate_policy(evaluation_env, pl.policy, nr_rollouts=100, nr_arrivals=2500, parallel=False)
        new_policy_reward = np.mean(rewards)
        print('Reward of policy version', i, ':', new_policy_reward)
        print(f"Reward of best policy, version {best_policy_v}: {best_policy_reward}. Reward of new policy, version {i}: {new_policy_reward}.")
        if new_policy_reward < best_policy_reward: # New policy is not better than the previous policy
            print("Reward of policy version", i, "is worse than previous policy, reverting to previous policy.")
            pl = policy_learner.PolicyLearner.load(filename_without_extension + config_type + ".v" + str(best_policy_v) + extension)
            pl.cache = fill_cache(env, pl)
        else:
            print(f"Policy version {best_policy_v} improved, continuing with policy version {i}.")
            best_policy_reward = new_policy_reward
            best_policy_v = i
            pl = policy_learner.PolicyLearner.load(filename_without_extension + config_type + ".v" + str(best_policy_v) + extension)
            pl.cache = fill_cache(env, pl)
        print('\n')

    pl = policy_learner.PolicyLearner.load(filename_without_extension + config_type + ".v" + str(best_policy_v) + extension)
    pl.save(filename_without_extension + config_type + ".best_policy" + extension)            


def fill_cache(env, policy):
    if env.config_type != 'composite':
        _, frequent_states, _ =  determine_state_space(env)
        frequent_states = torch.tensor(np.array([policy.normalize_observation(np.array(state, dtype=float)) for state in frequent_states]), dtype=torch.float32)
        probabilities = policy.model(frequent_states).detach()
        cache = {tuple(state.detach().numpy()): _probabilities.detach().numpy() for state, _probabilities in zip(frequent_states, probabilities)}
        return cache
    else:
        return {}

def determine_state_space(env, max_queue=30):
    def check_feasible_observation(observation):
        assert len(observation) == len(env.state_space), f"Observation ({len(observation)}) and state space ({len(env.state_space)}) length must match"

        for resource in env.resources:
            # Some features do not contain assigned_ features if they can only be assigned to one task
            for state in env.state_space:
                if f'assigned_{resource}' in state:
                    assert sum([1 for i, x in enumerate(observation) if x == 1 and resource in env.state_space[i]]) == 1

        # All values should be positive
        assert all([x >= 0 for x in observation])

        # The values not related to the queue should be 0 or 1
        for i, x in enumerate(observation):
            if 'waiting' not in env.state_space[i]:
                assert x == 0 or x == 1

    feature_ranges = []
    for state_label in env.state_space:
        if 'is_available_' in state_label:
            feature_ranges.append(2)
        elif 'assigned_' in state_label:
            feature_ranges.append(2)
        elif 'waiting_' in state_label:
            feature_ranges.append(max_queue + 1)
    all_states = [list(state) for state in itertools.product(*[range(i) for i in feature_ranges])]

    state_space = []
    for observation in all_states:
        try:
            check_feasible_observation(observation)
            state_space.append(observation)                
        except AssertionError:
            continue
    return feature_ranges, state_space, all_states

def show_policy(filename):
    pl = policy_learner.PolicyLearner.load(filename)
    
    print("If r1 is available, the logical thing is always assign r1")
    for waiting in range(1, 10):
        observation = (1, 1, 0, 0, waiting)
        print(observation, pl.predict(observation, [True, True, True, False]))

    print("If r2 is available, but r1 is not, the logical thing is to postpone up to a certain number of waiting cases and assign r2 after")
    for waiting in range(1, 10):
        observation = (0, 1, 1, 0, waiting)
        print(observation, pl.predict(observation, [False, True, True, False]))
    
def evaluate_policy(filename, config_type, episode_length=10, nr_rollouts=100, results_dir=None, model_type=None):
    if config_type == 'composite':
        env = mdp_composite.MDP_composite(episode_length, config_type)
    else:
        env = mdp.MDP(episode_length, config_type)
    pl = policy_learner.PolicyLearner.load(filename, model_type)
    version_number = re.search(r'v(\d+)', filename).group(1)
    average_reward = rollouts.evaluate_policy(env, pl.policy, nr_rollouts)
    
    # Prepare results directory and file
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        results_file_path = os.path.join(results_dir, f'results_{model_type}_{model_type}.txt')
    else:
        results_file_path = f'results_{model_type}.txt'
    
    if version_number != '1':
        with open(results_file_path, 'a') as results_file:
            results_file.write(f"{model_type} v{version_number}, {average_reward}\n")
            print(f"Learned {model_type} policy:", average_reward)
    else:
        with open(results_file_path, 'w') as results_file:
            results_file.write(f"{model_type} v{version_number}, {average_reward}\n")
            print("Learned policy:", average_reward)
            
            if config_type == 'single_activity':
                average_reward = rollouts.evaluate_policy(env, mdp.threshold_policy, nr_rollouts)
                results_file.write(f"Threshold, {average_reward}\n")
                print("Threshold policy:", average_reward)
            else:
                average_reward = rollouts.evaluate_policy(env, mdp.greedy_policy, nr_rollouts)
                results_file.write(f"Greedy, {average_reward}\n")
                print('Greedy:', average_reward)
                
                average_reward = rollouts.evaluate_policy(env, mdp.random_policy, nr_rollouts)
                results_file.write(f"Random, {average_reward}\n")
                print('Random policy:', average_reward)

                average_reward = rollouts.evaluate_policy(env, mdp.fifo_policy, nr_rollouts)
                results_file.write(f"FIFO, {average_reward}\n")
                print('FIFO policy:', average_reward)


def compare_all_states(filename, episode_length=10, nr_rollouts=100):
    """
    Gets all observations by doing rollouts and storing all states.
    Apply the policy from the file as well as the threshold policy to each state, 
    print the observation and resulting actions for comparison. 
    """
    env = mdp.MDP(episode_length)
    pl = policy_learner.PolicyLearner.load(filename)
    states = set()
    for _ in range(nr_rollouts):
        env.reset()
        _, rollout = rollouts.rollout_with_full_information(env, mdp.threshold_policy)
        for (observation, _, mask, _) in rollout:
            states.add((tuple(observation), tuple(mask)))
    for s in states:
        observation, mask = s
        action_policy = pl.predict(observation, mask)
        action_threshold = mdp.threshold_policy(None, observation, mask)
        if action_policy != action_threshold:
            print(observation, mask, action_policy, action_threshold)

def main():
    """
    Test random states
    """
    # env = mdp.MDP(2500, 'high_utilization')
    # states = rollouts.random_states(env, mdp.random_policy, 5000)
    # observations = []
    # for state in states:
    #     env.set_state(state)
    #     observations.append(env.observation())

    # state_counts = Counter(tuple(obs) for obs in observations)

    # print("Unique states and their counts (sorted by count):")
    # for state, count in sorted(state_counts.items(), key=lambda x: x[1], reverse=False):
    #     print(state, count)

    """
    Test of rollout
    """

    # env = mdp.MDP(3000, 'n_system')
    # reward = rollouts.evaluate_policy(env, mdp.fifo_policy, nr_rollouts=100)
    # print(reward)

    """
    Training of the policies
    """

    # average_rewards = rollouts.evaluate_policy(mdp.MDP(3000, 'single_activity'), mdp.greedy_policy, 100)
    # print(sum(average_rewards) / len(average_rewards))
    
    average_step_time_smdp = {
        'slow_server': 0.6700290812922858,
        'low_utilization': 0.6667579762550317,
        'high_utilization': 0.6695359811479276,
        'n_system': 1.0013236383088864,
        'parallel': 0.6680392125284501,
        'down_stream': 0.6681612256898539,
        'single_activity': 1.0042253394472318,
        'composite': 0.1702603617565683
    }

    minimium_transition_time = {
        'slow_server': 1 / (1/2.0 + 1/1.4 + 1/1.8),
        'low_utilization': 1 / (1/2.0 + 1/1.4 + 1/1.4),
        'high_utilization': 1 / (1/2.0 + 1/1.8 + 1/1.8),
        'n_system': 1 / (1/2.0 + 1/2.0 + 1/3.0),
        'parallel': 1 / (1/2.0 + 1/1.6 + 1/1.6),
        'down_stream': 1 / (1/2.0 + 1/1.6 + 1/1.6),
        'single_activity': 1/ (1/2.0 + 1/1.8 + 1/10.0),
        'composite': 1 / (1/2.0 + 1/1.4 + 1/1.8 
                                + 1/1.4 + 1/1.4 
                                + 1/1.8 + 1/1.8
                                + 1/2.0 + 1/3.0
                                + 1/1.6 + 1/1.6 
                                + 1/1.6 + 1/1.6)
    }
    print(minimium_transition_time)
    nr_rollouts = 100
    nr_steps_per_rollout = 100
    config_type = sys.argv[1] if len(sys.argv) > 1 else 'composite'
    model_type = 'neural_network'
    learning_iterations = 3

    tau_multiplier = 0.5
    tau = minimium_transition_time[config_type] * tau_multiplier
    mdp_steps = int(np.ceil((nr_steps_per_rollout * average_step_time_smdp[config_type] / tau)))

    dir = f".//models//pi//mdp//{config_type}//"
    #dir = f".//models//pi//tuning//{nr_rollouts}_{nr_steps_per_rollout}_{tau_multiplier}//"
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    print('Learning MDP policy for', config_type, 'with', model_type, 'model.', flush=True)
    print('Tau:', tau, f'(multiplier = {tau_multiplier})', flush=True)
    print('Nr rollouts:', nr_rollouts, flush=True)
    print('Rollout length:', mdp_steps, f'(smdp steps = {nr_steps_per_rollout})', flush=True)

    learn(config_type, 
        greedy_policy,
        dir,
        learning_iterations=learning_iterations,
        episode_length=2500, # nr_cases
        nr_states_to_explore=20000,
        nr_rollouts=nr_rollouts,
        nr_steps_per_rollout=mdp_steps,
        model_type=model_type,
        tau=tau)
    
    """
    Evaluation of the learned policies
    """

    # for model_type in ['neural_network', 'xgboost']:
    #     pass
    # model_type = 'neural_network'
    # extension = 'keras' if model_type == 'neural_network' else 'xgb.json'
    # env_type = 'mdp'
    # results_dir = f".//models/smdp//{config_type}//"

    # filename = f".//models//{env_type}//{config_type}//{config_type}.v1.{extension}"
    # evaluate_policy(filename, config_type, episode_length=3000, nr_rollouts=100, results_dir=results_dir, model_type=model_type)

if __name__ == '__main__':
    main()
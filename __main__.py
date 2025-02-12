import smdp
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
          only_statistically_significant=False):
    
    env = smdp.SMDP(episode_length, config_type, crn=CRN(), track_cycle_times=False)
    pl = rollouts.learn_iteration(env, bootstrap_policy, nr_states_to_explore, nr_rollouts, nr_steps_per_rollout, model_type=model_type)
    pl.cache = fill_cache(env, pl)
    extension = '.pth'
    pl.save(filename_without_extension + config_type + ".v1" + extension)

    best_pl = pl.copy() # Copy the best policy so far
    evaluation_env = smdp.SMDP(episode_length, config_type)
    print('Evaluating policy 1..')
    rewards, _ = rollouts.evaluate_policy(evaluation_env, best_pl.policy, nr_rollouts=300, nr_arrivals=2500, parallel=True)
    best_policy_reward = np.mean(rewards)
    print('Reward of policy version 1:', best_policy_reward)
    best_policy_v = 1

    for i in range(2, learning_iterations+1):
        pl = rollouts.learn_iteration(env, pl.policy, nr_states_to_explore, nr_rollouts, nr_steps_per_rollout, pl)
        pl.save(filename_without_extension + config_type + ".v" + str(i) + extension)
        pl.cache = fill_cache(env, pl)

        print(f'Evaluating new policy {i}..')
        # Evaluate the new policy
        rewards, _ = rollouts.evaluate_policy(evaluation_env, pl.policy, nr_rollouts=300, nr_arrivals=2500, parallel=True)
        new_policy_reward = np.mean(rewards)
        print(f'Reward of policy version {i}:', new_policy_reward)
        print(f"Reward of best policy, version {best_policy_v}: {best_policy_reward}. Reward of new policy, version {i}: {new_policy_reward}.")
        if new_policy_reward < best_policy_reward: # New policy is not better than the previous policy
            print("Reward of policy version", i, "is worse than previous policy, reverting to previous policy.")
            pl = best_pl.copy() # Revert to the previous policy
        else:
            print(f"Policy version {best_policy_v} improved, continuing with policy version {i}.")
            best_pl = pl.copy()
            best_policy_reward = new_policy_reward
            best_policy_v = i
        print('\n')

    best_pl.save(filename_without_extension + config_type + ".best_policy" + extension)           


def fill_cache(env, policy):
    _, frequent_states, _ =  determine_state_space(env)
    frequent_states = torch.tensor(np.array([policy.normalize_observation(np.array(state, dtype=float)) for state in frequent_states]), dtype=torch.float32)
    probabilities = policy.model(frequent_states).detach()
    cache = {tuple(state.detach().numpy()): _probabilities.detach().numpy() for state, _probabilities in zip(frequent_states, probabilities)}
    return cache

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
    
def evaluate_policy(filename, config_type, episode_length=10, nr_rollouts=100, results_dir=None, env_type='smdp'):
    env = smdp.SMDP(episode_length, config_type)
    pl = policy_learner.PolicyLearner.load(filename)
    version_number = re.search(r'v(\d+)', filename).group(1)
    rewards = rollouts.evaluate_policy(env, pl.policy, nr_rollouts)
    average_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    # Prepare results directory and file
    os.makedirs(results_dir, exist_ok=True)
    results_file_path = os.path.join(results_dir, f'results_{config_type}.txt')

    if version_number != '1':
        with open(results_file_path, 'a') as results_file:
            results_file.write(f"v{version_number}, {env_type}, {average_reward}, {std_reward}\n")
            print(f"Learned policy v{version_number}:", average_reward)
    else:
        with open(results_file_path, 'a') as results_file:
            results_file.write(f"v{version_number}, {env_type}, {average_reward}, {std_reward}\n")
            print(f"Learned policy v{version_number}:", average_reward)
            
            if env_type == 'smdp':
                if config_type == 'single_activity':
                    rewards, cycle_times = rollouts.evaluate_policy(env, smdp.threshold_policy, nr_rollouts)
                    average_reward = np.mean(rewards)
                    std_reward = np.std(rewards)
                    results_file.write(f"Threshold, {env_type}, {average_reward}, {std_reward}\n")
                    print("Threshold policy:", average_reward)
                
                rewards = rollouts.evaluate_policy(env, smdp.greedy_policy, nr_rollouts)
                average_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                results_file.write(f"Greedy, {env_type}, {average_reward}, {std_reward}\n")
                print('Greedy:', average_reward)
                
                rewards = rollouts.evaluate_policy(env, smdp.random_policy, nr_rollouts)
                average_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                results_file.write(f"Random, {env_type}, {average_reward}, {std_reward}\n")
                print('Random policy:', average_reward)

                rewards = rollouts.evaluate_policy(env, smdp.fifo_policy, nr_rollouts)
                average_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                results_file.write(f"FIFO, {env_type}, {average_reward}, {std_reward}\n")
                print('FIFO policy:', average_reward)

def evaluate_single_policy(pl, config_type, episode_length=10, nr_rollouts=100, results_dir=None, env_type='smdp', results_string = ''):
    env = smdp.SMDP(episode_length, config_type, track_cycle_times=True)
    pl_name = None
    if pl == greedy_policy:
        pl_name = 'spt'
    elif pl == random_policy:
        pl_name = 'random'
    elif pl == fifo_policy:
        pl_name = 'fifo'
    elif pl == threshold_policy:
        pl_name = 'threshold'
    elif type(pl) == policy_learner.ValueIterationPolicy:
        pl_name = 'vi'
    elif type(pl) == policy_learner.PolicyLearner:
        pl_name = 'pi'
    else:
        Exception('Unknown policy type')

    # Prepare results directory and file
    os.makedirs(results_dir, exist_ok=True)
    if pl_name == 'pi':
        results_file_path = os.path.join(results_dir, f'{pl_name}_{env_type}_{config_type}.txt')
        #results_file_path = os.path.join(results_dir, f'{pl_name}_{results_string}.txt')
    else:        
        results_file_path = os.path.join(results_dir, f'{pl_name}_{config_type}.txt')

    with open(results_file_path, 'w') as results_file:
        results_file.write(f"mean_reward,std_reward,mean_cycle_time,std_cycle_time\n")
        if pl_name in ['pi', 'vi']:
            rewards, cycle_times = rollouts.evaluate_policy(env, pl.policy, nr_rollouts, parallel=True)
        else:
            rewards, cycle_times = rollouts.evaluate_policy(env, pl, nr_rollouts, parallel=False)
        print(f'Mean reward: {np.mean(rewards)}, std reward: {np.std(rewards)}')
        print(f'Mean cycle time: {np.mean(cycle_times)}, std cycle time: {np.std(cycle_times)}')
        results_file.write(f"{np.mean(rewards)},{np.std(rewards)},{np.mean(cycle_times)},{np.std(cycle_times)}")


def compare_all_states(filename, episode_length=10, nr_rollouts=100):
    """
    Gets all observations by doing rollouts and storing all states.
    Apply the policy from the file as well as the threshold policy to each state, 
    print the observation and resulting actions for comparison. 
    """
    env = smdp.SMDP(episode_length)
    pl = policy_learner.PolicyLearner.load(filename)
    states = set()
    for _ in range(nr_rollouts):
        env.reset()
        _, rollout = rollouts.rollout_with_full_information(env, smdp.threshold_policy)
        for (observation, _, mask, _) in rollout:
            states.add((tuple(observation), tuple(mask)))
    for s in states:
        observation, mask = s
        action_policy = pl.predict(observation, mask)
        action_threshold = smdp.threshold_policy(None, observation, mask)
        if action_policy != action_threshold:
            print(observation, mask, action_policy, action_threshold)


def main():
    """
    Test random states
    """
    # env = smdp.SMDP(2500, 'high_utilization')
    # states = rollouts.random_states(env, random_policy, 5000)
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
    # for i in [1, 2, 3, 4, 5, 6, 7]:
    #     env = smdp.SMDP(3000, 'single_activity')
    #     reward = rollouts.evaluate_policy(env, threshold_policy, nr_rollouts=500)
    #     print(reward)

    """
    Training of the policies
    """

    # nr_rollouts = 100
    # nr_steps_per_rollout = 50
    # config_type = sys.argv[1] if len(sys.argv) > 1 else 'down_stream'
    # model_type = 'neural_network'
    # learning_iterations = 10

    # dir = f".//models//pi//smdp//{config_type}//"
    # if not os.path.exists(dir):
    #     os.makedirs(dir, exist_ok=True)

    # print('Learning SMDP policy for', config_type, 'with', model_type, 'model', flush=True)
    # print('Nr rollouts:', nr_rollouts, flush=True)
    # print('Rollout length:', nr_steps_per_rollout, f'(smdp steps = {nr_steps_per_rollout})', flush=True)

    # learn(config_type,
    #     greedy_policy,
    #     dir,
    #     learning_iterations=learning_iterations,
    #     episode_length=2500, # nr_cases
    #     nr_states_to_explore=20000,
    #     nr_rollouts=nr_rollouts,
    #     nr_steps_per_rollout=nr_steps_per_rollout,
    #     model_type=model_type)


    """
    Evaluation of the learned policies
    """

    config_type = 'slow_server'
    env_type = 'mdp'
    # config_type = sys.argv[1] if len(sys.argv) > 1 else 'slow_server'
    # env_type = sys.argv[2] if len(sys.argv) > 2 else 'mdp'
    policy = 'pi'

    config_types = ['single_activity']
    #config_types = ['single_activity']
    # env_type = 'mdp'
    for config_type in config_types:
        policies = []
        if config_type == 'single_activity':
            policies.append('threshold')
        for policy in policies:
            for env_type in ['smdp']:
                print(f'Evaluating policy for {config_type} with {policy} policy trained on an {env_type} environment.')
                #env_type = 'mdp'
                results_dir = f".//results//"

                env = smdp.SMDP(2500, config_type, reward_function='AUC', track_cycle_times=True)
                
                if policy == 'greedy':
                    pl = greedy_policy
                elif policy == 'random':
                    pl = random_policy
                elif policy == 'fifo':
                    pl = fifo_policy
                elif policy == 'threshold':
                    pl = threshold_policy
                elif policy == 'vi':
                    filename = f"./models/vi/{config_type}.npy"
                    pl = policy_learner.ValueIterationPolicy(env, max_queue=50, file=filename)
                elif policy == 'pi':
                    filename = f"./models/pi/{env_type}/{config_type}/{config_type}.best_policy.pth"
                    #filename = f"./models/pi/{env_type}/tuning/{config_type}_{nr_rollouts}_{nr_steps_per_rollout}_{tau_multiplier}/{config_type}.best_policy.keras"
                    pl = policy_learner.PolicyLearner.load(filename)
                    print('Filling cache..')
                    pl.cache = fill_cache(env, pl)
                    print(f'Cache filled with {len(pl.cache)} state-action pairs.')

                evaluate_single_policy(pl, config_type, episode_length=2500, nr_rollouts=300,
                                results_dir=results_dir, env_type=env_type)
                
                #nr_models = len([f for f in os.listdir(filename_folder) if f.endswith(extension)])
                # for i in range(1, nr_models):
                #     filename = filename_folder + f"{config_type}.v{i}.{extension}"
                #     evaluate_policy(filename, config_type, episode_length=3000, nr_rollouts=1, 
                #                     results_dir=results_dir, env_type=env_type)
                print('\n')

if __name__ == '__main__':
    main()
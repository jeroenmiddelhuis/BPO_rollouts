import smdp
import rollouts
import policy_learner
import sys, os, re
from collections import Counter
import numpy as np

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
    
    env = smdp.SMDP(episode_length, config_type)
    
    pl = rollouts.learn_iteration(env, bootstrap_policy, nr_states_to_explore, nr_rollouts, nr_steps_per_rollout, model_type=model_type)
    extension = '.keras' if model_type == 'neural_network' else '.xgb.json'
    pl.save(filename_without_extension + config_type + ".v1" + extension)

    best_policy = pl   
    best_policy_reward = np.mean(rollouts.evaluate_policy(env, best_policy.policy, nr_rollouts=100, nr_arrivals=2500))
    print('Reward of policy version 1:', best_policy_reward)
    best_policy_v = 1

    for i in range(1, learning_iterations+1):
        if i > 1:
            pl.save(filename_without_extension + config_type + ".v" + str(i) + extension)
        print("Policy verion " + str(i) + " learned, now testing")
        # print('Trained policy:', np.mean(rollouts.evaluate_policy(env, pl.policy, nr_rollouts=100, nr_arrivals=2500)))
        # print('Greedy policy:', np.mean(rollouts.evaluate_policy(env, smdp.greedy_policy, nr_rollouts=100, nr_arrivals=2500)))
        # print('FIFO policy:', np.mean(rollouts.evaluate_policy(env, smdp.fifo_policy, nr_rollouts=100, nr_arrivals=2500)))
        # print('Random policy:', np.mean(rollouts.evaluate_policy(env, smdp.random_policy, nr_rollouts=100, nr_arrivals=2500)))

        if i < learning_iterations:
            pl = rollouts.learn_iteration(env, pl.policy, nr_states_to_explore, nr_rollouts, nr_steps_per_rollout, pl)
            
            print('Evaluating new policy..')
            new_policy_reward = np.mean(rollouts.evaluate_policy(env, pl.policy, nr_rollouts=100, nr_arrivals=2500))
            print('Reward of policy version', i+1, ':', new_policy_reward)
            print(f"Reward of best policy, version {best_policy_v}: {best_policy_reward}. Reward of new policy, version {i+1}: {new_policy_reward}.")
            if new_policy_reward < best_policy_reward: # Higher reward is better (less negative)
                print("Reward of policy version", i, "is worse than previous policy, reverting to previous policy.")
                pl = best_policy
            else:
                print(f"Policy version {best_policy_v} improved, continuing with policy version {i + 1}.")
                best_policy = pl
                best_policy_reward = new_policy_reward
                best_policy_v = i + 1
            print('\n')

    best_policy.save(filename_without_extension + config_type + ".best_policy" + extension)           


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
                    rewards = rollouts.evaluate_policy(env, smdp.threshold_policy, nr_rollouts)
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

def evaluate_single_policy(pl, config_type, episode_length=10, nr_rollouts=100, results_dir=None, env_type='smdp'):
    env = smdp.SMDP(episode_length, config_type)
    pl_name = None
    if pl == smdp.greedy_policy:
        pl_name = 'spt'
    elif pl == smdp.random_policy:
        pl_name = 'random'
    elif pl == smdp.fifo_policy:
        pl_name = 'fifo'
    elif pl == smdp.threshold_policy:
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
    else:        
        results_file_path = os.path.join(results_dir, f'{pl_name}_{config_type}.txt')

    with open(results_file_path, 'a') as results_file:
        results_file.write(f"mean,std\n")
        if pl_name in ['pi', 'vi']:
            rewards = rollouts.evaluate_policy(env, pl.policy, nr_rollouts)
        else:
            rewards = rollouts.evaluate_policy(env, pl, nr_rollouts)
        print(np.mean(rewards), np.std(rewards))
        results_file.write(f"{np.mean(rewards)},{np.std(rewards)}")


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
    # states = rollouts.random_states(env, smdp.random_policy, 5000)
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
    #     reward = rollouts.evaluate_policy(env, smdp.threshold_policy, nr_rollouts=500)
    #     print(reward)

    """
    Training of the policies
    """

    config_type = sys.argv[1] if len(sys.argv) > 1 else 'down_stream'
    model_type = 'neural_network' #sys.argv[2] if len(sys.argv) > 2 else 'neural_network'
    learning_iterations = 10

    dir = f".//models//pi//smdp//{config_type}//"
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    print('Learning SMDP policy for', config_type, 'with', model_type, 'model', flush=True)

    learn(config_type,
        smdp.random_policy,
        dir,
        learning_iterations=learning_iterations,
        episode_length=2500, # nr_cases
        nr_states_to_explore=100,
        nr_rollouts=100,
        nr_steps_per_rollout=50,
        model_type=model_type)


    """
    Evaluation of the learned policies
    """
    # config_type = sys.argv[1] if len(sys.argv) > 1 else ''
    # env_type = sys.argv[2] if len(sys.argv) > 2 else 'mdp'
    # policy = 'fifo'

    # # config_types = ['n_system', 'high_utilization', 'down_stream', 'low_utilization', 'parallel', 'single_activity', 'slow_server']
    # # env_type = 'mdp'
    # for config_type in ['n_system', 'high_utilization', 'down_stream', 'low_utilization', 'parallel', 'single_activity', 'slow_server']:
    #     for policy in ['vi']:
    #         for env_type in ['smdp']:
    #             print(config_type, env_type, policy)
    #             #env_type = 'mdp'
    #             results_dir = f".//results//"

    #             env = smdp.SMDP(2500, config_type, reward_function='AUC')
                
    #             if policy == 'greedy':
    #                 pl = smdp.greedy_policy
    #             elif policy == 'random':
    #                 pl = smdp.random_policy
    #             elif policy == 'fifo':
    #                 pl = smdp.fifo_policy
    #             elif policy == 'threshold':
    #                 pl = smdp.threshold_policy
    #             elif policy == 'vi':
    #                 filename = f"./models/vi/{config_type}.npy"
    #                 pl = policy_learner.ValueIterationPolicy(env, max_queue=50, file=filename)
    #             elif policy == 'pi':
    #                 filename = f"./models/pi/{env_type}/{config_type}/{config_type}.best_policy.keras"
    #                 pl = policy_learner.PolicyLearner.load(filename)

    #             evaluate_single_policy(pl, config_type, episode_length=2500, nr_rollouts=300, 
    #                             results_dir=results_dir, env_type=env_type)
                
    #             #nr_models = len([f for f in os.listdir(filename_folder) if f.endswith(extension)])
    #             # for i in range(1, nr_models):
    #             #     filename = filename_folder + f"{config_type}.v{i}.{extension}"
    #             #     evaluate_policy(filename, config_type, episode_length=3000, nr_rollouts=1, 
    #             #                     results_dir=results_dir, env_type=env_type)
    #             print('\n')

if __name__ == '__main__':
    main()
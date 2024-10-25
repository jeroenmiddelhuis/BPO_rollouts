import mdp
import rollouts
import policy_learner
import sys, os, re

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
    env = mdp.MDP(episode_length, config_type, tau)

    pl = rollouts.learn_iteration(env, bootstrap_policy, nr_states_to_explore, nr_rollouts, nr_steps_per_rollout, model_type=model_type)

    for i in range(1, learning_iterations+1):
        if model_type == 'neural_network':
            pl.save(filename_without_extension + config_type + ".v" + str(i) + ".keras")
        elif model_type == 'xgboost':
            pl.model.save_model(filename_without_extension + config_type + ".v" + str(i) + ".xgb.json")
        print("Policy verion " + str(i) + " learned, now testing")
        print('Trained policy:', rollouts.evaluate_policy(env, pl.policy, nr_rollouts, nr_arrivals=3000),
              'Greedy policy:', rollouts.evaluate_policy(env, mdp.greedy_policy, nr_rollouts, nr_arrivals=3000),
              'Random policy:', rollouts.evaluate_policy(env, mdp.random_policy, nr_rollouts, nr_arrivals=3000))
        if i < learning_iterations:
            # pl = rollouts.learn_iteration(env, pl.policy, nr_states_to_explore, nr_rollouts, only_statistically_significant, pl)
            pl = rollouts.learn_iteration(env, pl.policy, nr_states_to_explore, nr_rollouts, nr_steps_per_rollout, pl)


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
    env = mdp.MDP(episode_length, config_type, tau=0.25)
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


"""
Test of rollout
"""

# env = mdp.MDP(3000, 'n_system')
# reward = rollouts.evaluate_policy(env, mdp.fifo_policy, nr_rollouts=100)
# print(reward)

"""
Training of the policies
"""

config_type = sys.argv[1] if len(sys.argv) > 1 else 'n_system'
model_type = sys.argv[2] if len(sys.argv) > 2 else 'xgboost'
learning_iterations = 10

dir = f".//models//mdp//{config_type}//"
if not os.path.exists(dir):
    os.makedirs(dir, exist_ok=True)

print('Learning SMDP policy for', config_type, 'with', model_type, 'model', flush=True)

learn(config_type, 
      mdp.random_policy,
      dir,
      learning_iterations=learning_iterations,
      episode_length=2500, # nr_cases
      nr_states_to_explore=5000,
      nr_rollouts=50,
      nr_steps_per_rollout=100,
      model_type=model_type,
      tau=0.5)


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

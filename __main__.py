import smdp
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
          only_statistically_significant=False):
    
    env = smdp.SMDP(episode_length, config_type)

    pl = rollouts.learn_iteration(env, bootstrap_policy, nr_states_to_explore, nr_rollouts, nr_steps_per_rollout, model_type=model_type)

    for i in range(1, learning_iterations+1):
        if model_type == 'neural_network':
            pl.save(filename_without_extension + config_type + ".v" + str(i) + ".keras")
        elif model_type == 'xgboost':
            pl.model.save_model(filename_without_extension + config_type + ".v" + str(i) + ".xgb.json")
        print("Policy verion " + str(i) + " learned, now testing")
        print('Trained policy:', rollouts.evaluate_policy(env, pl.policy, nr_rollouts, nr_arrivals=3000),
              'Greedy policy:', rollouts.evaluate_policy(env, smdp.greedy_policy, nr_rollouts, nr_arrivals=3000),
              'Random policy:', rollouts.evaluate_policy(env, smdp.random_policy, nr_rollouts, nr_arrivals=3000))
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
    
def evaluate_policy(filename, config_type, episode_length=10, nr_rollouts=100, results_dir=None, model_type=None, env_type='smdp'):
    env = smdp.SMDP(episode_length, config_type)
    pl = policy_learner.PolicyLearner.load(filename, model_type)
    version_number = re.search(r'v(\d+)', filename).group(1)
    average_reward = rollouts.evaluate_policy(env, pl.policy, nr_rollouts)
    
    # Prepare results directory and file
    os.makedirs(results_dir, exist_ok=True)
    results_file_path = os.path.join(results_dir, f'results_{config_type}.txt')

    if version_number != '1':
        with open(results_file_path, 'a') as results_file:
            results_file.write(f"{model_type}, v{version_number}, {env_type}, {average_reward}\n")
            print(f"Learned {model_type} policy v{version_number}:", average_reward)
    else:
        with open(results_file_path, 'a') as results_file:
            results_file.write(f"{model_type}, v{version_number}, {env_type}, {average_reward}\n")
            print(f"Learned {model_type} policy v{version_number}:", average_reward)
            
            if config_type == 'single_activity':
                average_reward = rollouts.evaluate_policy(env, smdp.threshold_policy, nr_rollouts)
                results_file.write(f"Threshold, None, {env_type}, {average_reward}\n")
                print("Threshold policy:", average_reward)
            else:
                average_reward = rollouts.evaluate_policy(env, smdp.greedy_policy, nr_rollouts)
                results_file.write(f"Greedy, None, {env_type}, {average_reward}\n")
                print('Greedy:', average_reward)
                
                average_reward = rollouts.evaluate_policy(env, smdp.random_policy, nr_rollouts)
                results_file.write(f"Random, None, {env_type}, {average_reward}\n")
                print('Random policy:', average_reward)

                average_reward = rollouts.evaluate_policy(env, smdp.fifo_policy, nr_rollouts)
                results_file.write(f"FIFO, None, {env_type}, {average_reward}\n")
                print('FIFO policy:', average_reward)

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

"""
Test of rollout
"""

# env = smdp.SMDP(3000, 'n_system')
# reward = rollouts.evaluate_policy(env, smdp.fifo_policy, nr_rollouts=100)
# print(reward)

"""
Training of the policies
"""

# config_type = sys.argv[1] if len(sys.argv) > 1 else 'n_system'
# model_type = sys.argv[2] if len(sys.argv) > 2 else 'neural_network'
# learning_iterations = 10

# dir = f".//models//smdp//{config_type}//"
# if not os.path.exists(dir):
#     os.makedirs(dir, exist_ok=True)

# print('Learning SMDP policy for', config_type, 'with', model_type, 'model', flush=True)

# learn(config_type, 
#       smdp.random_policy,
#       dir,
#       learning_iterations=learning_iterations,
#       episode_length=2500, # nr_cases
#       nr_states_to_explore=5000,
#       nr_rollouts=50,
#       nr_steps_per_rollout=100,
#       model_type=model_type)


"""
Evaluation of the learned policies
"""
config_type = sys.argv[1] if len(sys.argv) > 1 else 'n_system'
model_type = sys.argv[2] if len(sys.argv) > 2 else 'xgboost'
env_type = sys.argv[3] if len(sys.argv) > 3 else 'smdp'

for config_type in [config_type]: # ['n_system', 'high_utilization']:
#config_type = 'down_stream'
    for model_type in [model_type]:
    #model_type = 'xgboost'
        for env_type in [env_type]:
            print(config_type, model_type, env_type)
            extension = 'keras' if model_type == 'neural_network' else 'xgb.json'
            #env_type = 'mdp'
            results_dir = f".//results//smdp//"

            filename_folder = f".//models//{env_type}//{config_type}//"

            nr_models = len([f for f in os.listdir(filename_folder) if f.endswith(extension)])

            for i in range(1, nr_models + 1):
                filename = filename_folder + f"{config_type}.v{i}.{extension}"
                evaluate_policy(filename, config_type, episode_length=1000, nr_rollouts=100, results_dir=results_dir, model_type=model_type, env_type=env_type)
            print('\n')
import smdp
import rollouts
import policy_learner
import sys, os


def learn(config_type, bootstrap_policy, filename_without_extension, learning_iterations=3, episode_length=10, nr_states_to_explore=1000, nr_rollouts=100, only_statistically_significant=False):
    env = smdp.SMDP(episode_length, config_type)

    pl = rollouts.learn_iteration(env, bootstrap_policy, nr_states_to_explore, nr_rollouts, only_statistically_significant)

    for i in range(1, learning_iterations+1):
        pl.save(filename_without_extension + config_type + ".v" + str(i) + ".keras")
        print("Policy verion " + str(i) + " learned, now testing")
        print(rollouts.evaluate_policy(env, pl.policy, nr_rollouts, nr_arrivals=200))
        if i < learning_iterations:
            pl = rollouts.learn_iteration(env, pl.policy, nr_states_to_explore, nr_rollouts, only_statistically_significant, pl)


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
    

def evaluate_policy(filename, config_type, episode_length=10, nr_rollouts=100):
    env = smdp.SMDP(episode_length, config_type)
    pl = policy_learner.PolicyLearner.load(filename)
    print("Learned policy:", rollouts.evaluate_policy(env, pl.policy, nr_rollouts))
    if config_type == 'single_activity':
        print("Threshold policy:", rollouts.evaluate_policy(env, smdp.threshold_policy, nr_rollouts))
    else:
        print('Greedy policy:', rollouts.evaluate_policy(env, smdp.greedy_policy, nr_rollouts))
        print('Random policy:', rollouts.evaluate_policy(env, smdp.random_policy, nr_rollouts))


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

config_type = 'slow_server'
learning_iterations = 20

dir = f".//models//mdp_full_rollout//{config_type}//"
if not os.path.exists(dir):
    os.makedirs(dir)

# policy = './models/smdp_full_rollout/slow_server/slow_server.v5.keras'

# evaluate_policy(policy, config_type, episode_length=200, nr_rollouts=100)

#
learn(config_type, smdp.random_policy, dir, episode_length=50, learning_iterations=learning_iterations, nr_states_to_explore=10)
#show_policy("./models/policy_smdp.random.v40.keras")
# print('standard rollout')
# evaluate_policy(f"./models/high_utilization_smdp_no_rollout.random.v5.keras", config_type, episode_length=300, nr_rollouts=1000)
# print('full rollout (+50 cases)')
# evaluate_policy(f"./models/high_utilization_smdp.random.v2.keras", config_type, episode_length=300, nr_rollouts=1000)
#compare_all_states("./models/policy_smdp.random.v40.keras", episode_length=50)
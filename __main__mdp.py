import mdp
import rollouts
import policy_learner
import sys, os


def learn(config_type, bootstrap_policy, filename_without_extension, learning_iterations=3, episode_length=10, nr_states_to_explore=1000, nr_rollouts=100, only_statistically_significant=False):
    env = mdp.MDP(episode_length, config_type)

    pl = rollouts.learn_iteration(env, bootstrap_policy, nr_states_to_explore, nr_rollouts, only_statistically_significant)

    for i in range(1, learning_iterations+1):
        pl.save(filename_without_extension + config_type + ".v" + str(i) + ".keras")
        print("Policy verion " + str(i) + " learned, now testing")
        print(rollouts.evaluate_policy(env, pl.policy, nr_rollouts, nr_arrivals=1000))
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
    

def evaluate_policy(filename, episode_length=10, nr_rollouts=100):
    env = mdp.MDP(episode_length)
    pl = policy_learner.PolicyLearner.load(filename)
    print("Learned policy:", rollouts.evaluate_policy(env, pl.policy, nr_rollouts))
    print("Threshold policy:", rollouts.evaluate_policy(env, mdp.threshold_policy, nr_rollouts))


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


config_type = sys.argv[1]#'slow_server'
learning_iterations = 20

dir = f".//models//mdp_full_rollout//{config_type}"
if not os.path.exists(dir):
    os.makedirs(dir)

learn(config_type, mdp.random_policy, dir, episode_length=50, learning_iterations=learning_iterations, nr_states_to_explore=1000)
# show_policy("./models/policy_mdp.random_mdp.v39.keras")
# evaluate_policy("./models/policy_mdp.random_mdp.v39.keras", episode_length=50)
# compare_all_states("./models/policy_mdp.random_mdp.v39.keras", episode_length=50)


# show_policy("./models/policy_smdp.random.v40.keras")
# evaluate_policy("./models/policy_smdp.random.v40.keras", episode_length=50, nr_rollouts=1000)
# compare_all_states("./models/policy_smdp.random.v40.keras", episode_length=50)
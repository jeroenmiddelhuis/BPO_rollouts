import scipy.stats as stats
import numpy as np
import math
from tqdm import tqdm
from policy_learner import PolicyLearner
from crn import CRN


def rollout(env, policy):
    """
    Does a rollout for the environment from the current state using the given policy.
    Assumes that the environment is terminating.
    Note that we assume no discounting of rewards here.
    """
    done = False
    total_reward = 0
    while not done:
        action = policy(env)
        _, reward, done, _, _ = env.step(action)
        total_reward += reward
    return total_reward


def rollout_with_full_information(env, policy):
    """
    Does a rollout for the environment from the current state using the given policy.
    Assumes that the environment is terminating.
    Note that we assume no discounting of rewards here.
    Returns each step (observation, action, reward) as a list. Thus, mainly useful for testing purposes.
    """
    done = False
    total_reward = 0
    rollout_info = []
    while not done:
        observation = env.observation()
        mask = env.action_mask()
        action = policy(env)
        _, reward, done, _, _ = env.step(action)
        total_reward += reward
        rollout_info.append((observation, action, mask, reward))
    return total_reward, rollout_info


def multiple_rollouts_per_action(env, policy, nr_rollouts_per_action):
    """
    Does multiple rollouts per possible action
    from the current state of the environment 
    using the given policy.
    Returns the observation corresponding to the current state, the list of possible actions,
    and a dictionary from actions to a list of rewards (one for each rollout) for each action.
    """
    state = env.get_state()
    observation = env.observation()
    action_mask = env.action_mask()
    if sum(action_mask) < 2:  # we need at least two actions to compare, otherwise we can't learn anything.
        return None, None, None
    # Note that actions are one-hot encoded, so we translate the action mask to a list of possible actions.
    # We then turn them into tuples, so we can use them as keys in a dictionary.
    possible_actions = [tuple([0]*i + [1] + [0]*(len(action_mask)-i-1)) for i in range(len(action_mask)) if action_mask[i]]
    rewards = {action: [] for action in possible_actions}
    crn = CRN()
    for _ in range(nr_rollouts_per_action):
        crn.reset()  # we reset the common random numbers generator for each rollout.
        for action in possible_actions:
            crn.restart_sequence()  # but for each action we use the same random numbers, so the rollouts differ because of the actions, not because of randomness.
            env.set_state(state)
            # step the action and get the reward
            _, reward, _, _, _ = env.step(action)
            rewards[action].append(rollout(env, policy) + reward)
    return observation, possible_actions, rewards


def find_learning_sample(env, policy, nr_rollouts_per_action, only_statistically_significant=False):
    """
    Finds a learning sample by comparing the rewards of different actions.
    If one action is significantly better than the others, we add a learning sample for that action.
    """
    observation, possible_actions, rewards = multiple_rollouts_per_action(env, policy, nr_rollouts_per_action)
    if observation is None:  # if we can't learn anything, we return None.
        return None
    means = {action: np.mean(rewards[action]) for action in possible_actions}
    best_action = max(means, key=means.get)
    if not only_statistically_significant:  # if we don't care about statistical significance, we just return the best action.
        return (observation, best_action)
    # We use a t-test to check if the best action is significantly better than the others.
    # We use a one-sided test, because we are only interested in the case where the best action is better.
    better_than_others = True
    for action in possible_actions:
        if action != best_action:
            t_stat, p_value = stats.ttest_ind(rewards[best_action], rewards[action], equal_var=False)
            if p_value > 0.05:
                better_than_others = False
                break
    if better_than_others:
        return (observation, best_action)
    else:
        return None


def random_states(env, policy, nr_states):
    """
    Generates nr (non-terminal) random states by rolling using the given policy.
    """
    # Roll out to the end of the episode to check how long an episode is.
    env.reset()
    done = False
    steps = 0
    while not done:
        action = policy(env)
        _, _, done, _, _ = env.step(action)
        steps += 1
    # Now we know how long an episode is.
    # Let's sample each rollout approximately equally often.
    nr_samples_per_rollout = int(math.sqrt(nr_states))
    # This means that we sample each time after approx. steps/nr_samples_per_rollout steps.
    # The next sample is taken after a normally distributed number of steps with mean steps/nr_samples_per_rollout and standard deviation steps/nr_samples_per_rollout.
    sampled_states = []
    env.reset()
    while len(sampled_states) < nr_states:
        next_sample_after = -1
        while next_sample_after < 0:
            next_sample_after = int(np.random.normal(steps/nr_samples_per_rollout, steps/nr_samples_per_rollout))
        current_step = 0
        while current_step != next_sample_after:
            action = policy(env)
            _, _, done, _, _ = env.step(action)
            current_step += 1
            if done:
                env.reset()
        sampled_states.append(env.get_state())
    return sampled_states


def learn_iteration(env, policy, nr_states_to_explore=100, nr_rollouts_per_action_per_state=50, only_statistically_significant=False):
    """
    Does one iteration of learning. In the following steps:
    1. Samples start states.
    2. For each start state, finds a learning sample (if any is available).
    3. Updates the policy.
    The policy is learned as neural network.
    Returns the policy as a class.
    """
    states = random_states(env, policy, nr_states_to_explore)
    learning_samples_X = []
    learning_samples_y = []
    for i in tqdm(range(len(states))):
        state = states[i]
        env.set_state(state)
        learning_sample = find_learning_sample(env, policy, nr_rollouts_per_action_per_state, only_statistically_significant)
        if learning_sample is not None:
            learning_samples_X.append(learning_sample[0])
            learning_samples_y.append(learning_sample[1])
    print("Number of learning samples found:", len(learning_samples_X), "out of", len(states), "states.")
    if only_statistically_significant:
        print("If the number of learning samples if low, you might want to increase the number of rollouts per action per state, to get more significantly better actions.")
    # now we have the learning samples, let's train a new policy.
    learner = PolicyLearner()
    learner.build_model(learning_samples_X, learning_samples_y)
    return learner


def evaluate_policy(env, policy, nr_rollouts=100):
    """
    Evaluates the policy by doing a number of rollouts and averaging the rewards.
    """
    total_reward = 0
    for _ in range(nr_rollouts):
        env.reset()
        total_reward += rollout(env, policy)
    return total_reward / nr_rollouts

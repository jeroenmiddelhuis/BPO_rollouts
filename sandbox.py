import rollouts
import smdp
import mdp
import policy_learner
import rollouts
from crn import CRN
from tqdm import tqdm
import os
import pickle

env = smdp.SMDP(2500, config_type='slow_server')
greedy_policy = smdp.greedy_policy

states = rollouts.random_states(env, smdp.random_policy, nr_states=5000) # we use the random policy to sample the start states. SMDP and MDP random policy is the same.

learning_samples_X = []
learning_samples_y = []
for i in tqdm(range(len(states))):
    state = states[i]
    env.set_state(state)
    learning_sample = rollouts.find_learning_sample(env, greedy_policy, nr_rollouts_per_action=100, nr_steps_per_rollout=50, only_statistically_significant=False)
    if learning_sample is not None:
        learning_samples_X.append(learning_sample[0])
        learning_samples_y.append(learning_sample[1])


# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save learning samples
with open('data/mdp_learning_samples.pkl', 'wb') as f:
    pickle.dump({
        'X': learning_samples_X,
        'y': learning_samples_y
    }, f)

# env.set_state([{'a':[1], 'b':[2]}, [(2, 'b')], [(3, 'a')], 0, 2, 10])

# self.waiting_cases = state[0].copy()
# self.processing_r1 = state[1].copy()
# self.processing_r2 = state[2].copy()
# self.total_time = state[3]
# self.total_arrivals = state[4]
# self.nr_arrivals = state[5]

# print(env.observation())
# #print([env.sample_next_task('Start') for _ in range(50)])
# print(env.action_space)
# env.step([0,0,0,0,1])  # now a case must have arrived consequently r1, r2, and postpone are all possible
# observation, possible_actions, rewards = rollouts.multiple_rollouts_per_action(env, mdp.greedy_policy, 5)
# print(observation)
# print(possible_actions)
# if rewards is not None:
#     for action in rewards:
#         print(round(sum(rewards[action])))


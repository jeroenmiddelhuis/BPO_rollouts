import rollouts
import smdp
import mdp
from crn import CRN

env = mdp.MDP(10)
env.set_state(([1, 2], [], [], 0, 2, 10))

# self.waiting_cases = state[0].copy()
# self.processing_r1 = state[1].copy()
# self.processing_r2 = state[2].copy()
# self.total_time = state[3]
# self.total_arrivals = state[4]
# self.nr_arrivals = state[5]


print(env.observation())
env.step([0,0,0,1])  # now a case must have arrived consequently r1, r2, and postpone are all possible
observation, possible_actions, rewards = rollouts.multiple_rollouts_per_action(env, mdp.greedy_policy, 5)
print(observation)
print(possible_actions)
if rewards is not None:
    for action in rewards:
        print(round(sum(rewards[action])))
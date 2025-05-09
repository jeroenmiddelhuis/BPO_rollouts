from gymnasium import spaces, Env
import numpy as np


class Environment(Env):
    def __init__(self, env) -> None:
        super().__init__()
        self.env = env
        self.config_type = env.config_type
        self.nr_actions = len(env.action_space)#
        print(self.config_type)
        self.episode_reward = 0
        self.selected_actions = {action: 0 for action in self.env.action_space}

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.nr_actions)  # Example: 4 possible actions
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.env.state_space),), dtype=np.float64)

    def normalize_observation(self, observation):
        if len(observation) == 3: # single_activity
            observation[-1] = np.minimum(1.0, observation[-1] / 100.0)
        elif len(observation) <= 8: # 2 activity scenarios
            observation[-2:] = np.minimum(1.0, observation[-2:] / 100.0)
        else: # composite model
            observation[-12:] = np.minimum(1.0, observation[-12:] / 100.0)
        return observation

    def reset(self, seed=None):
        if len(self.env.cycle_times) > 0:
            print('------------------------------------------')
            print('Episode over')
            print('Episode reward:', self.episode_reward)
            print('Number of cases:', self.env.total_arrivals)
            print('Completed cases:', len(self.env.cycle_times))
            print('Uncompleted cases:', self.env.total_arrivals - len(self.env.cycle_times))
            print('Average cycle time:', np.mean(list(self.env.cycle_times.values())))
            print('Total cycle time:', np.sum(list(self.env.cycle_times.values())))
            print('Selected actions:', self.selected_actions)
            print('------------------------------------------')
            print('\n')

        self.env.reset()
        
        self.episode_reward = 0
        self.selected_actions = {action: 0 for action in self.env.action_space}

        observation = np.array(self.env.observation(), dtype=float)
        observation = self.normalize_observation(observation)
        return observation, {}

    def step(self, action):
        self.selected_actions[self.env.action_space[action]] += 1
        selected_action = action
        action = [0] * self.nr_actions
        action[selected_action] = 1

        observation, reward, done, _, _ = self.env.step(action)
        observation = np.array(observation, dtype=float)
        observation = self.normalize_observation(observation)
        self.episode_reward += reward
        return observation, reward, done, {}, {}

    def render(self):
        pass

    def action_masks(self):
        return self.env.action_mask()

    def close(self):
        # Clean up resources
        pass

def main():
    import mdp, smdp
    from heuristic_policies import fifo_policy, random_policy, greedy_policy, threshold_policy
    env = mdp.MDP(2500, 'high_utilization')
    gym_env = Environment(env)
    pl = greedy_policy
    total_rewards = []
    cycle_times = []
    for _ in range(100):
        total_reward = 0
        gym_env.reset()
        while True:
            action = pl(gym_env.env)
            if isinstance(action, list):
                action = gym_env.env.action_space[np.argmax(action)]
            action = gym_env.env.action_space.index(action)
            observation, reward, done, _, _ = gym_env.step(action)
            total_reward += reward
            if done:
                cycle_times.append(np.mean(list(gym_env.env.cycle_times.values())))
                break
        total_rewards.append(total_reward)
    print(np.mean(total_rewards), np.std(total_rewards))
    print(np.mean(cycle_times), np.std(cycle_times))

if __name__ == '__main__':
    main()
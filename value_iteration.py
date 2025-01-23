import mdp
import numpy as np
import itertools
from tqdm import tqdm
import os

class ValueIteration:
    def __init__(self, env, gamma=0.9, theta=0.0001):
        self.env = env
        self.tau = env.tau
        self.config_type = self.env.config_type
        self.gamma = gamma
        self.theta = theta
        self.max_queue = 50
        self.out_of_bounds_penalty = -10000

        #print('state space', self.state_space, type(self.state_space[0]))
        self.feature_ranges, self.state_space, self.all_states = self.determine_state_space()
        print('state space', type(self.state_space[0]))

    def determine_state_space(self):
        feature_ranges = []
        for state_label in self.env.state_space:
            if 'is_available_' in state_label:
                feature_ranges.append(2)
            elif 'assigned_' in state_label:
                feature_ranges.append(2)
            elif 'waiting_' in state_label:
                feature_ranges.append(self.max_queue + 1)
        all_states = [list(state) for state in itertools.product(*[range(i) for i in feature_ranges])]
        state_space = []
        for observation in all_states:
            try:
                self.check_feasible_observation(observation)
                state_space.append(observation)                
            except AssertionError:
                continue
        return feature_ranges, state_space, all_states

    def clip_observation(self, observation):
        penalty = 0

        # Clip the queue
        if observation[-1] > self.max_queue:
            penalty += self.out_of_bounds_penalty
            observation[-1] = self.max_queue
        
        # Clip s8 if it exists (non-single activity case)
        if self.config_type != 'single_activity':
            if observation[-2] > self.max_queue:
                penalty += self.out_of_bounds_penalty
                observation[-2] = self.max_queue
        
        return observation, penalty
    
    def check_feasible_observation(self, observation):
        if self.config_type == 'single_activity':
            assert len(observation) == 3
        elif self.config_type == 'n_system':
            assert len(observation) == 6
        else:
            assert len(observation) == 8

        # The sum of all labels related to r1 and r2 should be 1
        # TODO SUM OF OBSERVATIONS FOR N_SYSTEM AND SINGLE_ACTIVITY
        if self.config_type not in ['n_system', 'single_activity']:
            assert sum([1 for i, x in enumerate(observation) if x == 1 and 'r1' in self.env.state_space[i]]) == 1
        if self.config_type != 'single_activity':
            assert sum([1 for i, x in enumerate(observation) if x == 1 and 'r2' in self.env.state_space[i]]) == 1

        # All values should be positive
        assert all([x >= 0 for x in observation])

        # The values not related to the queue should be 0 or 1
        for i, x in enumerate(observation):
            if 'waiting' not in self.env.state_space[i]:
                assert x == 0 or x == 1

    def set_state_from_observation(self, env, observation):
        state_space_labels = env.state_space
        waiting_cases = {task_type: [] for task_type in env.task_types}
        processing_r1 = []
        processing_r2 = []
        total_time = 0 # Not relevant for state transitions
        total_arrivals = 0 # Not relevant for state transitions
        nr_arrivals = 1000 # Not relevant for state transitions, but should be different than total_arrivals

        # Set waiting cases
        for task_type in self.env.task_types:
            nr_cases = observation[state_space_labels.index(f'waiting_{task_type}')]
            waiting_cases[task_type] = [i for i in range(nr_cases)]

        for state_space_label in state_space_labels:
            # Single activity case, so we don't have assigned_ features
            if self.config_type == 'single_activity' and state_space_label.startswith('is_available'):
                resource = state_space_label.split('_')[-1]
                if observation[state_space_labels.index(state_space_label)] == 0: # == 0 which means the resource is assigned
                    if resource == 'r1':
                        processing_r1 = [(len(waiting_cases[task_type]), task_type)]
                    if resource == 'r2':
                        processing_r2 = [(len(waiting_cases[task_type]) + len(processing_r1), task_type)]

            if self.config_type == 'n_system' and state_space_label == 'is_available_r1':
                # In n_system, r2 can process both activities, but r1 can only process B
                # Therefore, we don't have the assigned_ features for r1
                if observation[state_space_labels.index('is_available_r1')] == 0: # == 0 which means the resource is assigned
                    processing_r1 = [(len(waiting_cases[task_type]), 'b')]

            if state_space_label.startswith('assigned_'):
                # Generate processing_r1, r2 from the assigned_ features for remaining scenarios
                if observation[state_space_labels.index(state_space_label)] == 1:
                    assignment = state_space_label.split('_')[1]
                    resource, task_type = assignment[0:2], assignment[2]
                    if resource == 'r1':
                        processing_r1 = [(len(waiting_cases[task_type]), task_type)]
                    elif resource == 'r2':
                        if self.config_type != 'parallel':
                            processing_r2 = [(len(waiting_cases[task_type]) + len(processing_r1), task_type)]
                        else:
                            is_processing_r1_same_case = 1 if len(processing_r1) > 0 and processing_r1[0][1] == task_type else 0                                
                            processing_r2 = [(len(waiting_cases[task_type]) + is_processing_r1_same_case, task_type)]

        env.set_state((waiting_cases, processing_r1, processing_r2, total_time, total_arrivals, nr_arrivals, None))

    def observation(self):
        return self.env.observation()
    
    def get_transformed_evolutions(self, processing_r1, processing_r2, arrivals_coming=True, action=None):
        return self.env.get_transformed_evolutions(processing_r1, processing_r2, arrivals_coming, action)

    def observation_to_index(self, observation):
        """
        Convert a state tuple to a unique index.
        
        :param state: A tuple representing the state (e_1, e_2, ..., e_6).
        :param ranges: A list of the number of possible values for each element in the state.
        :return: A unique index for the state.
        """
        assert len(observation) == len(self.feature_ranges), f"State ({len(observation)}) and ranges ({len(self.feature_ranges)}) length must match"
        
        index = 0
        multiplier = 1
        for i in reversed(range(len(observation))):
            index += observation[i] * multiplier
            if i > 0:  # Prepare multiplier for the next element (if any)
                multiplier *= self.feature_ranges[i]        
        return index

    def process_observation(self, observation, action):
        if action == 'r1a':
            observation[self.env.state_space.index('is_available_r1')] = 0
            if self.config_type not in ['n_system', 'single_activity']:
                observation[self.env.state_space.index('assigned_r1a')] = 1
            observation[self.env.state_space.index('waiting_a')] -= 1
        elif action == 'r1b':
            observation[self.env.state_space.index('is_available_r1')] = 0
            if self.config_type not in ['n_system', 'single_activity']: 
                observation[self.env.state_space.index('assigned_r1b')] = 1
            observation[self.env.state_space.index('waiting_b')] -= 1
        elif action == 'r2a':
            observation[self.env.state_space.index('is_available_r2')] = 0
            if self.config_type != 'single_activity':
                observation[self.env.state_space.index('assigned_r2a')] = 1
            observation[self.env.state_space.index('waiting_a')] -= 1
        elif action == 'r2b':
            observation[self.env.state_space.index('is_available_r2')] = 0
            if self.config_type != 'single_activity':
                observation[self.env.state_space.index('assigned_r2b')] = 1
            observation[self.env.state_space.index('waiting_b')] -= 1
        return observation

    def evolve_observation(self, observation, evolution):
        observation = observation.copy()
        if evolution == 'return_to_state':
            return observation
        elif evolution == 'arrival':
            if self.config_type == 'single_activity':
                observation[-1] += 1
                return observation
            elif self.config_type == 'n_system':
                observation1 = observation.copy()
                observation2 = observation.copy()
                observation1[-1] += 1 # 50% chance that an arrival will happen at either activity
                observation2[-2] += 1
                return (observation1, observation2)
            elif self.config_type == 'parallel':
                # Arrival at both activities
                observation[-1] += 1
                observation[-2] += 1
                return observation
            else:
                # Arrival at activity a
                observation[-2] += 1
                return observation
        elif evolution == 'r1a':         
            observation[self.env.state_space.index('is_available_r1')] = 1 # Resource becomes available
            if self.config_type not in ['n_system', 'single_activity']:
                observation[self.env.state_space.index('assigned_r1a')  ] = 0 # Assignment is removed
            next_task = 'b' if self.config_type not in ['n_system', 'parallel', 'single_activity'] else None
            if next_task is not None:
                observation[self.env.state_space.index(f'waiting_{next_task}')] += 1 # New task is generated
            return observation
        elif evolution == 'r1b':
            observation[self.env.state_space.index('is_available_r1')] = 1
            if self.config_type not in ['n_system', 'single_activity']:
                observation[self.env.state_space.index('assigned_r1b')] = 0
            return observation
        elif evolution == 'r2a':
            observation[self.env.state_space.index('is_available_r2')] = 1 # Resource becomes available
            if self.config_type != 'single_activity':
                observation[self.env.state_space.index('assigned_r2a')  ] = 0 # Assignment is removed
            next_task = 'b' if self.config_type not in ['n_system', 'parallel', 'single_activity'] else None
            if next_task is not None:
                observation[self.env.state_space.index(f'waiting_{next_task}')] += 1 # New task is generated
            return observation
        elif evolution == 'r2b':
            observation[self.env.state_space.index('is_available_r2')] = 1
            if self.config_type != 'single_activity':
                observation[self.env.state_space.index('assigned_r2b')] = 0
            return observation


    def value_iteration(self, max_iterations=0):
        # All states, so we can use the state_to_index function
        q = np.full((len(self.all_states), len(self.env.action_space)), 0.0)
        v = np.full(len(self.all_states), 0.0)
        policy = np.full(len(self.all_states), -1) 
        old_delta = np.inf
        delta_delta = np.inf

        iteration = 0
        delta = 0
        done = False
        state = []
        while not done:
            print(f'Iteration {iteration} with delta {delta} and delta_delta {delta_delta}')
            
            delta = 0
            for s in tqdm(self.state_space, total=len(self.state_space), disable=True):
                self.set_state_from_observation(self.env, s)
                action_mask = self.env.action_mask()

                if self.config_type != 'parallel':
                    if self.config_type != 'single_activity':                        
                        nr_active_cases = len(list(self.env.processing_r2 + self.env.processing_r2 + self.env.waiting_cases['a'] + self.env.waiting_cases['b']))
                    else:
                        nr_active_cases = len(list(self.env.processing_r1 + self.env.processing_r2 + self.env.waiting_cases['a']))
                else:
                    processing_type_a = [case for case in self.env.processing_r1 if case[1] == 'a'] + [case for case in self.env.processing_r2 if case[1] == 'a']
                    processing_type_b = [case for case in self.env.processing_r1 if case[1] == 'b'] + [case for case in self.env.processing_r2 if case[1] == 'b']
                    nr_active_cases = max(len(list(processing_type_a + self.env.waiting_cases['a'])), len(list(processing_type_b + self.env.waiting_cases['b'])))

                step_reward = -nr_active_cases * self.tau

                for i, mask in enumerate(action_mask):
                    if mask == 1:
                        action = [0 for _ in range(len(action_mask))]
                        action[i] = 1
                        action = tuple(action)
                        action_name = self.env.action_space[i]

                        evolutions = []

                        evolution_probabilities, _ = self.get_transformed_evolutions(self.env.processing_r1, self.env.processing_r2, arrivals_coming=True, action=action_name)
                        for evolution, probability in evolution_probabilities.items():
                            if evolution == 'return_to_state':
                                next_observation = s.copy()
                            else:
                                observation = s.copy()
                                observation = self.process_observation(observation, action_name)
                                next_observation = self.evolve_observation(observation, evolution)                   
                            
                            if type(next_observation) != tuple:
                                next_observation, penalty = self.clip_observation(next_observation)
                                reward = step_reward + penalty
                                evolutions.append((next_observation, probability, reward))
                            else:
                                for next_obs in next_observation:
                                    next_obs, penalty = self.clip_observation(next_obs)
                                    reward = step_reward + penalty
                                    evolutions.append((next_obs, probability * 0.5, reward)) # n_system case, so arrival can happen at both activities

                        sum = 0
                        for next_observation, probability, reward in evolutions:
                            sum += probability * (reward + self.gamma * v[self.observation_to_index(next_observation)])
                        q[self.observation_to_index(s), i] = sum
 
                
                old_value = v[self.observation_to_index(s)]
                new_value = max([q[self.observation_to_index(s), i] for i, mask in enumerate(action_mask) if mask == 1])
                delta = max(delta, abs(new_value - old_value))

                v[self.observation_to_index(s)] = new_value
                policy[self.observation_to_index(s)] = np.argmax([q[self.observation_to_index(s), i] if mask == 1 else -np.inf for i, mask in enumerate(action_mask)])
                
            delta_delta = abs(old_delta - delta)
            old_delta = delta
            iteration += 1
            if max_iterations != 0:
                if iteration == max_iterations:
                    done = True
            else:
                if self.gamma == 1:
                    if delta_delta < self.theta:
                        done = True
                elif delta < self.theta:
                    done = True

        return q, v, policy
    
def main():
    average_step_time_smdp = {
    'slow_server': 0.6700290812922858,
    'low_utilization': 0.6667579762550317,
    'high_utilization': 0.6695359811479276,
    'n_system': 1.0014597219587165,
    'parallel': 0.6680392125284501,
    'down_stream': 0.6681612256898539,
    'single_activity': 1.0042253394472318
    }
    for config in ['slow_server', 'low_utilization', 'high_utilization']:	
        tau = average_step_time_smdp[config] / 2
        env = mdp.MDP(2500, config, tau)

        gamma = 1
        theta = 0.001
        vi = ValueIteration(env, gamma=gamma,theta=theta)
        q, v, policy = vi.value_iteration()
        directory = 'models/vi'
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(f'{directory}/{config}.npy', policy)


    """
        Observation examples:
        single_activity: [0, 0, 12]   ---   two resources and one queue

        n_system: [0, 1, 0, 0, 10, 12]   ---   resource 1 can only process activity B, so no assigned_ features
                                         ---   resource 2 can process both activities, so two assigned_ features

        others: [0, 0, 1, 0, 0, 1, 10, 12]   ---   both resources can process both activities
                                             ---   r1 is assigned to activity A, r2 is assigned to activity B
    """


    # observation = [0, 1, 1, 0, 0, 0, 0, 9]
    # print('pre observation', observation)
    # vi.set_state_from_observation(env, observation)
    # print('post observation', env.observation())
    # print(env.processing_r1, env.processing_r2)

    # #print(vi.clip_observation(observation))
    # print(vi.env.action_space[3])
    # print(vi.get_transformed_evolutions(env.processing_r1, env.processing_r2, arrivals_coming=True, action=vi.env.action_space[3])) # r2b
    # print('\n')
if __name__ == '__main__':

    main()
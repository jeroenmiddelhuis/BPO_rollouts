import mdp, mdp_composite
import numpy as np
import itertools
from tqdm import tqdm
import os
import sys

class ValueIteration:
    def __init__(self, env, gamma=1, theta=0.0001):
        self.env = env
        self.tau = env.tau
        self.config_type = self.env.config_type
        self.gamma = gamma
        self.theta = theta
        self.max_queue = 100
        self.out_of_bounds_penalty = -100000

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
        print(feature_ranges)
        all_states = [list(state) for state in itertools.product(*[range(i) for i in feature_ranges])]
        print('all states', len(all_states))
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
        assert len(observation) == len(self.env.state_space), f"Observation ({len(observation)}) and state space ({len(self.env.state_space)}) length must match"

        for resource in self.env.resources:
            # Some features do not contain assigned_ features if they can only be assigned to one task
            for state in self.env.state_space:
                if f'assigned_{resource}' in state:
                    assert sum([1 for i, x in enumerate(observation) if x == 1 and resource in self.env.state_space[i]]) == 1

        # All values should be positive
        assert all([x >= 0 for x in observation])

        # The values not related to the queue should be 0 or 1
        for i, x in enumerate(observation):
            if 'waiting' not in self.env.state_space[i]:
                assert x == 0 or x == 1

    def set_state_from_observation(self, env, observation):
        state_space_labels = env.state_space
        waiting_cases = {task_type: [] for task_type in env.task_types}
        processing_resources = []
        total_time = 0 # Not relevant for state transitions
        total_arrivals = 0 # Not relevant for state transitions
        nr_arrivals = 1000 # Not relevant for state transitions, but should be different than total_arrivals

        # Set waiting cases
        for task_type in self.env.task_types:
            nr_cases = observation[state_space_labels.index(f'waiting_{task_type}')]
            already_waiting_cases = sum([len(waiting_cases[task_type]) for task_type in self.env.task_types])
            waiting_cases[task_type] = [i + already_waiting_cases for i in range(nr_cases)]
        # Set processing resources
        for resource in self.env.resources:
            # The resource is available
            if observation[state_space_labels.index(f'is_available_{resource}')] == 1: # The resource is available
                processing_resources.append([])
            # The resource is busy and there is an assigned resource variable. Check which task is assigned
            elif any(label.startswith(f'assigned_{resource}') for label in state_space_labels):
                for task_type in self.env.task_types:
                    if f'assigned_{resource}{task_type}' in state_space_labels and observation[state_space_labels.index(f'assigned_{resource}{task_type}')] == 1:
                        processing_resources.append([(sum([len(waiting_cases[task_type]) for task_type in self.env.task_types]) 
                                                      + sum([len(processing_resource) for processing_resource in processing_resources]), 
                                                      task_type)])
            else: # No assigned resource variable, so we need to check which task is assigned
                for task_type in self.env.task_types:
                    resource_pool = self.env.resource_pools[task_type]
                    # Check if the resource is in the resource pool and if it is the only resource processing the task                    
                    if resource in resource_pool and sum([resource in self.env.resource_pools[task_type2] for task_type2 in self.env.task_types if task_type2 != task_type]) == 0:
                        processing_resources.append([(sum([len(waiting_cases[task_type]) for task_type in self.env.task_types]) 
                                                      + sum([len(processing_resource) for processing_resource in processing_resources]), 
                                                      task_type)])
        env.set_state((waiting_cases, processing_resources, total_time, total_arrivals, nr_arrivals, None))

    def observation(self):
        return self.env.observation()
    
    def get_transformed_evolutions(self, processing_resources, arrivals_coming=True, action=None):
        return self.env.get_transformed_evolutions(processing_resources, arrivals_coming, action)

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

    def process_observation(self, observation, actions):
        if actions not in ['postpone', 'do_nothing']:
            if isinstance(actions, str):
                actions = [actions]
            for action in actions:
                resource, task = action[:-1], action[-1]
                observation[self.env.state_space.index(f'is_available_{resource}')] = 0
                if f'assigned_{resource}{task}' in self.env.state_space:
                    observation[self.env.state_space.index(f'assigned_{resource}{task}')] = 1
                observation[self.env.state_space.index(f'waiting_{task}')] -= 1
        return observation

    def evolve_observation(self, observation, evolution):
        observation = observation.copy()
        if evolution == 'return_to_state':
            return observation
        elif evolution == 'arrival':
            next_states = []
            transition_matrix = self.env.transitions['Start']
            next_observation = observation.copy()
            for i, probability in enumerate(transition_matrix[:-1]): # Excluding 'Complete' event
                if probability > 0:                    
                    next_task = self.env.task_types_all[i]
                    next_observation[self.env.state_space.index(f'waiting_{next_task}')] += 1
                    if sum(transition_matrix) <= 1: # If there are no parallel tasks, add the state to the next_states
                        next_states.append((next_observation, probability))
                        next_observation = observation.copy()
            if len(next_states) == 0: # Append parallel tasks
                next_states.append((next_observation, 1))
            return next_states
        else:
            resource, task = evolution[:-1], evolution[-1]
            # Process the completion of the assignment
            observation[self.env.state_space.index(f'is_available_{resource}')] = 1
            if f'assigned_{resource}{task}' in self.env.state_space:
                observation[self.env.state_space.index(f'assigned_{resource}{task}')] = 0
            
            # Generate new task due to completion of the assignment
            next_states = []
            transition_matrix = self.env.transitions[task]
            next_observation = observation.copy()
            for i, probability in enumerate(transition_matrix[:-1]): # Excluding 'Complete' event
                if probability > 0:                    
                    next_task = self.env.task_types_all[i]
                    next_observation[self.env.state_space.index(f'waiting_{next_task}')] += 1
                    if sum(transition_matrix) <= 1: # If there are no parallel tasks, add the state to the next_states
                        next_states.append((next_observation, probability))
                        next_observation = observation.copy()
            if len(next_states) == 0: # Append parallel tasks
                next_states.append((next_observation, 1))
            return next_states

    def value_iteration(self, max_iterations=0):
        # All states, so we can use the state_to_index function
        q = np.full((len(self.all_states), len(self.env.action_space)), 0.0)
        v = np.full(len(self.all_states), 0.0)
        policy = np.full(len(self.all_states), -1) 
        old_delta = np.inf
        delta_delta = np.inf

        iteration = 0
        #delta = 0
        max_delta = -np.inf
        min_delta = np.inf
        done = False
        while not done:
            #print(f'Iteration {iteration} with delta {delta} and delta_delta {delta_delta}')
            print(f'Iteration {iteration} for {self.config_type}', flush=True)
            print(f'Max_delta {np.round(max_delta, 2)} and min_delta {np.round(min_delta,2)}', flush=True)
            print(f'Difference: {np.round(max_delta - min_delta, 10)}', flush=True)
            delta = 0
            max_delta = -np.inf
            min_delta = np.inf
            for s in tqdm(self.state_space, total=len(self.state_space), disable=False):
                self.set_state_from_observation(self.env, s)
                action_mask = self.env.action_mask()

                # Create set of unique active cases
                unique_active_cases = {getattr(self.env, f'processing_r{i}')[0][0] 
                                    for i in range(1, len(self.env.resources)+1) 
                                    if len(getattr(self.env, f'processing_r{i}')) > 0}            
                for task in self.env.task_types:
                    unique_active_cases.update(self.env.waiting_cases[task])

                nr_active_cases = len(unique_active_cases)

                # Correct for parallel cases
                if self.config_type == 'parallel':
                    cases_a = len([case for case in self.env.processing_r1 if case[1] == 'a']) + len([case for case in self.env.processing_r2 if case[1] == 'a']) + len(self.env.waiting_cases['a'])
                    cases_b = len([case for case in self.env.processing_r1 if case[1] == 'b']) + len([case for case in self.env.processing_r2 if case[1] == 'b']) + len(self.env.waiting_cases['b'])
                    # Each cases is assigned a different id, but we correct here for cases that are in parallel
                    # For example state [0, 1, 1, 0, 0, 0, 10, 15] has 26 active tasks, but only 15 unique cases
                    nr_active_cases -=  min(cases_a, cases_b)
                elif self.config_type == 'composite':
                    cases_k = len([case for case in self.env.processing_r11 if case[1] == 'k']) + len([case for case in self.env.processing_r12 if case[1] == 'k']) + len(self.env.waiting_cases['k'])
                    cases_l = len([case for case in self.env.processing_r11 if case[1] == 'l']) + len([case for case in self.env.processing_r12 if case[1] == 'l']) + len(self.env.waiting_cases['l'])
                    nr_active_cases -=  min(cases_k, cases_l)
  
                step_reward = -nr_active_cases * self.tau

                for i, mask in enumerate(action_mask):
                    if mask == 1:
                        action = self.env.action_space[i]

                        next_states = []

                        processing_resources = [getattr(self.env, f'processing_r{i}') for i in range(1, len(self.env.resources)+1)]
                        evolution_probabilities, _ = self.get_transformed_evolutions(processing_resources, arrivals_coming=True, action=action)
                        for evolution, probability in evolution_probabilities.items():
                            if evolution == 'return_to_state':
                                next_observation = s.copy()
                                next_states.append((next_observation, probability, step_reward))
                            else:
                                observation = s.copy()
                                observation = self.process_observation(observation, action)
                                next_observations = self.evolve_observation(observation, evolution)

                                for next_observation, next_state_p in next_observations: # Some evolutions may lead to multiple states
                                    next_observation, penalty = self.clip_observation(next_observation)
                                    reward = step_reward + penalty
                                    next_states.append((next_observation, probability * next_state_p, reward)) # Scale the evolution probability with the state probability

                        sum = 0
                        for next_observation, probability, reward in next_states:
                            sum += probability * (reward + self.gamma * v[self.observation_to_index(next_observation)])
                        q[self.observation_to_index(s), i] = sum
 
                
                old_value = v[self.observation_to_index(s)]
                new_value = max([q[self.observation_to_index(s), i] for i, mask in enumerate(action_mask) if mask == 1])
                delta = max(delta, abs(new_value - old_value))
                max_delta = max(max_delta, abs(new_value - old_value))
                min_delta = min(min_delta, abs(new_value - old_value))
                #print(min_delta, max_delta)

                v[self.observation_to_index(s)] = new_value
                policy[self.observation_to_index(s)] = np.argmax([q[self.observation_to_index(s), i] if mask == 1 else -np.inf for i, mask in enumerate(action_mask)])
                
            # delta_delta = abs(old_delta - delta)
            # old_delta = delta
            iteration += 1
            if max_iterations != 0:
                if iteration == max_iterations:
                    done = True
            else:
                if self.gamma == 1:
                    if max_delta - min_delta < self.theta:
                        done = True
                    # if delta_delta < self.theta:
                    #     done = True
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
    'single_activity': 1.0042253394472318,
    'composite': 0.1699383116459651,
    }

    minimium_transition_time = {
        'slow_server': 1 / (1/2.0 + 1/1.4 + 1/1.8),
        'low_utilization': 1 / (1/2.0 + 1/1.4 + 1/1.4),
        'high_utilization': 1 / (1/2.0 + 1/1.8 + 1/1.8),
        'n_system': 1 / (1/2.0 + 1/2.0 + 1/3),
        'parallel': 1 / (1/2.0 + 1/1.6 + 1/1.6),
        'down_stream': 1 / (1/2.0 + 1/1.6 + 1/1.6),
        'single_activity': 1/ (1/2.0 + 1/1.8 + 1/10.0)
    }

    tau_multiplier = 0.5

    configs = [sys.argv[1]]
    for config in configs:
    #for config in ['high_utilization', 'parallel', 'down_stream']:    
        tau = minimium_transition_time[config] * tau_multiplier
        if config == 'composite':
            env = mdp_composite.MDP_composite(2500, config, tau)
        else:
            env = mdp.MDP(2500, config, tau)

        gamma = 1
        theta = 0.001
        vi = ValueIteration(env, gamma=gamma, theta=theta)
        q, v, policy = vi.value_iteration()
        directory = f'models/vi/{config}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(f'{directory}/{config}_q.npy', q)
        np.save(f'{directory}/{config}_v.npy', v)
        np.save(f'{directory}/{config}_policy.npy', policy)

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
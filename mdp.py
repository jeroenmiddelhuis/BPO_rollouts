from reporters import EventLogReporter, ProcessReporter
import sys, os, json
import numpy as np
import random

PRINT_TRAJECTORY = False

class MDP:
    def __init__(self, nr_arrivals, config_type='single_activity', tau=0.5, reporter=None, reward_function='AUC'):
        """
        For now, we just implement the simple SMDP with:
        one task (A), two resources (r1, r2), case arrival rate lambda = 0.5, 
        exponential processing rate mu_(r1, a) = 1/1.8, exponential processing rate mu_(r2, a) = 1/10.
        state = (available r1, available r2, active r1, active r2, waiting a)
        action = one of (r1, a), (r2, a), postpone, do nothing
        For efficiency, we encode the action one-hot here already, so the action is a list of 4 elements, only one of them can be 1.
        We implement as a gym environment, so we need the following functions:
        reset, step, action_mask, is_done
        We use a common random numbers generator, which allows us to do repeat a rollout with the same random numbers. This has been shown to help with learning and is usful for testing.
        """
        # Read the config file and set the process parameters
        self.config_type = config_type
        self.reward_function = reward_function
        self.env_type = 'mdp'

        with open(os.path.join(sys.path[0], "config.txt"), "r") as f:
            data = f.read()

        config = json.loads(data)
        config = config[config_type]
        
        self.task_types = [task for task in list(config['task_types']) if task != 'Start']
        self.task_types_all = [task for task in list(config['task_types'])] + ['Complete']
        self.resources = sorted(list(config['resources']))
        self.resource_pools = config['resource_pools']

        self.arrival_rate = 1/config['mean_interarrival_time']

        self.transitions = config['transitions']


        if self.config_type == 'n_system':
            # r2 can process both activities, but r1 can only process B. 
            # Therefore, we don't need the assigned_ features for r1
            self.state_space =  ([f'is_available_{r}' for r in self.resources] +
                                [f'assigned_r2{task_type}' for task_type in self.task_types] +
                                [f'waiting_{task}' for task in self.task_types])
        elif self.config_type == 'single_activity':
            # only one activity, so no need for assigned_ features
            self.state_space =  ([f'is_available_{r}' for r in self.resources] +
                                [f'waiting_{task}' for task in self.task_types])
        else:
            self.state_space =  ([f'is_available_{r}' for r in self.resources] +
                                [f'assigned_{r}{task_type}' for r in self.resources for task_type in self.task_types] +
                                [f'waiting_{task}' for task in self.task_types])
   
        self.action_space = [f"{resource}{task}" for resource in self.resources for task in self.task_types if resource in self.resource_pools[task] and task != 'Start'] + ['postpone', 'do_nothing']
        self.waiting_cases = {task: [] for task in self.task_types}
        self.partially_completed_cases = []

        self.processing_r1 = []
        self.processing_r2 = []
        self.total_time = 0
        self.total_arrivals = 0
        self.nr_arrivals = nr_arrivals
        self.original_nr_arrivals = nr_arrivals
        self.tau = tau
        self.locked_action = None
        self.arrival_times = {}
        self.cycle_times = {}
        self.actions_taken = {}
        self.actual_actions_taken = {}
        self.episodic_reward = 0

        self.reporter = reporter

    def observation(self):
        is_available_r1 = len(self.processing_r1) == 0 # True if there is a case being processed by r1
        is_available_r2 = len(self.processing_r2) == 0
        if self.config_type == 'single_activity': # single activity
            waiting_cases = [len(self.waiting_cases.get(task, [])) for task in self.task_types if task != "Start"]
            return [is_available_r1, is_available_r2] + waiting_cases
        else: # Other scenarios
            assigned_r1a = 1 if not is_available_r1 and self.processing_r1[-1][1] == 'a' else 0
            assigned_r1b = 1 if not is_available_r1 and self.processing_r1[-1][1] == 'b' else 0
            assigned_r2a = 1 if not is_available_r2 and self.processing_r2[-1][1] == 'a' else 0
            assigned_r2b = 1 if not is_available_r2 and self.processing_r2[-1][1] == 'b' else 0
            waiting_cases = [len(self.waiting_cases.get(task, [])) for task in self.task_types if task != "Start"]
            if self.config_type != 'n_system':
                return [is_available_r1, is_available_r2, assigned_r1a, assigned_r1b, assigned_r2a, assigned_r2b] + waiting_cases
            else:
                return [is_available_r1, is_available_r2, assigned_r2a, assigned_r2b] + waiting_cases
        
    def action_mask(self):
        # single activity
        if len(self.task_types) == 1: 
            r1_available = len(self.processing_r1) == 0
            r2_available = len(self.processing_r2) == 0
            a_waiting = len(self.waiting_cases['a']) > 0
            r1a_possible = a_waiting and r1_available and 'r1' in self.resource_pools['a']
            r2a_possible = a_waiting and r2_available and 'r2' in self.resource_pools['a']
            if self.arrivals_coming():
                # If all assignments are possible, postpone is not allowed
                postpone_possible = 1 <= sum([r1a_possible, r2a_possible]) < 2 
            else:
                # If both resources are available and no more cases are arriving, postpone is not allowed
                postpone_possible = r1_available != r2_available and a_waiting
            do_nothing_possible = sum([r1a_possible, r2a_possible, postpone_possible]) == 0
            return [r1a_possible, r2a_possible, postpone_possible, do_nothing_possible]
        else: # Other scenarios
            r1_available = len(self.processing_r1) == 0
            r2_available = len(self.processing_r2) == 0
            a_waiting = len(self.waiting_cases['a']) > 0
            b_waiting = len(self.waiting_cases['b']) > 0

            r1a_possible = a_waiting and r1_available and 'r1' in self.resource_pools['a']
            r1b_possible = b_waiting and r1_available and 'r1' in self.resource_pools['b']
            r2a_possible = a_waiting and r2_available and 'r2' in self.resource_pools['a']
            r2b_possible = b_waiting and r2_available and 'r2' in self.resource_pools['b']

            if self.arrivals_coming():
                if self.config_type not in ['parallel', 'n_system']:
                    # If there are cases waiting at activity A and both resources are avaible, postpone is not allowed
                    postpone_possible = (not ((r1a_possible and r2a_possible) and not b_waiting) and # If there are no cases at B, postpone is not allowed
                                        1 <= sum([r1a_possible, r1b_possible, r2a_possible, r2b_possible]) < 4)
                else:
                    # If all actions are possible, postpone is not allowed
                    postpone_possible = 1 <= sum([r1a_possible, r1b_possible, r2a_possible, r2b_possible]) < 4
            else:
                # If both resources are available and no more cases are arriving, postpone is not allowed
                # If both resources are not available, postpone is not allowed. Instead do nothing is allowed
                postpone_possible = (r1_available != r2_available) and (a_waiting or b_waiting)

            do_nothing_possible = sum([r1a_possible, r1b_possible, r2a_possible, r2b_possible, postpone_possible]) == 0

            if self.config_type != 'n_system':
                return [r1a_possible, r1b_possible, r2a_possible, r2b_possible, postpone_possible, do_nothing_possible]
            else:
                return [r1b_possible, r2a_possible, r2b_possible, postpone_possible, do_nothing_possible]

    def reset(self):
        self.waiting_cases = {task: [] for task in self.task_types}
        self.partially_completed_cases = []
        self.processing_r1 = []
        self.processing_r2 = []
        self.total_time = 0
        self.total_arrivals = 0
        self.nr_arrivals = self.original_nr_arrivals
        self.locked_action = None
        self.arrival_times = {}
        self.cycle_times = {}
        #print(self.episodic_reward)
        self.episodic_reward = 0
    
    def get_state(self, rollout_length=None):
        nr_arrivals = self.nr_arrivals if rollout_length is None else self.nr_arrivals + rollout_length
        return ({task: cases.copy() for task, cases in self.waiting_cases.items()},
                self.processing_r1.copy(),
                self.processing_r2.copy(),
                self.total_time,
                self.total_arrivals,
                nr_arrivals,
                self.partially_completed_cases.copy())
    
    def set_state(self, state):
        self.waiting_cases = {task: cases.copy() for task, cases in state[0].items()}
        self.processing_r1 = state[1].copy()
        self.processing_r2 = state[2].copy()
        self.total_time = state[3]
        self.total_arrivals = state[4]
        self.nr_arrivals = state[5]
        # For parallel systems, we need to keep track of the partially completed cases
        if state[6] is not None:
            self.partially_completed_cases = state[6].copy()
        self.locked_action = None

    def from_state(state):
        mdp = MDP(0)
        mdp.set_state(state)
        return mdp

    def sample_next_task(self, current_task, case_id=None):
        # Calculate the sum of the values
        p_transitions = self.transitions[current_task]
        total_sum = sum(p_transitions)
        if self.config_type != 'parallel':
            # Check if the sum is approximately 1 or less, considering rounding errors
            if np.isclose(total_sum, 1, atol=1e-9) or total_sum < 1:
                # Draw a random sample using the crn class
                return [np.random.choice(self.task_types_all, p=p_transitions)]
            else:
                raise ValueError("The sum of the transition probabilities must be 1 or less.")

        elif self.config_type == 'parallel':
            if current_task == 'Start':
                # Return the indices of the list that contain a nonzero value
                nonzero_indices = [index for index, value in enumerate(p_transitions) if value > 0]
                return [self.task_types_all[index] for index in nonzero_indices]
            elif current_task != 'Complete':
                if case_id in self.partially_completed_cases:
                    return ['Complete']
        return []
                
    def get_evolution_rates(self, processing_r1, processing_r2, arrivals_coming, action=None):
        """
        Returns a dictionary with the possible evolutions and their rates.
        For the MDP, the selected action is used to determine the possible evolutions.
        """
        r1_processing = len(processing_r1) > 0
        r2_processing = len(processing_r2) > 0

        evolution_rates = {key: value for key, value in {
            'arrival': self.arrival_rate if arrivals_coming else 0,
            'r1a': 1/self.resource_pools['a']['r1'][0] if r1_processing and processing_r1[0][1] == 'a' else 0,
            'r2a': 1/self.resource_pools['a']['r2'][0] if r2_processing and processing_r2[0][1] == 'a' else 0,
            'r1b': 1/self.resource_pools['b']['r1'][0] if r1_processing and processing_r1[0][1] == 'b' else 0,
            'r2b': 1/self.resource_pools['b']['r2'][0] if r2_processing and processing_r2[0][1] == 'b' else 0
        }.items() if value > 0}

        # add possible evolution based on action
        if action == 'r1a': # (r1, a)
            evolution_rates['r1a'] = 1/self.resource_pools['a']['r1'][0]
        elif action == 'r2a': # (r2, a)
            evolution_rates['r2a'] = 1/self.resource_pools['a']['r2'][0]
        elif action == 'r1b': # (r1, b)
            evolution_rates['r1b'] = 1/self.resource_pools['b']['r1'][0]
        elif action == 'r2b': # (r2, b)
            evolution_rates['r2b'] = 1/self.resource_pools['b']['r2'][0]

        return evolution_rates

    def get_evolutions(self, processing_r1, processing_r2, arrivals_coming, action=None):
        evolution_rates = self.get_evolution_rates(processing_r1, processing_r2, arrivals_coming, action)
        sum_of_rates = sum(evolution_rates.values())
        evolutions = {}
        for evolution, rate in evolution_rates.items():
            evolutions[evolution] = rate/sum_of_rates
        return evolutions, evolution_rates

    def get_transformed_evolutions(self, processing_r1, processing_r2, arrivals_coming, action=None):
        evolutions, evolution_rates = self.get_evolutions(processing_r1, processing_r2, arrivals_coming, action)
        sum_of_rates = sum(evolution_rates.values())

        transformed_evolutions = {}
        expected_next_event_time = 1 / sum_of_rates
        for evolution, p in evolutions.items():
            transformed_evolutions[evolution] = self.tau / expected_next_event_time * p
        transformed_evolutions['return_to_state'] = 1 - self.tau / expected_next_event_time
        #print('transformed_evolutions', transformed_evolutions, sum(transformed_evolutions.values()), '\n')
        return transformed_evolutions, evolution_rates

    def arrivals_coming(self):
        return 1 if self.total_arrivals < self.nr_arrivals else 0
    
    def is_done(self):
        """
        The simulation is done if we have reached the maximum number of arrivals and there are no more tasks to process.
        """
        return not self.arrivals_coming() and sum(len(v) for v in self.waiting_cases.values()) == 0 and len(self.processing_r1) == 0 and len(self.processing_r2) == 0

    def step(self, action):
        original_action = action
        reward = 0

        action_index = action.index(1)
        action = self.action_space[action_index]
        if action not in self.actions_taken:
            self.actions_taken[action] = 0
            self.actual_actions_taken[action] = 0
        self.actions_taken[action] += 1
        transformed_evolutions, evolution_rates = self.get_transformed_evolutions(self.processing_r1, self.processing_r2, self.arrivals_coming(), action)
        
        if self.reward_function == 'AUC':
            if len(self.processing_r1) > 0:
                processing_r1_case = [self.processing_r1[0][0]]
            else:
                processing_r1_case = []
            if len(self.processing_r2) > 0:
                processing_r2_case = [self.processing_r2[0][0]]
            else:
                processing_r2_case = []
            if self.config_type != 'single_activity':
                unique_active_cases = list(set(processing_r1_case + processing_r2_case + self.waiting_cases['a'] + self.waiting_cases['b']))
            else:
                unique_active_cases = list(set(processing_r1_case + processing_r2_case + self.waiting_cases['a']))

            reward += -len(unique_active_cases) * self.tau # Not needed to multiply by constant tau but kept for consistency with SMDP

        events, probs = zip(*list(transformed_evolutions.items()))
        evolution = np.random.choice(events, p=probs)
        if evolution != 'return_to_state':
            next_task = None
            self.locked_action = None
            #print('Action unlocked', self.locked_action)
            # process the action, 'postpone' and 'do nothing', do nothing to the state.
            if action in ['r1a', 'r2a', 'r1b', 'r2b']:
                resource, task = action[0:2], action[2]
                if len(self.waiting_cases[task]) > 0:
                    case_id = self.waiting_cases[task].pop(0)
                    getattr(self, f'processing_{resource}').append((case_id, task))
                    if self.reporter:
                        self.reporter.callback(case_id, task, '<task:start>', self.total_time, resource)
            
            # the total time changes before the evolution happens
            # it changes after processing the action
            self.total_time += self.tau

            # now calculate the next state and how long it takes to reach that state
            # the time is to the next state is exponentially distributed with rate 
            # min(lambda, mu_(r1, a) * active r1, mu_(r2, a) * active r2) = 
            # exponential(0.5 + 1/1.8 * len(self.nr_processing_r1) + 1/10 * len(self.nr_processing_r2))
            # this is the probability of the next state being the consequence of:
            # an arrival, r1 processing the task or r2 processing the task
            # the probability of one of these evolutions happening is proportional to the rate of that evolution.
            if evolution == 'arrival':
                self.arrival_times[self.total_arrivals] = self.total_time
                # sample the first task from the transition matrix
                next_tasks = self.sample_next_task('Start')
                for task in next_tasks:
                    self.waiting_cases[task].append(self.total_arrivals)
                if self.reporter is not None:
                    self.reporter.callback(self.total_arrivals, 'start', '<start_event>', self.total_time)
                self.total_arrivals += 1
            else:
                resource, task = evolution[0:2], evolution[2]                
                case_id = getattr(self, f'processing_{resource}').pop(0)[0]
                next_tasks = self.sample_next_task(task, case_id)
                self.partially_completed_cases.append(case_id)
                if self.reporter:
                    self.reporter.callback(case_id, task, '<task:complete>', self.total_time, resource)
                for next_task in next_tasks:
                    if next_task and next_task != 'Complete':
                        self.waiting_cases[next_task].append(case_id)
                    elif next_task == 'Complete':                        
                        if self.reward_function == 'case_cycle_time':
                            self.cycle_times[case_id] = self.total_time - self.arrival_times[case_id]
                            reward += -self.cycle_times[case_id]
                        elif self.reward_function == 'inverse_case_cycle_time':
                            self.cycle_times[case_id] = self.total_time - self.arrival_times[case_id]
                            reward += 1/(1 + self.cycle_times[case_id])
                        if self.reporter:
                            self.reporter.callback(case_id, 'complete', '<end_event>', self.total_time)
            self.episodic_reward += reward
            return self.observation(), reward, self.is_done(), False, None
        else:
            #print('Action locked:', original_action)
            #print('Final state', self.observation(), '\n')
            self.locked_action = original_action
            self.total_time += self.tau
            self.episodic_reward += reward
            return self.observation(), reward, self.is_done(), False, None

def random_policy(env):
    action_mask = env.action_mask()
    action = [0] * len(action_mask)
    choices = [i for i in range(len(action_mask)) if action_mask[i] and env.action_space[i] not in ['postpone', 'do_nothing']]
    if len(choices) == 0:
        possible_action = 'postpone' if action_mask[env.action_space.index('postpone')] else 'do_nothing'
        action_index = env.action_space.index(possible_action)
        action[action_index] = 1
        return action
    else:
        action_index = np.random.choice(choices)
        action[action_index] = 1
        return action

def totally_random_policy(env):
    """
    Testing policy which also takes the postpone and do nothing action randomly
    """
    action_mask = env.action_mask()
    action = [0] * len(action_mask)
    choices = [i for i in range(len(action_mask)) if action_mask[i] and env.action_space[i]]
    action_index = np.random.choice(choices)
    action[action_index] = 1
    return action

def greedy_policy(env):
    action_mask = env.action_mask()
    action = [0] * len(action_mask)
    
    min_processing_time = float('inf')
    best_action_index = []
    possible_actions = []
    for i, possible in enumerate(action_mask):
        if possible:
            action_str = env.action_space[i]
            possible_actions.append(action_str)
            if action_str in ['postpone', 'do_nothing']:
                continue
            resource, task = action_str[0:-1], action_str[-1]
            processing_time = env.resource_pools[task][resource][0]
            if processing_time < min_processing_time:
                min_processing_time = processing_time
                best_action_index = [i]
            elif processing_time == min_processing_time:
                best_action_index.append(i)
    
    if len(best_action_index) == 1:
        action[best_action_index[0]] = 1
    elif len(best_action_index) > 1:
        action_index = np.random.choice(best_action_index)
        action[action_index] = 1
    else:
        # If no valid action found, default to 'do_nothing' or 'postpone' if available
        if action_mask[env.action_space.index('postpone')] == 1:
            action[env.action_space.index('postpone')] = 1
        elif action_mask[env.action_space.index('do_nothing')] == 1:
            action[env.action_space.index('do_nothing')] = 1      
    return action

def fifo_policy(env):
    action_mask = env.action_mask()
    possible_actions = [env.action_space[i] for i, action in enumerate(action_mask) if action]
    #print(action_mask)
    action = [0] * len(action_mask)
    #print(env.waiting_cases['a'], env.waiting_cases['b'])
    if len(possible_actions) == 1:
        # Check if 'postpone' is feasible
        postpone_index = env.action_space.index('postpone')
        if action_mask[postpone_index]:
            action[postpone_index] = 1
            return action
        
        do_nothing_index = env.action_space.index('do_nothing')
        if action_mask[do_nothing_index]:
            action[do_nothing_index] = 1
            return action

    # Identify the case that has been in the system the longest
    longest_waiting_case_a = min(env.waiting_cases['a']) if len(env.waiting_cases['a']) > 0 else None
    if 'b' in env.waiting_cases:
        longest_waiting_case_b = min(env.waiting_cases['b']) if len(env.waiting_cases['b']) > 0 else None
    else:
        longest_waiting_case_b = None
    #print('longest_case', longest_waiting_case_a, longest_waiting_case_b)
    if longest_waiting_case_a is not None and longest_waiting_case_b is not None:
        longest_waiting_case_type = 'a' if longest_waiting_case_a < longest_waiting_case_b else 'b'
    elif longest_waiting_case_a is not None:
        longest_waiting_case_type = 'a'
    elif longest_waiting_case_b is not None:
        longest_waiting_case_type = 'b'
    
    
    #print(env.waiting_cases, possible_actions)
    feasible_actions = [resource + longest_waiting_case_type for resource in ['r1', 'r2'] if resource + longest_waiting_case_type in possible_actions]
    if len(feasible_actions) == 1:
        index = env.action_space.index(feasible_actions[0])
        action[index] = 1
        return action
    elif len(feasible_actions) > 1:
        index = env.action_space.index(np.random.choice(feasible_actions))
        action[index] = 1
        return action
    
    other_case_type = 'a' if longest_waiting_case_type == 'b' else 'b'
    if len(env.waiting_cases[other_case_type]) > 0:
        feasible_actions = [resource + other_case_type for resource in ['r1', 'r2'] if resource + other_case_type in possible_actions]
        if len(feasible_actions) == 1:
            index = env.action_space.index(feasible_actions[0])
            action[index] = 1
            return action
        elif len(feasible_actions) > 1:
            index = env.action_space.index(np.random.choice(feasible_actions))
            action[index] = 1
            return action
    return action

def epsilon_greedy_policy(env):
    if np.random.uniform() < 0.25:
        return env.random_policy()
    else:
        return env.greedy_policy()

def threshold_policy(env, observation=None, action_mask=None):
    if action_mask is None and observation is None:  # this is mainly for testing purposes
        observation = env.observation()
        action_mask = env.action_mask()
    action = [0] * len(action_mask)
    action_index = min([i for i in range(len(action_mask)) if action_mask[i]])
    action[action_index] = 1
    if action_index == 1:  # if we are assigning r2, but there are few cases waiting, it may be better to postpone
        if observation[4] < 5 and action_mask[2]:  # but note that this is only allowed if we can postpone
            action = [0, 0, 1, 0]
    return action


if __name__ == '__main__':
    nr_replications = 100
    avg_cycle_times = []
    total_rewards = []
    for _ in range(nr_replications):
        #reporter = EventLogReporter("mdp_log_0.5.txt")
        reporter = ProcessReporter()
        tau = 0.25
        env = MDP(3000, tau=tau, reporter=reporter, config_type='parallel')

        done = False
        steps = 0
        total_reward = 0
        while not done:
            action = greedy_policy(env)            
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            time = env.total_time

            # print(action, state, reward, time)
            steps += 1
        # print('nr_steps:', steps)
        # print('reward:', total_reward)
        reporter.close()
        # reporter.print_result()
        avg_cycle_times.append(reporter.total_cycle_time / reporter.nr_completed)
        total_rewards.append(total_reward)
        #print(total_reward, reporter.total_cycle_time, reporter.nr_completed, reporter.total_cycle_time / reporter.nr_completed)
    # print mean and 95% confidence interval of the average cycle time
    avg_cycle_times = np.array(avg_cycle_times)
    total_rewards = np.array(total_rewards)
    print('tau:', env.tau)
    print('mean CT:', np.mean(avg_cycle_times))
    print('95% CI CT:', np.percentile(avg_cycle_times, [2.5, 97.5]))
    print('mean reward:', np.mean(total_rewards))
    print('95% CI reward:', np.percentile(total_rewards, [2.5, 97.5]))
    reporter.print_result()
    print('actions taken:', env.actions_taken)
    print('actual actions taken:', env.actual_actions_taken)
    print('partially completed cases:', len(env.partially_completed_cases))
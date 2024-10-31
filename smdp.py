from reporters import EventLogReporter, ProcessReporter
import sys, os, json
import numpy as np
import random

class SMDP:

    def __init__(self, nr_arrivals, config_type='single_activity', reporter=None):
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

        self.state_space = [f"{resource}_{status}" for status in ['available', 'assigned'] for resource in self.resources] +\
                           [f"{task}_queue" for task in self.task_types]
        self.action_space = [f"{resource}{task}" for resource in self.resources for task in self.task_types if resource in self.resource_pools[task] and task != 'Start'] + ['postpone', 'do_nothing']

        self.waiting_cases = {task: [] for task in self.task_types}
        #self.completed_cases = []
        #self.cases = {}
        self.processing_r1 = []
        self.processing_r2 = []
        self.total_time = 0
        self.total_arrivals = 0
        self.nr_arrivals = nr_arrivals
        self.original_nr_arrivals = nr_arrivals

        self.actions_taken = {}

        self.reporter = reporter

    def observation(self):
        # TODO change the state based on the number of resoruces and activites.
        # TODO self.processing_r1 and r2 and be more than 1 if we have multiple activities
        # TODO normalization of the observation
        is_processing_r1 = len(self.processing_r1) > 0 # True if there is a case being processed by r1
        is_processing_r2 = len(self.processing_r2) > 0
        assigned_r1 = 1 if is_processing_r1 and self.processing_r1[-1][1] == 'a' else 2 if is_processing_r1 and self.processing_r1[-1][1] == 'b' else 0
        assigned_r2 = 1 if is_processing_r2 and self.processing_r2[-1][1] == 'a' else 2 if is_processing_r2 and self.processing_r2[-1][1] == 'b' else 0
        waiting_cases = [len(self.waiting_cases.get(task, [])) for task in self.task_types if task != "Start"]
        return [1-is_processing_r1, 1-is_processing_r2, assigned_r1, assigned_r2] + waiting_cases

    def reset(self):
        self.waiting_cases = {task: [] for task in self.task_types}
        #self.cases = {}
        #self.completed_cases = []
        self.processing_r1 = []
        self.processing_r2 = []
        self.total_time = 0
        self.total_arrivals = 0
        self.nr_arrivals = self.original_nr_arrivals
    
    def get_state(self, rollout_length=None):
        nr_arrivals = self.nr_arrivals if rollout_length is None else self.nr_arrivals + rollout_length
        return ({task: cases.copy() for task, cases in self.waiting_cases.items()}, self.processing_r1.copy(), self.processing_r2.copy(), self.total_time, self.total_arrivals, nr_arrivals)
    
    def set_state(self, state):
        self.waiting_cases = {task: cases.copy() for task, cases in state[0].items()}
        self.processing_r1 = state[1].copy()
        self.processing_r2 = state[2].copy()
        self.total_time = state[3]
        self.total_arrivals = state[4]
        self.nr_arrivals = state[5]

        # TODO: add check for eligibility of the state

    def from_state(state):
        smdp = SMDP(0)
        smdp.set_state(state)
        return smdp

    def sample_next_task(self, current_task):
        # Calculate the sum of the values
        p_transitions = self.transitions[current_task]
        total_sum = sum(p_transitions)
        
        # Check if the sum is approximately 1 or less, considering rounding errors
        if np.isclose(total_sum, 1, atol=1e-9) or total_sum < 1:
            # Draw a random sample using the crn class
            return np.random.choice(self.task_types_all, p=p_transitions)
        else:
            # Check if all nonzero values are equal to 1
            nonzero_indices = [index for index, value in enumerate(p_transitions) if value != 0]
            if any(p_transitions[index] != 1 for index in nonzero_indices):
                raise ValueError("All nonzero values must be equal to 1 when using parallism.")
            # Return the indices of the list that contain a nonzero value
            return [self.task_types_all[index] for index in nonzero_indices]

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

    def step(self, action):
        action_index = action.index(1)
        action = self.action_space[action_index]
        if action not in self.actions_taken:
            self.actions_taken[action] = 0
        self.actions_taken[action] += 1
        # process the action, 'postpone' and 'do nothing', do nothing to the state.
        if action in ['r1a', 'r2a', 'r1b', 'r2b']:
            resource, task = action[0:2], action[2]
            if len(self.waiting_cases[task]) > 0:
                getattr(self, f'processing_{resource}').append((self.waiting_cases[task].pop(0), task))
                if self.reporter:
                    self.reporter.callback(getattr(self, f'processing_{resource}')[-1][0], task, '<task:start>', self.total_time, resource)

        # now calculate the next state and how long it takes to reach that state
        # the time is to the next state is exponentially distributed with rate
        # min(lambda, mu_(r1, a) * active r1, mu_(r2, a) * active r2) = 
        # exponential(0.5 + 1/1.8 * len(self.nr_processing_r1) + 1/10 * len(self.nr_processing_r2))
        # this is the probability of the next state being the consequence of:
        # an arrival, r1 processing the task or r2 processing the task
        # the probability of one of these evolutions happening is proportional to the rate of that evolution.
        evolutions, evolution_rates = self.get_evolutions(self.processing_r1, self.processing_r2, self.arrivals_coming())

        nr_active_cases = len(self.processing_r1) + len(self.processing_r2) + sum(len(v) for v in self.waiting_cases.values())
        sum_of_rates = sum(evolution_rates.values())
        expected_event_time = 1 / sum_of_rates
        expected_reward = -nr_active_cases * expected_event_time

        if sum_of_rates == 0:
            return self.observation(), 0, self.is_done(), False, None
        else:
            time = random.expovariate(sum_of_rates)
            self.total_time += time
            events, probs = zip(*list(evolutions.items()))
            evolution = np.random.choice(events, p=probs)

            if evolution == 'arrival':
                task = self.sample_next_task('Start')
                self.waiting_cases[task].append(self.total_arrivals)
                #self.cases[self.total_arrivals] = [("Start", self.total_time)]
                if self.reporter:
                    self.reporter.callback(self.total_arrivals, 'start', '<start_event>', self.total_time)
                self.total_arrivals += 1
            else:
                resource, task = evolution[0:2], evolution[2]
                case_id = getattr(self, f'processing_{resource}').pop(0)[0]
                #self.cases[case_id].append((task, self.total_time))
                next_task = self.sample_next_task(task)
                if self.reporter:
                    self.reporter.callback(case_id, task, '<task:complete>', self.total_time, resource)
                if next_task and next_task != 'Complete':
                    self.waiting_cases[next_task].append(case_id)
                elif next_task == 'Complete':
                    if self.reporter:
                        self.reporter.callback(case_id, 'complete', '<end_event>', self.total_time)
            reward = expected_reward
            return self.observation(), reward, self.is_done(), False, None
        
    def arrivals_coming(self):
        return 1 if self.total_arrivals < self.nr_arrivals else 0

    def action_mask(self):
        # (r1, a) is only possible if there is a task waiting and r1 is available
        # (r2, a) is only possible if there is a task waiting and r2 is available
        # postpone is only possible if there is something to postpone, i.e. there is a task waiting and a resource is available
        # do nothing is only possible if nothing can be done, i.e. there is no task waiting or no resource is available
        if len(self.task_types) == 1:
            r1_available = len(self.processing_r1) == 0
            r2_available = len(self.processing_r2) == 0
            a_waiting = len(self.waiting_cases['a']) > 0
            a1_possible = a_waiting and len(self.processing_r1) == 0 and 'r1' in self.resource_pools['a']
            a2_possible = a_waiting and len(self.processing_r2) == 0 and 'r2' in self.resource_pools['a']
            postpone_possible = self.arrivals_coming() > 0 or (not r1_available and not r2_available)
            do_nothing_possible = not (a1_possible or a2_possible or postpone_possible)
            return [a1_possible, a2_possible, postpone_possible, do_nothing_possible]
        else:
            r1_available = len(self.processing_r1) == 0
            r2_available = len(self.processing_r2) == 0
            a_waiting = len(self.waiting_cases['a']) > 0
            b_waiting = len(self.waiting_cases['b']) > 0

            r1a_possible = a_waiting and r1_available and 'r1' in self.resource_pools['a']
            r1b_possible = b_waiting and r1_available and 'r1' in self.resource_pools['b']
            r2a_possible = a_waiting and r2_available and 'r2' in self.resource_pools['a']
            r2b_possible = b_waiting and r2_available and 'r2' in self.resource_pools['b']

            # postpone is allowed if:
            # - there are arrivals and one resource is occupied
            # - there are arrivals and there is a queue at only one of the tasks
            # postpone is not allowed if:
            # - no assignments are possible
            # - there are no arrivals and both resources are available
            # - all assignments are possible
            if self.arrivals_coming():
                # (one resource, one task), (two resources, one task), (one resource, two tasks), not (two resources, two tasks)
                postpone_possible = 1 <= sum([r1a_possible, r1b_possible, r2a_possible, r2b_possible]) <= 4
            else:
                # (one resource, one task), not (two resources, one task), (one resource, two tasks), not (two resources, two tasks)
                postpone_possible = (1 <= sum([r1a_possible, r1b_possible, r2a_possible, r2b_possible]) <= 4 and
                                    not (r1_available and r2_available)) # can't postpone if both resources are available, but no arrivals 

            do_nothing_possible = sum([r1a_possible, r1b_possible, r2a_possible, r2b_possible, postpone_possible]) == 0

            if self.config_type != 'n_system':
                return [r1a_possible, r1b_possible, r2a_possible, r2b_possible, postpone_possible, do_nothing_possible]
            else:
                return [r1a_possible, r1b_possible, r2b_possible, postpone_possible, do_nothing_possible]
    
    def is_done(self):
        """
        The simulation is done if we have reached the maximum number of arrivals and there are no more tasks to process.
        """
        return not self.arrivals_coming() and sum(len(v) for v in self.waiting_cases.values()) == 0 and len(self.processing_r1) == 0 and len(self.processing_r2) == 0
        

def random_policy(env):
    action_mask = env.action_mask()
    action = [0] * len(action_mask)
    action_index = np.random.choice([i for i in range(len(action_mask)) if action_mask[i]])
    action[action_index] = 1
    #print('policy:', action_mask, action_index, action)
    return action

def greedy_policy(env):
    action_mask = env.action_mask()
    action = [0] * len(action_mask)
    
    min_processing_time = float('inf')
    best_action_index = None
    
    for i, possible in enumerate(action_mask):
        if possible:
            action_str = env.action_space[i]
            if action_str in ['postpone', 'do_nothing']:
                continue
            resource, task = action_str[0:2], action_str[2]
            processing_time = env.resource_pools[task][resource][0]
            if processing_time < min_processing_time:
                min_processing_time = processing_time
                best_action_index = i
    
    if best_action_index is not None:
        action[best_action_index] = 1
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
    longest_waiting_case_b = min(env.waiting_cases['b']) if len(env.waiting_cases['b']) > 0 else None
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
        return random_policy(env)
    else:
        return greedy_policy(env)

def threshold_policy(env, observation=None, action_mask=None, threshold=5):
    if action_mask is None and observation is None:  # this is mainly for testing purposes
        observation = env.observation()
        action_mask = env.action_mask()
    action = [0] * len(action_mask)
    action_index = min([i for i in range(len(action_mask)) if action_mask[i]])
    action[action_index] = 1
    if action_index == 1:  # if we are assigning r2, but there are few cases waiting, it may be better to postpone
        if observation[4] < threshold and action_mask[2]:  # but note that this is only allowed if we can postpone
            action = [0, 0, 1, 0]
    return action


if __name__ == '__main__':
    nr_replications = 1
    avg_cycle_times = []
    total_rewards = []
    for _ in range(nr_replications):
        reporter = EventLogReporter("smdp_log.txt")
        #reporter = ProcessReporter()
        env = SMDP(100000, 'slow_server', reporter)

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
    # print mean and 95% confidence interval of the average cycle time and rewards
    avg_cycle_times = np.array(avg_cycle_times)
    total_rewards = np.array(total_rewards)
    print('mean CT:', np.mean(avg_cycle_times))
    print('95% CI CT:', np.percentile(avg_cycle_times, [2.5, 97.5]))
    print('mean reward:', np.mean(total_rewards))
    print('95% CI reward:', np.percentile(total_rewards, [2.5, 97.5]))
    reporter.print_result()
    print('Actions taken:', env.actions_taken)

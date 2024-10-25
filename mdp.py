from reporters import EventLogReporter, ProcessReporter
from crn import CRN
import sys, os, json
import numpy as np

PRINT_TRAJECTORY = False

class MDP:
    def __init__(self, nr_arrivals, config_type='single_activity', tau=0.5, reporter=None, crn=None):
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
        self.completed_cases = []

        self.processing_r1 = []
        self.processing_r2 = []
        self.total_time = 0
        self.total_arrivals = 0
        self.nr_arrivals = nr_arrivals
        self.original_nr_arrivals = nr_arrivals
        self.tau = tau

        if crn is None:
            crn = CRN()
        self.crn = crn
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
        self.completed_cases = []
        self.processing_r1 = []
        self.processing_r2 = []
        self.total_time = 0
        self.total_arrivals = 0 
        self.nr_arrivals = self.original_nr_arrivals
        # Add crn.reset()
    
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

    def from_state(state):
        smdp = MDP(0)
        smdp.set_state(state)
        return smdp

    def sample_next_task(self, current_task):
        # Calculate the sum of the values
        p_transitions = self.transitions[current_task]
        total_sum = sum(p_transitions)
        
        # Check if the sum is approximately 1 or less, considering rounding errors
        if np.isclose(total_sum, 1, atol=1e-9) or total_sum < 1:
            # Draw a random sample using the crn class
            return self.crn.choice(self.task_types_all, weights=p_transitions)
        else:
            # Check if all nonzero values are equal to 1
            nonzero_indices = [index for index, value in enumerate(p_transitions) if value != 0]
            if any(p_transitions[index] != 1 for index in nonzero_indices):
                raise ValueError("All nonzero values must be equal to 1 when using parallism.")
            # Return the indices of the list that contain a nonzero value
            return [self.task_types_all[index] for index in nonzero_indices]

    def step(self, action):
        action_index = action.index(1)
        action = self.action_space[action_index]

        if PRINT_TRAJECTORY: print("Obseration:", self.observation(), 'Action:', action, self.total_time, self.action_mask())
        # create an intermediate state to calculate the expected reward and the next state
        # we use this state to calculate if the state actually transitions to the next state
        # or if the state remains the same. In latter, no resources are added/removed.
        
        # Initialize evolutions dictionary
        r1_processing = len(self.processing_r1) > 0
        r2_processing = len(self.processing_r2) > 0
        nr_active_cases = r1_processing + r2_processing + sum(len(v) for v in self.waiting_cases.values())

        # Calculate the possible evolutions and the rate at which they happen
        evolutions = {key: value for key, value in {
            'arrival': self.arrival_rate if self.arrivals_coming() else 0,
            'r1a': 1/self.resource_pools['a']['r1'][0] if r1_processing and self.processing_r1[0][1] == 'a' else 0,
            'r2a': 1/self.resource_pools['a']['r2'][0] if r2_processing and self.processing_r2[0][1] == 'a' else 0,
            'r1b': 1/self.resource_pools['b']['r1'][0] if r1_processing and self.processing_r1[0][1] == 'b' else 0,
            'r2b': 1/self.resource_pools['b']['r2'][0] if r2_processing and self.processing_r2[0][1] == 'b' else 0
        }.items() if value > 0}

        # add possible evolution based on action 
        if action == 'r1a': # (r1, a)
            evolutions['r1a'] = 1/self.resource_pools['a']['r1'][0]
        elif action == 'r2a': # (r2, a)
            evolutions['r2a'] = 1/self.resource_pools['a']['r2'][0]
        elif action == 'r1b': # (r1, b)
            evolutions['r1b'] = 1/self.resource_pools['b']['r1'][0]
        elif action == 'r2b': # (r2, b)
            evolutions['r2b'] = 1/self.resource_pools['b']['r2'][0]

        sum_of_rates = sum(evolutions.values())
        if sum_of_rates == 0:
            return self.observation(), 0, self.is_done(), False, None

        expected_event_time = 1 / sum_of_rates
        expected_reward = -expected_event_time * nr_active_cases
        reward_rate = expected_reward / expected_event_time
        
        transformed_evolutions = {}
        for evolution, rate in evolutions.items():
            transformed_evolutions[evolution] = rate * self.tau / expected_event_time
        transformed_evolutions['return_to_state'] = 1 - self.tau / expected_event_time
        reward = reward_rate * self.tau

        events, probs = zip(*list(transformed_evolutions.items()))
        evolution = self.crn.choice(events, weights=probs)
        if evolution != 'return_to_state':
            next_task = None
            # process the action, 'postpone' and 'do nothing', do nothing to the state.
            if action in ['r1a', 'r2a', 'r1b', 'r2b']:
                resource, task = action[0:2], action[2]
                if len(self.waiting_cases[task]) > 0:
                    getattr(self, f'processing_{resource}').append((self.waiting_cases[task].pop(0), task))
                    if self.reporter:
                        self.reporter.callback(getattr(self, f'processing_{resource}')[-1][0], task, '<task:start>', self.total_time, resource)
                else:                    
                    print("Invalid action")
                    print('Observation:', self.observation())
                    print('Action:', action)
                    print('Action mask:', self.action_mask())
                    print('nr_arrivals:', self.nr_arrivals, 'total_arrivals:', self.total_arrivals)
            # now calculate the next state and how long it takes to reach that state
            # the time is to the next state is exponentially distributed with rate 
            # min(lambda, mu_(r1, a) * active r1, mu_(r2, a) * active r2) = 
            # exponential(0.5 + 1/1.8 * len(self.nr_processing_r1) + 1/10 * len(self.nr_processing_r2))
            # this is the probability of the next state being the consequence of:
            # an arrival, r1 processing the task or r2 processing the task
            # the probability of one of these evolutions happening is proportional to the rate of that evolution.
            if evolution == 'arrival':
                # sample the first task from the transition matrix
                self.waiting_cases[self.sample_next_task('Start')].append(self.total_arrivals)
                if self.reporter is not None:
                    self.reporter.callback(self.total_arrivals, 'start', '<start_event>', self.total_time)
                self.total_arrivals += 1
            else:
                resource, task = evolution[0:2], evolution[2]
                case_id = getattr(self, f'processing_{resource}').pop(0)[0]
                next_task = self.sample_next_task(task)
                if self.reporter:
                    self.reporter.callback(case_id, task, '<task:complete>', self.total_time, resource)
                if next_task and next_task != 'Complete':
                    self.waiting_cases[next_task].append(case_id)
                elif next_task == 'Complete':
                    if self.reporter:
                        self.reporter.callback(case_id, 'complete', '<end_event>', self.total_time)
                    self.completed_cases.append(case_id)
            self.total_time += self.tau
            return self.observation(), reward, self.is_done(), False, None
        else:
            self.total_time += self.tau
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
                postpone_possible = 1 < sum([r1a_possible, r1b_possible, r2a_possible, r2b_possible]) <= 4
            else:
                # (one resource, one task), not (two resources, one task), (one resource, two tasks), not (two resources, two tasks)
                postpone_possible = (1 < sum([r1a_possible, r1b_possible, r2a_possible, r2b_possible]) <= 4 and
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
    action_index = env.crn.choice([i for i in range(len(action_mask)) if action_mask[i]])
    action[action_index] = 1
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
        # Check if 'postpone' is feasible
        if 'postpone' in env.action_space:
            postpone_index = env.action_space.index('postpone')
            if action_mask[postpone_index]:
                action[postpone_index] = 1
                return action
        
        # Check if 'do_nothing' is feasible
        if 'do_nothing' in env.action_space:
            do_nothing_index = env.action_space.index('do_nothing')
            if action_mask[do_nothing_index]:
                action[do_nothing_index] = 1
    
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
        index = env.action_space.index(env.crn.choice(feasible_actions))
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
            index = env.action_space.index(env.crn.choice(feasible_actions))
            action[index] = 1
            return action
    return action

def epsilon_greedy_policy(env):
    if env.crn.generate_uniform() < 0.1:
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
    reporter = ProcessReporter()
    tau = 0.5
    env = MDP(50, tau=tau, reporter=reporter, config_type='slow_server')

    done = False
    steps = 0
    total_reward = 0
    max_steps = 100000
    while steps < max_steps and not done:        
        action = greedy_policy(env)
        
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        time = env.total_time

        # print(action, state, reward, time)

        steps += 1
    print('nr_steps:', steps)
    print('reward:', total_reward)
    reporter.close()
    reporter.print_result()
from reporters import EventLogReporter
from crn import CRN
import sys, os, json
import numpy as np

class SMDP:

    def __init__(self, nr_arrivals, config_type='single_activity', reporter=None, crn=None):
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

        self.state_space = [resource + '_available' for resource in self.resources] +\
                           [resource + '_assigned' for resource in self.resources] +\
                           [task + '_queue' for task in self.task_types]
        self.action_space = [resource+task for resource in self.resources for task in self.task_types if resource in self.resource_pools[task] if task != 'Start'] + ['postpone'] + ['do_nothing']
        self.waiting_cases = {task: [] for task in self.task_types}
        self.completed_cases = []

        self.processing_r1 = []
        self.processing_r2 = []
        self.total_time = 0
        self.total_arrivals = 0
        self.nr_arrivals = nr_arrivals
        self.original_nr_arrivals = nr_arrivals

        if crn is None:
            crn = CRN()
        self.crn = crn
        self.reporter = reporter

    def observation(self):
        # TODO change the state based on the number of resoruces and activites.
        # TODO self.processing_r1 and r2 and be more than 1 if we have multiple activities
        # TODO normalization of the observation
        is_processing_r1 = 1 if len(self.processing_r1) > 0 else 0
        is_processing_r2 = 1 if len(self.processing_r2) > 0 else 0
        if is_processing_r1:
            assigned_r1 = 1 if self.processing_r1[-1][1] == 'a' else 2 if self.processing_r1[-1][1] == 'b' else 0
        else:
            assigned_r1 = 0
        if is_processing_r2:
            assigned_r2 = 1 if self.processing_r2[-1][1] == 'a' else 2 if self.processing_r2[-1][1] == 'b' else 0
        else:
            assigned_r2 = 0
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
    
    def get_state(self, rollout=False):
        nr_arrivals = self.nr_arrivals if not rollout else self.nr_arrivals + self.total_arrivals
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
        #print('step:  ', self.action_mask(), self.action_space[action.index(1)], action)
        action = self.action_space[action_index]
        mask = self.action_mask()
        if not mask[action_index] == 0:
            # process the action, 'postpone' and 'do nothing', do nothing to the state.
            if action == 'r1a':  # (r1, a)
                self.processing_r1.append((self.waiting_cases['a'].pop(0), 'a'))
                if self.reporter is not None:
                    self.reporter.callback(self.processing_r1[-1], 'a', '<task:start>', self.total_time)
            elif action == 'r2a': # (r2, a)
                self.processing_r2.append((self.waiting_cases['a'].pop(0), 'a'))
                if self.reporter is not None:
                    self.reporter.callback(self.processing_r2[-1], 'a', '<task:start>', self.total_time)
            elif action == 'r1b':  # (r1, b)
                self.processing_r1.append((self.waiting_cases['b'].pop(0), 'b'))
                if self.reporter is not None:
                    self.reporter.callback(self.processing_r1[-1], 'b', '<task:start>', self.total_time)
            elif action == 'r2b': # (r2, b)
                self.processing_r2.append((self.waiting_cases['b'].pop(0), 'b'))
                if self.reporter is not None:
                    self.reporter.callback(self.processing_r2[-1], 'b', '<task:start>', self.total_time)
        else:
            print("Invalid action")
            print('Observation:', self.observation())
            print('Action:', action)
            print('Action mask:', mask)
            print('nr_arrivals:', self.nr_arrivals, 'total_arrivals:', self.total_arrivals)
        # now calculate the next state and how long it takes to reach that state
        # the time is to the next state is exponentially distributed with rate
        # min(lambda, mu_(r1, a) * active r1, mu_(r2, a) * active r2) = 
        # exponential(0.5 + 1/1.8 * len(self.nr_processing_r1) + 1/10 * len(self.nr_processing_r2))
        # this is the probability of the next state being the consequence of:
        # an arrival, r1 processing the task or r2 processing the task
        # the probability of one of these evolutions happening is proportional to the rate of that evolution.
        nr_active_cases = len(self.processing_r1) + len(self.processing_r2) + sum(len(v) for v in self.waiting_cases.values())

        # Calculate the possible evolutions and the rate at which they happen
        evolutions = {}
        if self.arrivals_coming():
            evolutions['arrival'] = self.arrival_rate
        if len(self.processing_r1) > 0: # if processing, adds the evaluation of the completion of the task (e.g. r1a)
            processing_task = self.processing_r1[-1][1]
            if 'r1' in self.resource_pools[processing_task].keys():
                evolutions['r1'+processing_task] = 1/self.resource_pools[processing_task]['r1'][0]
        if len(self.processing_r2) > 0: # if processing, adds the evaluation of the completion of the task (e.g. r2a)
            processing_task = self.processing_r2[-1][1]
            if 'r2' in self.resource_pools[processing_task].keys():
                evolutions['r2'+processing_task] = 1/self.resource_pools[processing_task]['r2'][0]

        sum_of_rates = sum(evolutions.values())
        if sum_of_rates == 0:
            return self.observation(), 0, self.is_done(), False, None
        else:
            next_task = None
            # now we sample the next event time which depends on the sum of rates
            time = self.crn.generate_exponential(sum_of_rates)
            self.total_time += time
            choices = list(evolutions.keys())
            probabilities = [evolutions[choice]/sum_of_rates for choice in choices]
            evolution = self.crn.choice(choices, weights=probabilities)
            if evolution == 'arrival':
                # sample the first task from the transition matrix
                self.waiting_cases[self.sample_next_task('Start')].append(self.total_arrivals)
                if self.reporter is not None:
                    self.reporter.callback(self.total_arrivals, 'start', '<event:arrival>', self.total_time)
                self.total_arrivals += 1
            elif evolution == 'r1a':
                case_id = self.processing_r1.pop(0)[0]
                next_task = self.sample_next_task('a')
                if self.reporter is not None:
                    self.reporter.callback(case_id, 'a', '<task:complete>', self.total_time, 'r1')
            elif evolution == 'r2a':
                case_id = self.processing_r2.pop(0)[0]
                next_task = self.sample_next_task('a')
                if self.reporter is not None:
                    self.reporter.callback(case_id, 'a', '<task:complete>', self.total_time, 'r2')
            elif evolution == 'r1b':
                case_id = self.processing_r1.pop(0)[0]
                next_task = self.sample_next_task('b')
                if self.reporter is not None:
                    self.reporter.callback(case_id, 'b', '<task:complete>', self.total_time, 'r1')
            elif evolution == 'r2b':
                case_id = self.processing_r2.pop(0)[0]
                next_task = self.sample_next_task('b')
                if self.reporter is not None:
                    self.reporter.callback(case_id, 'b', '<task:complete>', self.total_time, 'r2')

            if next_task is not None and next_task != 'Complete':
                self.waiting_cases[next_task].append(case_id)
                if self.reporter is not None:
                    self.reporter.callback(case_id, next_task, '<task:start>', self.total_time)
            elif next_task == 'Complete':
                self.completed_cases.append(case_id)
            reward = -time*nr_active_cases
            return self.observation(), reward, self.is_done(), False, None
        
    def arrivals_coming(self):
        return 1 if self.total_arrivals < self.nr_arrivals else 0

    def action_mask(self):
        # (r1, a) is only possible if there is a task waiting and r1 is available
        # (r2, a) is only possible if there is a task waiting and r2 is available
        # postpone is only possible if there is something to postpone, i.e. there is a task waiting and a resource is available
        # do nothing is only possible if nothing can be done, i.e. there is no task waiting or no resource is available
        if len(self.task_types) == 1:
            a1_possible = len(self.waiting_cases['a']) > 0 and len(self.processing_r1) == 0 and 'r1' in self.resource_pools['a'].keys()
            a2_possible = len(self.waiting_cases['a']) > 0 and len(self.processing_r2) == 0 and 'r2' in self.resource_pools['a'].keys()
            postpone_possible = (self.arrivals_coming() > 0) and (a1_possible or a2_possible)
            do_nothing_possible = not (a1_possible or a2_possible or postpone_possible)
            return [a1_possible, a2_possible, postpone_possible, do_nothing_possible]
        else:
            r1a_possible = len(self.waiting_cases['a']) > 0 and len(self.processing_r1) == 0 and 'r1' in self.resource_pools['a'].keys()
            r1b_possible = len(self.waiting_cases['b']) > 0 and len(self.processing_r1) == 0 and 'r1' in self.resource_pools['b'].keys()
            r2a_possible = len(self.waiting_cases['a']) > 0 and len(self.processing_r2) == 0 and 'r2' in self.resource_pools['a'].keys()
            r2b_possible = len(self.waiting_cases['b']) > 0 and len(self.processing_r2) == 0 and 'r2' in self.resource_pools['b'].keys()
            postpone_possible = (self.arrivals_coming() > 0) and (r1a_possible or r1b_possible or r2a_possible or r2b_possible)
            do_nothing_possible = not (r1a_possible or r1b_possible or r2a_possible or r2b_possible or postpone_possible)
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
    #print('policy:', action_mask, action_index, action)
    return action

def greedy_policy(env):
    action_mask = env.action_mask()
    action = [0] * len(action_mask)
    action_index = min([i for i in range(len(action_mask)) if action_mask[i]])
    action[action_index] = 1
    return action

def epsilon_greedy_policy(env):
    if env.crn.generate_uniform() < 0.1:
        return env.random_policy()
    else:
        return env.greedy_policy()

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
    reporter = EventLogReporter("test.csv")

    env = SMDP(50, 'slow_server', reporter)

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
    print('nr_steps:', steps)
    print('reward:', total_reward)
    reporter.close()
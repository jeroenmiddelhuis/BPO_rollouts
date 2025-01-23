from reporters import EventLogReporter, ProcessReporter
import os, json
import numpy as np
import random

class SMDP_composite:
    def __init__(self, nr_arrivals, config_type='single_activity', reporter=None, reward_function='AUC'):
        # Read the config file and set the process parameters
        self.config_type = config_type
        self.reward_function = reward_function
        self.env_type = 'smdp'

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.txt"), "r") as f:
            data = f.read()
        
        config = json.loads(data)
        config = config[config_type]
        
        self.task_types = [task for task in list(config['task_types']) if task != 'Start']
        self.task_types_all = [task for task in list(config['task_types'])] + ['Complete']
        self.resources = list(config['resources'])
        self.resource_pools = config['resource_pools']
        self.arrival_rate = 1/config['mean_interarrival_time']
        self.transitions = config['transitions']

        self.evolution_rates = {f'r{i}{task}': 1/self.resource_pools[task][f'r{i}'][0] 
                     for i in range(1, 13) 
                     for task in self.task_types 
                     if f'r{i}' in self.resource_pools[task]}
        self.evolution_rates['arrival'] = self.arrival_rate

        self.state_space = ([f'is_available_{r}' for r in self.resources] + 
                            [f'assigned_{resource}{task}' for resource in self.resources 
                                            for task in self.task_types                                             
                                            if resource in self.resource_pools[task] and resource != 'r9'] + # r9 can only be assigned to one task
                            [f'waiting_{task}' for task in self.task_types])
        
        self.action_space = [f"{resource}{task}" for resource in self.resources for task in self.task_types if resource in self.resource_pools[task] and task != 'Start'] + ['postpone', 'do_nothing']

        self.waiting_cases = {task: [] for task in self.task_types}
        self.partially_completed_cases = []

        self.processing_r1 = []
        self.processing_r2 = []
        self.processing_r3 = []
        self.processing_r4 = []
        self.processing_r5 = []
        self.processing_r6 = []
        self.processing_r7 = []
        self.processing_r8 = []
        self.processing_r9 = []
        self.processing_r10 = []
        self.processing_r11 = []
        self.processing_r12 = []

        self.total_time = 0
        self.total_arrivals = 0
        self.nr_arrivals = nr_arrivals
        self.original_nr_arrivals = nr_arrivals
        self.episodic_reward = 0
        self.arrival_times = {}
        self.cycle_times = {}

        self.processing_starts = {}
        self.processing_times = {}
        self.waiting_starts = {}
        self.waiting_times = {}
        self.observations = {}

        self.actions_taken = {}
        self.reporter = reporter

    def observation(self):
        is_available_r1, is_available_r2, is_available_r3, is_available_r4, \
        is_available_r5, is_available_r6, is_available_r7, is_available_r8, \
        is_available_r9, is_available_r10, is_available_r11, is_available_r12 = \
            [1 if len(getattr(self, f'processing_r{i}')) == 0 else 0 for i in range(1, 13)]
        assigned_r1a = 1 if not is_available_r1 and self.processing_r1[-1][1] == 'a' else 0
        assigned_r1b = 1 if not is_available_r1 and self.processing_r1[-1][1] == 'b' else 0
        assigned_r2a = 1 if not is_available_r2 and self.processing_r2[-1][1] == 'a' else 0
        assigned_r2b = 1 if not is_available_r2 and self.processing_r2[-1][1] == 'b' else 0
        assigned_r3c = 1 if not is_available_r3 and self.processing_r3[-1][1] == 'c' else 0
        assigned_r3d = 1 if not is_available_r3 and self.processing_r3[-1][1] == 'd' else 0
        assigned_r4c = 1 if not is_available_r4 and self.processing_r4[-1][1] == 'c' else 0
        assigned_r4d = 1 if not is_available_r4 and self.processing_r4[-1][1] == 'd' else 0
        assigned_r5e = 1 if not is_available_r5 and self.processing_r5[-1][1] == 'e' else 0
        assigned_r5f = 1 if not is_available_r5 and self.processing_r5[-1][1] == 'f' else 0
        assigned_r6e = 1 if not is_available_r6 and self.processing_r6[-1][1] == 'e' else 0
        assigned_r6f = 1 if not is_available_r6 and self.processing_r6[-1][1] == 'f' else 0
        assigned_r7g = 1 if not is_available_r7 and self.processing_r7[-1][1] == 'g' else 0
        assigned_r7h = 1 if not is_available_r7 and self.processing_r7[-1][1] == 'h' else 0
        assigned_r8g = 1 if not is_available_r8 and self.processing_r8[-1][1] == 'g' else 0
        assigned_r8h = 1 if not is_available_r8 and self.processing_r8[-1][1] == 'h' else 0
        assigned_r9j = 1 if not is_available_r9 and self.processing_r9[-1][1] == 'j' else 0
        assigned_r10i = 1 if not is_available_r10 and self.processing_r10[-1][1] == 'i' else 0
        assigned_r10j = 1 if not is_available_r10 and self.processing_r10[-1][1] == 'j' else 0
        assigned_r11k = 1 if not is_available_r11 and self.processing_r11[-1][1] == 'k' else 0
        assigned_r11l = 1 if not is_available_r11 and self.processing_r11[-1][1] == 'l' else 0
        assigned_r12k = 1 if not is_available_r12 and self.processing_r12[-1][1] == 'k' else 0
        assigned_r12l = 1 if not is_available_r12 and self.processing_r12[-1][1] == 'l' else 0

        waiting_cases = [len(self.waiting_cases.get(task, [])) for task in self.task_types if task != "Start"]

        return [is_available_r1, is_available_r2, is_available_r3, is_available_r4,
                is_available_r5, is_available_r6, is_available_r7, is_available_r8, 
                is_available_r9, is_available_r10, is_available_r11, is_available_r12,
                assigned_r1a, assigned_r1b, assigned_r2a, assigned_r2b, assigned_r3c, assigned_r3d,
                assigned_r4c, assigned_r4d, assigned_r5e, assigned_r5f, assigned_r6e, assigned_r6f,
                assigned_r7g, assigned_r7h, assigned_r8g, assigned_r8h, assigned_r9j, assigned_r10i,
                assigned_r10j, assigned_r11k, assigned_r11l, assigned_r12k, assigned_r12l] + waiting_cases

    def action_mask(self):
        processing_resources = self.get_processing_resources()
        availability = [True if len(processing_resources[i]) == 0 else False for i in range(len(processing_resources))]

        is_waiting_tasks = {task: len(self.waiting_cases[task]) > 0 for task in self.task_types if task != 'Start'}
        possible_actions = [availability[i] and is_waiting_tasks[task] 
                                for i in range(len(processing_resources)) 
                                for task in self.task_types 
                                if f'r{i + 1}{task}' in self.action_space]
        
        r1a_possible, r1b_possible, r2a_possible, r2b_possible, r3c_possible, r3d_possible, \
        r4c_possible, r4d_possible, r5e_possible, r5f_possible, r6e_possible, r6f_possible, \
        r7g_possible, r7h_possible, r8g_possible, r8h_possible, r9j_possible, r10i_possible, \
        r10j_possible, r11k_possible, r11l_possible, r12k_possible, r12l_possible = possible_actions

        assignments_possible = [r1a_possible, r1b_possible, r2a_possible, r2b_possible, r3c_possible, r3d_possible,
                                r4c_possible, r4d_possible, r5e_possible, r5f_possible, r6e_possible, r6f_possible,
                                r7g_possible, r7h_possible, r8g_possible, r8h_possible, r9j_possible, r10i_possible,
                                r10j_possible, r11k_possible, r11l_possible, r12k_possible, r12l_possible]
        if self.arrivals_coming():            
            postpone_possible = any(possible_actions) and not (r1a_possible and r2a_possible and sum(assignments_possible) == 2)
        else:
            postpone_possible = any(possible_actions)
        do_nothing_possible = not postpone_possible

        return assignments_possible + [postpone_possible, do_nothing_possible]

    def reset(self):
        self.waiting_cases = {task: [] for task in self.task_types}
        self.partially_completed_cases = []

        self.processing_r1 = []
        self.processing_r2 = []
        self.processing_r3 = []
        self.processing_r4 = []
        self.processing_r5 = []
        self.processing_r6 = []
        self.processing_r7 = []
        self.processing_r8 = []
        self.processing_r9 = []
        self.processing_r10 = []
        self.processing_r11 = []
        self.processing_r12 = []

        self.total_time = 0
        self.total_arrivals = 0
        self.nr_arrivals = self.original_nr_arrivals
        self.arrival_times = {}
        self.cycle_times = {}
        self.processing_starts = {}
        self.processing_times = {}
        self.waiting_starts = {}
        self.waiting_times = {}
        self.episodic_reward = 0

    def get_state(self, rollout_length=None):
        # cases, list of processing, total time, total arrivals, nr arrivals
        nr_arrivals = self.nr_arrivals if rollout_length is None else self.nr_arrivals + rollout_length
        return ({task: cases.copy() for task, cases in self.waiting_cases.items()},
            [self.processing_r1.copy(),
            self.processing_r2.copy(),
            self.processing_r3.copy(),
            self.processing_r4.copy(),
            self.processing_r5.copy(),
            self.processing_r6.copy(),
            self.processing_r7.copy(),
            self.processing_r8.copy(),
            self.processing_r9.copy(),
            self.processing_r10.copy(),
            self.processing_r11.copy(),
            self.processing_r12.copy()],
            self.total_time,
            self.total_arrivals,
            nr_arrivals,
            self.partially_completed_cases.copy())

    def set_state(self, state):
        # cases, list of processing, total time, total arrivals, nr arrivals
        self.waiting_cases = {task: cases.copy() for task, cases in state[0].items()}
        for i in range(1, 13):
            setattr(self, f'processing_r{i}', state[1][i-1].copy())
        self.total_time = state[2]
        self.total_arrivals = state[3]
        self.nr_arrivals = state[4]
        self.partially_completed_cases = state[5].copy()
        self.locked_action = None

    def sample_next_task(self, current_task, case_id=None):
        # Calculate the sum of the values
        p_transitions = self.transitions[current_task]
        total_sum = sum(p_transitions)

        if current_task == 'Complete':
            raise ValueError('The current task cannot be Complete.')
        elif current_task == 'i' or current_task == 'j': # If the parallel task is i or j, return k and l
            return ['k', 'l']
        elif current_task == 'k' or current_task == 'l': # If the parallel task is k or l, return Complete
            if case_id in self.partially_completed_cases:
                return ['Complete']
            else:
                return []
        else: # If the current task is not a parallel or XOR task
            # Check if the sum is approximately 1 or less, considering rounding errors
            if np.isclose(total_sum, 1, atol=1e-9) or total_sum < 1:
                # Draw a random sample using the crn class
                return [np.random.choice(self.task_types_all, p=p_transitions)]
            else:
                raise ValueError("The sum of the transition probabilities must be 1 or less.")

    def get_evolution_rates(self, processing_resources, arrivals_coming, action=None):
        """
        Returns a dictionary with the possible evolutions and their rates. \n
        processing_resources should be a list,
        arrivals_coming should be a boolean,
        action should be a (feasible) action from the action space. \n
        For the MDP, the selected action is used to determine the possible evolutions.
        """
        processing_resources = self.get_processing_resources()
        is_available_resources = [len(processing_resources[i]) == 0 for i in range(len(processing_resources))]
        # Create evolution dictionary depending on arrivals_coming
        evolution_rates = {'arrival': self.evolution_rates['arrival'] if arrivals_coming else 0}

        for i in range(1, 13):
            for task in self.task_types:
                key = f'r{i}{task}'
                if not is_available_resources[i-1] and processing_resources[i-1][0][1] == task:
                    evolution_rates[key] = self.evolution_rates[key]

        # evolution_rates = {key: value for key, value in evolution_rates.items() if value > 0}

        # add possible evolution based on action 
        if action and action not in ['postpone', 'do_nothing']:
            evolution_rates[action] = self.evolution_rates[action]
        return evolution_rates

    def get_evolutions(self, processing_resources, arrivals_coming, action=None):
        evolution_rates = self.get_evolution_rates(processing_resources, arrivals_coming, action)
        sum_of_rates = sum(evolution_rates.values())
        evolutions = {}
        for evolution, rate in evolution_rates.items():
            evolutions[evolution] = rate/sum_of_rates
        return evolutions, evolution_rates

    def arrivals_coming(self):
        return 1 if self.total_arrivals < self.nr_arrivals else 0
   
    def is_done(self):
        return self.total_time > 100000
        """
        The simulation is done if we have reached the maximum number of arrivals and there are no more tasks to process.
        """
        cases_processing = sum(len(is_processing) for is_processing in self.get_processing_resources())
        waiting_cases = sum(len(v) for v in self.waiting_cases.values())
        return not self.arrivals_coming() and cases_processing == 0 and waiting_cases == 0
    
    def get_processing_resources(self):
        return [getattr(self, f'processing_r{i}') for i in range(1, 13)]

    def step(self, action):
        reward = 0
        action_index = action.index(1)
        action = self.action_space[action_index]
        if action not in self.actions_taken:
            self.actions_taken[action] = 0
        self.actions_taken[action] += 1
        # process the action, 'postpone' and 'do nothing', do nothing to the state.
        if action.startswith('r'): # only assignments start with r (i.e., r1a)
            resource, task = action[0:-1], action[-1]
            if len(self.waiting_cases[task]) > 0:
                case_id = self.waiting_cases[task].pop(0)
                self.processing_starts[case_id][(task, resource)] = self.total_time
                self.waiting_times[case_id][task] = self.total_time - self.waiting_starts[case_id][task]
                getattr(self, f'processing_{resource}').append((case_id, task))
                if self.reporter:
                    self.reporter.callback(getattr(self, f'processing_{resource}')[-1][0], task, '<task:start>', self.total_time, resource)

        # now calculate the next state and how long it takes to reach that state
        # the time is to the next state is exponentially distributed with rate
        # min(lambda, mu_(r1, a) * active r1, mu_(r2, a) * active r2) = 
        # exponential(0.5 + 1/1.8 * len(self.nr_processing_r1) + 1/10 * len(self.nr_processing_r2))
        # this is the probability of the next state being the consequence of:
        # an arrival, r1 processing the task or r2 processing the task
        # the probability of one of these evolutions happening is proportional to the rate of that evolution.
        evolutions, evolution_rates = self.get_evolutions(self.get_processing_resources(), self.arrivals_coming())
        if self.reward_function == 'AUC':
            unique_active_cases = set()
            for i in range(1, 13):
                processing_case = [getattr(self, f'processing_r{i}')[0][0]] if len(getattr(self, f'processing_r{i}')) > 0 else []
                unique_active_cases.update(processing_case)
            
            for task in self.task_types:
                unique_active_cases.update(self.waiting_cases[task])

        sum_of_rates = sum(evolution_rates.values())

        if sum_of_rates == 0:
            return self.observation(), 0, self.is_done(), False, None
        else:
            time = random.expovariate(sum_of_rates) # sample time from T~exp(sum_of_rates)
            self.total_time += time # time of next event
            events, probs = zip(*list(evolutions.items()))
            evolution = np.random.choice(events, p=probs)

            if evolution == 'arrival': # arrival event
                self.arrival_times[self.total_arrivals] = self.total_time

                self.waiting_starts[self.total_arrivals] = {}
                self.processing_starts[self.total_arrivals] = {}
                self.waiting_times[self.total_arrivals] = {}
                self.processing_times[self.total_arrivals] = {}

                # sample the first task from the transition matrix
                next_tasks = self.sample_next_task('Start')
                for task in next_tasks:
                    self.waiting_starts[self.total_arrivals][task] = self.total_time
                    self.waiting_cases[task].append(self.total_arrivals)
                if self.reporter:
                    self.reporter.callback(self.total_arrivals, 'start', '<start_event>', self.total_time)
                self.total_arrivals += 1
            else: # task completion event
                resource, task = evolution[0:-1], evolution[-1]
                case_id = getattr(self, f'processing_{resource}').pop(0)[0]
                self.processing_times[case_id][(task, resource)] = self.total_time - self.processing_starts[case_id][(task, resource)]
                next_tasks = self.sample_next_task(task, case_id)
                # completion the parallel task adds to list of partially completed cases
                # if task i or j has been completed previously, the case may now continue
                if task == 'i' or task == 'j': 
                    self.partially_completed_cases.append(case_id)

                if self.reporter:
                    self.reporter.callback(case_id, task, '<task:complete>', self.total_time, resource)
                for next_task in next_tasks:                    
                    if next_task and next_task != 'Complete':
                        self.waiting_starts[case_id][next_task] = self.total_time
                        self.waiting_cases[next_task].append(case_id)
                    elif next_task == 'Complete':
                        self.cycle_times[case_id] = self.total_time - self.arrival_times[case_id]
                        if self.reward_function == 'case_cycle_time':                            
                            reward += -self.cycle_times[case_id]
                        elif self.reward_function == 'inverse_case_cycle_time':
                            reward += 1/(1 + self.cycle_times[case_id])
                        if self.reporter:
                            self.reporter.callback(case_id, 'complete', '<end_event>', self.total_time)
            if self.reward_function == 'AUC':
                reward += time * -len(unique_active_cases)
            self.episodic_reward += reward
            observation = self.observation()
            if tuple(observation) not in self.observations:
                self.observations[tuple(observation)] = 0
            self.observations[tuple(observation)] += 1
            return observation, reward, self.is_done(), False, None

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
ACTION_R10I = 0
ACTION_R10J = 0
ACTION_R9J = 0
def greedy_policy(env):
    action_mask = env.action_mask()
    action = [0] * len(action_mask)
    # if action_mask[env.action_space.index('r10i')]:
    #     action[env.action_space.index('r10i')] = 1
    #     return action
    min_processing_time = float('inf')
    best_action_index = []
    for i, possible in enumerate(action_mask):
        if possible:
            action_str = env.action_space[i]
            if action_str in ['postpone', 'do_nothing']:
                continue # skips this iteration
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
    action_index = action.index(1)
    action_str = env.action_space[action_index]
    if action_str == 'r10i':
        global ACTION_R10I
        ACTION_R10I += 1
    elif action_str == 'r10j':
        global ACTION_R10J
        ACTION_R10J += 1
    elif action_str == 'r9j':
        global ACTION_R9J
        ACTION_R9J += 1
    return action


if __name__ == '__main__':
    # smdp = SMDP_composite(2500, 'complete')
    # print(smdp.sample_next_task('k'))
    # print(smdp.state_space)
    # print(smdp.action_space)

    # # Sample state space where only action r1a and r2b are available
    # smdp.waiting_cases = {task: [] for task in smdp.task_types}
    # smdp.waiting_cases['a'] = [1]  # Task 'a' is waiting
    # smdp.waiting_cases['b'] = [2]  # Task 'b' is waiting
    # smdp.waiting_cases['l'] = [13]  # Task 'c' is waiting
    # smdp.waiting_cases['j'] = [14]  # Task 'd' is waiting

    # smdp.processing_r1 = [(1, 'a')]
    # smdp.processing_r2 = []
    # smdp.processing_r3 = []
    # smdp.processing_r4 = []
    # smdp.processing_r5 = [(3, 'e')]
    # smdp.processing_r6 = [(4, 'f')]
    # smdp.processing_r7 = []
    # smdp.processing_r8 = []
    # smdp.processing_r9 = []
    # smdp.processing_r10 = []
    # smdp.processing_r11 = [(12, 'k')]
    # smdp.processing_r12 = [(13, 'l')]

    # evolutions, evolution_rates = smdp.get_evolutions(smdp.get_processing_resources(), smdp.arrivals_coming())
    # print(evolutions)
    # print(evolution_rates)
    # print(sum(evolution_rates.values()))
    # print(np.mean([random.expovariate(sum(evolution_rates.values())) for _ in range(1000)]))

    # observation = smdp.observation()
    # action_mask = smdp.action_mask()

    # for i in range(len(smdp.state_space)):
    #     if i < len(smdp.action_space):
    #         print(smdp.state_space[i],  '\t' if 'available' in smdp.state_space[i] else '\t\t', observation[i], '\t\t', smdp.action_space[i], '\t\t', action_mask[i])
    #     else:
    #         print(smdp.state_space[i], '\t' if 'available' in smdp.state_space[i] else '\t\t', observation[i])



    # print("State Space:", smdp.observation())
    # print("Action Mask:", smdp.action_mask())

    # print('Evolutions:', smdp.get_evolutions(smdp.get_processing_resources(), smdp.arrivals_coming(), 'r1a'))








    nr_replications = 10
    avg_cycle_times = []
    total_rewards = []
    for _ in range(nr_replications):
        time_iter = 1
        #reporter = EventLogReporter("smdp_log.txt")
        reporter = ProcessReporter()
        env = SMDP_composite(1000000, 'complete')

        done = False
        steps = 0
        total_reward = 0
        while not done:        
            action = greedy_policy(env)            
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            time = env.total_time
            steps += 1
            if time > time_iter * 2500:
                time_iter += 1
                print('action r10j:', ACTION_R10J, 'action r10i:', ACTION_R10I, 'action r9j:', ACTION_R9J)
                ACTION_R10I = 0
                ACTION_R10J = 0 
                ACTION_R9J = 0                
        print(env.total_arrivals, env.total_time)
        avg_cycle_times.append(np.mean(list(env.cycle_times.values())))
        total_rewards.append(total_reward)
        print(np.mean(list(env.cycle_times.values())), total_reward)
        for task_type in env.task_types:
            print(f'Mean waiting time for task {task_type}: {np.mean([env.waiting_times[case_id][task_type] for case_id in env.waiting_times if task_type in env.waiting_times[case_id]])} {len([env.waiting_times[case_id][task_type] for case_id in env.waiting_times if task_type in env.waiting_times[case_id]])}')
            for resource in env.resource_pools[task_type]:
                print(f'Mean processing time for task {task_type} with resource {resource}: {np.mean([env.processing_times[case_id][(task_type, resource)] for case_id in env.processing_times if (task_type, resource) in env.processing_times[case_id]])}')
        for case in env.cycle_times:
            processing_waiting_sum = sum([env.processing_times[case][(task, resource)] for (task, resource) in env.processing_times[case] if task not in ['k', 'l']]) + \
                                     sum([env.waiting_times[case][task] for task in env.waiting_times[case] if task not in ['k', 'l']])
            keys_with_k = [key for key in env.processing_times[case] if 'k' in key]
            keys_with_l = [key for key in env.processing_times[case] if 'l' in key]
            processing_waiting_k = 0
            for key in keys_with_k:
                processing_waiting_k += env.processing_times[case][key] + env.waiting_times[case][key[0]]
            processing_waiting_l = 0
            for key in keys_with_l:
                processing_waiting_l += env.processing_times[case][key] + env.waiting_times[case][key[0]]

            processing_waiting_sum += max(processing_waiting_k, processing_waiting_l)
            #print(f'Case {case} cycle time: {env.cycle_times[case]} Sum of processing times and waiting times: {processing_waiting_sum}')


        #print('Top observations:', sorted(env.observations.items(), key=lambda x: x[1], reverse=True)[:50])

    avg_cycle_times = np.array(avg_cycle_times)
    total_rewards = np.array(total_rewards)
    print('mean CT:', np.mean(avg_cycle_times))
    print('95% CI CT:', np.percentile(avg_cycle_times, [2.5, 97.5]))
    print('mean reward:', np.mean(total_rewards))
    print('95% CI reward:', np.percentile(total_rewards, [2.5, 97.5]))
    reporter.print_result()
    print('Actions taken:', env.actions_taken)
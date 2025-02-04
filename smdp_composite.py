from reporters import EventLogReporter, ProcessReporter
import os, json
import numpy as np
import random

class SMDP_composite:
    def __init__(self, nr_arrivals, config_type='composite', reporter=None, reward_function='AUC'):
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
                     for i in range(1, len(self.resources)+1) 
                     for task in self.task_types 
                     if f'r{i}' in self.resource_pools[task]}
        self.evolution_rates['arrival'] = self.arrival_rate

        self.state_space = ([f'is_available_{r}' for r in self.resources] + 
                            [f'assigned_{resource}{task}' for resource in self.resources 
                                            for task in self.task_types                                             
                                            if resource in self.resource_pools[task] and resource != 'r9'] + # r9 can only be assigned to one task
                            [f'waiting_{task}' for task in self.task_types])
        
        self.action_space = [f"{resource}{task}" for resource in self.resources 
                             for task in self.task_types 
                             if resource in self.resource_pools[task] and task != 'Start'] 
        
        self.assignments = self.action_space.copy()
        self.assignment_indices = {assignment: idx for idx, assignment in enumerate(self.assignments)}
        # double actions
        # high utilization
        self.double_assignments =  [(ass1, ass2) 
                                    for ass1 in ['r1a', 'r2a', 'r1b', 'r2b'] 
                                    for ass2 in ['r3c', 'r4c']]
        # slow server
        self.double_assignments += [(ass1, ass2) 
                                    for ass1 in ['r3c', 'r4c', 'r3d', 'r4d'] 
                                    for ass2 in ['r5e', 'r6e']]
        # downstream
        self.double_assignments += [(ass1, ass2) 
                                    for ass1 in ['r5e', 'r6e', 'r5f', 'r6f'] 
                                    for ass2 in ['r7g', 'r8g']]
        # n_system
        self.double_assignments += [(ass1, ass2) 
                                    for ass1 in ['r7g', 'r8g', 'r7h', 'r8h'] 
                                    for ass2 in ['r9j', 'r10i', 'r10j']]
        # parallel
        self.double_assignments += [(ass1, ass2) 
                                    for ass1 in ['r9j', 'r10i', 'r10j'] 
                                    for ass2 in ['r11k', 'r12k', 'r11l', 'r12l']]
        self.double_assignments += [('r11k', 'r12l'), ('r11l', 'r12k')]

        self.action_space += self.double_assignments
        self.action_space += ['postpone', 'do_nothing']

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

        self.actions_taken = {}
        self.reporter = reporter
        self.test = 0

    def observation(self):
        processing_resources = self.get_processing_resources()
        # Check if the resources are available (resource 1 has index 0)
        is_available_resources = [1 if len(processing_resources[i]) == 0 else 0 for i in range(len(processing_resources))]
        # Check which assignments are assigned
        assigned_assignments = [0 if not is_available_resources[int(assignment[1:-1])-1] 
                                and getattr(self, f'processing_r{int(assignment[1:-1])}')[-1][1] == assignment[-1] 
                                else 1
                                for assignment in self.assignments]
        # Check the number of waiting cases at each activity
        waiting_cases = [len(self.waiting_cases.get(task, [])) for task in self.task_types if task != "Start"]

        return is_available_resources + assigned_assignments + waiting_cases

    def action_mask(self):
        processing_resources = self.get_processing_resources()
        # Check if the resources are available (resource 1 has index 0)
        is_available_resources = [True if len(processing_resources[i]) == 0 else False for i in range(len(processing_resources))]
        # Check if there are tasks waiting for the resources
        is_waiting_tasks = {task: len(self.waiting_cases[task]) > 0 for task in self.task_types if task != 'Start'}
        # Check which assignments are possible
        assignments_possible = [is_available_resources[int(assignment[1:-1])-1] 
                                and is_waiting_tasks[assignment[-1]] 
                                for assignment in self.assignments]
        # Check which double assignments are possible based on the single assignments
        double_assignments_possible = [assignments_possible[self.assignment_indices[assignment[0]]]
                                        and assignments_possible[self.assignment_indices[assignment[1]]] 
                                        for assignment in self.double_assignments]

        if self.arrivals_coming():
            # When arrivals are coming, postponing is possible when there are tasks waiting for the resources
            # However, if there are only tasks at a, and resource 1 and 2 are not assigned, we don't allow postponing
            postpone_possible = (any(assignments_possible) 
                                 and not (assignments_possible[self.assignment_indices['r1a']] 
                                          and assignments_possible[self.assignment_indices['r2a']] 
                                          and sum(assignments_possible) == 2))
        else:
            postpone_possible = any(assignments_possible)
        do_nothing_possible = not postpone_possible
        
        return assignments_possible + double_assignments_possible + [postpone_possible, do_nothing_possible]

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
            [processing_resouce.copy() for processing_resouce in self.get_processing_resources()],
            self.total_time,
            self.total_arrivals,
            nr_arrivals,
            self.partially_completed_cases.copy())

    def set_state(self, state):
        self.reset()
        # cases, list of processing, total time, total arrivals, nr arrivals
        self.waiting_cases = {task: cases.copy() for task, cases in state[0].items()}
        for i in range(1, len(self.resources)+1):
            setattr(self, f'processing_r{i}', state[1][i-1].copy())
        self.total_time = state[2]
        self.total_arrivals = state[3]
        self.nr_arrivals = state[4]
        self.partially_completed_cases = state[5].copy()

    def sample_next_task(self, current_task, case_id=None):
        # Calculate the sum of the values
        p_transitions = self.transitions[current_task]
        total_sum = sum(p_transitions)
        if current_task == 'Complete':
            raise ValueError('The current task cannot be "Complete".')
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
        
        # Create evolution dictionary depending on arrivals_coming
        evolution_rates = {
            assignment:self.evolution_rates[assignment] 
            for assignment in self.assignments 
            if len(processing_resources[int(assignment[1:-1])-1]) > 0 # The resource is assigned
            and processing_resources[int(assignment[1:-1])-1][-1][1] == assignment[-1] # Task is assigned to resource
            }
        if arrivals_coming:
            evolution_rates['arrival'] = self.evolution_rates['arrival']
        
        # add possible evolution based on action 
        if action is not None and action not in ['postpone', 'do_nothing']:
            if isinstance(action, tuple):
                for act in action:
                    evolution_rates[act] = 1/self.resource_pools[act[-1]][act[:-1]][0]
            else:
                evolution_rates[action] = 1/self.resource_pools[action[-1]][action[:-1]][0]
        return evolution_rates

    def get_evolutions(self, processing_resources, arrivals_coming, action=None):
        evolution_rates = self.get_evolution_rates(processing_resources, arrivals_coming, action)
        sum_of_rates = sum(evolution_rates.values())
        evolutions = {evolution: rate/sum_of_rates for evolution, rate in evolution_rates.items()}
        return evolutions, evolution_rates

    def arrivals_coming(self):
        return 1 if self.total_arrivals < self.nr_arrivals else 0
   
    def is_done(self):
        """
        The simulation is done if we have reached the maximum number of arrivals and there are no more tasks to process.
        """
        return (not self.arrivals_coming() 
                and sum(len(v) for v in self.waiting_cases.values()) == 0
                and sum(len(is_processing) for is_processing in self.get_processing_resources()) == 0)
    
    def get_processing_resources(self):
        return [getattr(self, f'processing_r{i}') for i in range(1, len(self.resources)+1)]

    def step(self, action):
        reward = 0
        # The action is passed as a binary list (neural network). We need to convert it to the action name
        # The heuristics return the action name directly or a tuple of two actions
        if isinstance(action, list):
            action_index = action.index(1)
            action = self.action_space[action_index]
        # process the action, 'postpone' and 'do nothing', do nothing to the state.
        if action not in ['postpone', 'do_nothing']:       
            if isinstance(action, str):
                actions = [action]
            else:
                actions = action
            for act in actions:
                resource, task = act[0:-1], act[-1]
                if self.waiting_cases[task]:
                    case_id = self.waiting_cases[task].pop(0)
                    self.processing_starts[case_id][(task, resource)] = self.total_time
                    self.waiting_times[case_id][task] = self.total_time - self.waiting_starts[case_id][task]
                    getattr(self, f'processing_{resource}').append((case_id, task))
                    if self.reporter:
                        self.reporter.callback(case_id, task, '<task:start>', self.total_time, resource)
                else:
                    raise Exception(f'No cases waiting for task {task} to be processed by resource {resource}')
        
        # now calculate the next state and how long it takes to reach that state
        # the time is to the next state is exponentially distributed with rate
        # min(lambda, mu_(r1, a) * active r1, mu_(r2, a) * active r2) = 
        # exponential(0.5 + 1/1.8 * len(self.nr_processing_r1) + 1/10 * len(self.nr_processing_r2))
        # this is the probability of the next state being the consequence of:
        # an arrival, r1 processing the task or r2 processing the task
        # the probability of one of these evolutions happening is proportional to the rate of that evolution.
        evolutions, evolution_rates = self.get_evolutions(self.get_processing_resources(), self.arrivals_coming())

        if self.reward_function == 'AUC':
            # Create set of unique active cases
            unique_active_cases = {getattr(self, f'processing_r{i}')[0][0] 
                                   for i in range(1, len(self.resources)+1) 
                                   if len(getattr(self, f'processing_r{i}')) > 0}            
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
            return self.observation(), reward, self.is_done(), False, None

def greedy_policy(env):
    action_mask = env.action_mask()  
    if sum(action_mask) == 1: # only do nothing possible
        #print('Locked action:', env.action_space[action_mask.index(1)])
        return env.action_space[action_mask.index(1)]

    possible_actions = [env.action_space[i] for i, action in enumerate(action_mask[:-2]) if action]
    assignments = [assignment for assignment in possible_actions if isinstance(assignment, str)]
    double_assignments = [assignment for assignment in possible_actions if isinstance(assignment, tuple)]

    min_processing_time = float('inf')
    lowest_processing_times = []

    for assignment in assignments:
        resource, task = assignment[0:-1], assignment[-1]
        processing_time = env.resource_pools[task][resource][0]
        if processing_time < min_processing_time:
            min_processing_time = processing_time
            lowest_processing_times = [assignment]
        elif processing_time == min_processing_time:
            lowest_processing_times.append(assignment)

    assignment = random.choice(lowest_processing_times)
    possible_double_assignments = [double_assignment for double_assignment in double_assignments if assignment in double_assignment]

    if len(possible_double_assignments) > 0:
        min_processing_time = float('inf')
        lowest_processing_times = []

        for double_assignment in possible_double_assignments:
            if double_assignment[0] == assignment:
                other_action = double_assignment[1]
            else:
                other_action = double_assignment[0]
            resource, task = other_action[0:-1], other_action[-1]
            processing_time = env.resource_pools[task][resource][0]
            if processing_time < min_processing_time:
                min_processing_time = processing_time
                lowest_processing_times = [double_assignment]
            elif processing_time == min_processing_time:
                lowest_processing_times.append(double_assignment)
    
    if len(lowest_processing_times) > 0:
        return random.choice(lowest_processing_times)
    else:
        return assignment

def fifo_policy(env):
    action_mask = env.action_mask()
    if sum(action_mask) == 1: # only do nothing possible
        return 'do_nothing'
    possible_actions = [env.action_space[i] for i, action in enumerate(action_mask[:-2]) if action] # Excluding postpone, do_nothing
    possible_tasks = [action[-1] for action in possible_actions if isinstance(action, str)]

    # Identify the case that has been in the system the longest
    all_waiting_cases = sorted([(case_id, task) for task in env.waiting_cases for case_id in env.waiting_cases[task] if task in possible_tasks], key=lambda x: x[0])
    # print('Waiting cases:', all_waiting_cases)
    # print('Possible actions:', possible_actions)

    i = 0
    while i < len(all_waiting_cases):
        longest_waiting_case_id = all_waiting_cases[i][0] # if len(all_waiting_cases) > 0 else None
        longest_case_tasks = [(case_id, task) for case_id, task in all_waiting_cases if case_id == longest_waiting_case_id]
        
        # Identify if the longest waiting case can be processed        
        longest_waiting_case_tasks = [task for case_id, task in longest_case_tasks]
        if len(longest_waiting_case_tasks) == 0:
            i += 1
            continue
        random.shuffle(longest_waiting_case_tasks) # Randomly assign a task of the longest waiting case
        selected_task = longest_waiting_case_tasks[0]

        # Get an assignment with the selected task (of the longest waiting case)
        possible_assignments = [action for action in possible_actions if action[-1] == selected_task]
        if len(possible_assignments) > 0:
            assignment = random.choice([action for action in possible_actions if action[-1] == selected_task])
        else:
            i += 1
            continue
        #print('Selected assignment:', assignment)
        # Create list of possible double assignments
        possible_double_assignments = [double_assignment for double_assignment in possible_actions if isinstance(double_assignment, tuple) and assignment in double_assignment]

        #print(possible_double_assignments)
        if len(possible_double_assignments) > 0:
            checked_case_ids = []
            # Check for each case if the selected task is in the double assignment
            for (case_id, task) in all_waiting_cases:
                #print('checking case:', case_id)
                if case_id not in checked_case_ids:
                    checked_case_ids.append(case_id)
                else:
                    continue
                # Create list of tasks of the (second) longest waiting case
                tasks_of_case_id = [task2 for case_id2, task2 in all_waiting_cases if case_id2 == case_id]
                #print('Tasks of case:', case_id, tasks_of_case_id)
                possible_double_assignments_case = []
                for task2 in tasks_of_case_id:
                    if task2 != selected_task:
                        test = [double_assignment for double_assignment in possible_double_assignments if task2 in double_assignment[0] or task2 in double_assignment[1]]
                        #print('Additional double assignmetns:', test)
                        possible_double_assignments_case += test#[double_assignment for double_assignment in possible_double_assignments if (task2 in action[0] or task2 in action[1]) and task2 != selected_task]
                if len(possible_double_assignments_case) > 0:
                    assignment = random.choice(possible_double_assignments_case)
                    #print('returned assignment', assignment, '\n')
                    return tuple(assignment)
        else:
            #print('returned assignment', assignment, '\n')
            return assignment

def random_policy(env):
    action_mask = env.action_mask()  
    if sum(action_mask) == 1: # only do nothing possible
        return env.action_space[action_mask.index(1)]
    possible_actions = [env.action_space[i] for i, action in enumerate(action_mask[:-2]) if action]
    assignments = [assignment for assignment in possible_actions if isinstance(assignment, str)]
    double_assignments = [assignment for assignment in possible_actions if isinstance(assignment, tuple)]

    assignment = random.choice(assignments)
    possible_double_asignments = [double_assignment for double_assignment in double_assignments if assignment in double_assignment]

    if len(possible_double_asignments) > 0:
        return tuple(random.choice(possible_double_asignments))
    else:
        return assignment

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

    nr_replications = 100
    avg_cycle_times = []
    total_rewards = []
    for _ in range(nr_replications):
        time_iter = 1
        #reporter = EventLogReporter("smdp_log.txt")
        reporter = ProcessReporter()
        env = SMDP_composite(2500, 'complete')

        done = False
        steps = 0
        total_reward = 0
        while not done:        
            action = fifo_policy(env)            
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            time = env.total_time
            steps += 1

              
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
    print('95% CI CT:', [np.mean(avg_cycle_times)-1.96*np.std(avg_cycle_times)/np.sqrt(len(avg_cycle_times)), np.mean(avg_cycle_times)+1.96*np.std(avg_cycle_times)/np.sqrt(len(avg_cycle_times))])
    print('mean reward:', np.mean(total_rewards))
    print('95% CI reward:', [np.mean(total_rewards)-1.96*np.std(total_rewards)/np.sqrt(len(total_rewards)), np.mean(total_rewards)+1.96*np.std(total_rewards)/np.sqrt(len(total_rewards))])
    print('\n')
    #reporter.print_result()

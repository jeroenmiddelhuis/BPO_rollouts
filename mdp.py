from reporters import EventLogReporter, ProcessReporter
import sys, os, json
import numpy as np
import random
from crn import CRN

class MDP:
    def __init__(self, nr_arrivals, config_type='single_activity', 
                 tau=0.5, reporter=None, crn=None, reward_function='AUC', 
                 track_cycle_times=True, is_stopping_criteria_time=False):
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
        self.crn = crn
        self.track_cycle_times = track_cycle_times
        self.is_stopping_criteria_time = is_stopping_criteria_time
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
        self.evolution_rates = {f'r{i}{task}': 1/self.resource_pools[task][f'r{i}'][0] 
                     for i in range(1, len(self.resources)+1) 
                     for task in self.task_types 
                     if f'r{i}' in self.resource_pools[task]}
        self.evolution_rates['arrival'] = self.arrival_rate


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
        self.action_space = [f"{resource}{task}" for resource in self.resources for task in self.task_types if resource in self.resource_pools[task] and task != 'Start'] 
        self.assignments = self.action_space.copy()
        self.assignment_indices = {assignment: idx for idx, assignment in enumerate(self.assignments)}
        
        if self.config_type == 'parallel':
            self.action_space += [('r1a', 'r2b'), ('r1b', 'r2a')]

        self.action_space += ['postpone', 'do_nothing']

        self.waiting_cases = {task: [] for task in self.task_types}
        self.partially_completed_cases = []

        self.processing_r1 = []
        self.processing_r2 = []

        self.total_time = 0
        self.total_arrivals = 0
        if self.is_stopping_criteria_time:
            nr_arrivals = nr_arrivals * 2 # To ensure that the simulation runs for a longer time
        self.nr_arrivals = nr_arrivals
        self.original_nr_arrivals = nr_arrivals
        self.tau = tau
        self.locked_action = None
        self.episodic_reward = 0
        if self.track_cycle_times:
            self.arrival_times = {}
            self.cycle_times = {}

        self.reporter = reporter

    def observation(self):
        is_available_r1 = 1 if len(self.processing_r1) == 0 else 0 # True if there is a case being processed by r1
        is_available_r2 = 1 if len(self.processing_r2) == 0 else 0
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
        # The action is locked, so return the mask with only the locked action available
        if self.locked_action is not None:
            if isinstance(self.locked_action, list):
                return self.locked_action
            action = [0] * len(self.action_space)
            action[self.action_space.index(self.locked_action)] = 1
            return action
        
        # single activity
        if len(self.task_types) == 1: 
            r1_available = len(self.processing_r1) == 0
            r2_available = len(self.processing_r2) == 0
            a_waiting = len(self.waiting_cases['a']) > 0
            r1a_possible = a_waiting and r1_available
            r2a_possible = a_waiting and r2_available
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

            if self.config_type == 'n_system':
                # If all actions are possible, postpone is not allowed
                if self.arrivals_coming():
                    postpone_possible = 1 <= sum([r1b_possible, r2a_possible, r2b_possible]) < 3 
                else:
                    # If both resources are busy, postpone is not allowed (no assignment possible)
                    # If both resources are available, postpone is not allowed, since no new cases will arrive
                    # Only when one of the resources is busy, postpone is allowed to wait for its completion
                    postpone_possible = r1_available != r2_available
                do_nothing_possible = sum([r1b_possible, r2a_possible, r2b_possible, postpone_possible]) == 0
                return [r1b_possible, r2a_possible, r2b_possible, 
                        postpone_possible, do_nothing_possible]
            
            elif self.config_type == 'parallel':
                r1a_r2b_possible = r1a_possible and r2b_possible
                r2a_r1b_possible = r2a_possible and r1b_possible
                if self.arrivals_coming():
                    postpone_possible = 1 <= sum([r1a_possible, r1b_possible, r2a_possible, r2b_possible, r1a_r2b_possible, r2a_r1b_possible]) < 6
                else:
                    postpone_possible = r1_available != r2_available
                do_nothing_possible = sum([r1a_possible, r1b_possible, r2a_possible, r2b_possible, r1a_r2b_possible, r2a_r1b_possible, postpone_possible]) == 0
                return [r1a_possible, r1b_possible, r2a_possible, r2b_possible, 
                        r1a_r2b_possible, r2a_r1b_possible, 
                        postpone_possible, do_nothing_possible]
            
            else: # All other scenarios
                # If there are cases waiting at activity A and both resources are avaible, postpone is not allowed
                if self.arrivals_coming():
                    postpone_possible = (not (r1a_possible and r2a_possible) and # If there are no cases at B, postpone is not allowed
                                        1 <= sum([r1a_possible, r1b_possible, r2a_possible, r2b_possible]) < 4)
                else:
                    postpone_possible = r1_available != r2_available
                do_nothing_possible = sum([r1a_possible, r1b_possible, r2a_possible, r2b_possible, postpone_possible]) == 0
                return [r1a_possible, r1b_possible, r2a_possible, r2b_possible, 
                        postpone_possible, do_nothing_possible]

    def reset(self):
        if self.crn:
            self.crn.reset()
        self.waiting_cases = {task: [] for task in self.task_types}
        self.partially_completed_cases = []
        self.processing_r1 = []
        self.processing_r2 = []
        self.total_time = 0
        self.total_arrivals = 0
        self.nr_arrivals = self.original_nr_arrivals
        self.locked_action = None
        if self.track_cycle_times:
            self.arrival_times = {}
            self.cycle_times = {}

        self.episodic_reward = 0
    
    def get_state(self, rollout_length=None):
        nr_arrivals = self.nr_arrivals if rollout_length is None else self.nr_arrivals + rollout_length
        return ({task: cases.copy() for task, cases in self.waiting_cases.items()},
                [self.processing_r1.copy(), self.processing_r2.copy()],
                self.total_time,
                self.total_arrivals,
                nr_arrivals,
                self.partially_completed_cases.copy())
    
    def set_state(self, state):
        self.reset()    
        self.waiting_cases = {task: cases.copy() for task, cases in state[0].items()}
        self.processing_r1 = state[1][0].copy()
        self.processing_r2 = state[1][1].copy()
        self.total_time = state[2]
        self.total_arrivals = state[3]
        self.nr_arrivals = state[4]
        # For parallel systems, we need to keep track of the partially completed cases
        if state[5] is not None:
            self.partially_completed_cases = state[5].copy()

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
                if self.crn:
                    return [self.crn.choice(self.task_types_all, weights=p_transitions)]
                else:
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

    def get_transformed_evolutions(self, processing_resources, arrivals_coming, action=None):
        evolutions, evolution_rates = self.get_evolutions(processing_resources, arrivals_coming, action)
        sum_of_rates = sum(evolution_rates.values())
        #if sum_of_rates == 0:
        expected_next_event_time = 1 / sum_of_rates
        transformed_evolutions = {evolution: self.tau / expected_next_event_time * p 
                                  for evolution, p in evolutions.items()}
        transformed_evolutions['return_to_state'] = 1 - self.tau / expected_next_event_time
        return transformed_evolutions, evolution_rates

    def arrivals_coming(self):
        return 1 if self.total_arrivals < self.nr_arrivals else 0
    
    def is_done(self):
        """
        The simulation is done if we have reached the maximum number of arrivals and there are no more tasks to process.
        """
        if self.is_stopping_criteria_time:
            return self.total_time > 5000
        else:
            return (not self.arrivals_coming() 
                    and sum(len(v) for v in self.waiting_cases.values()) == 0 
                    and len(self.processing_r1) == 0 and len(self.processing_r2) == 0)

    def get_processing_resources(self):
        return [getattr(self, f'processing_r{i}') for i in range(1, len(self.resources)+1)]

    def step(self, action):
        # We save the action so we can lock it if we return to the same state
        # The locking is done in the action mask function
        original_action = action
        reward = 0
        # The action is passed as a binary list (neural network). We need to convert it to the action name
        # The heuristics return the action name directly or a tuple of two actions
        if isinstance(action, list):
            action_index = action.index(1)
            action = self.action_space[action_index]
        # Determine the evolution based on the current state and action
        transformed_evolutions, _ = self.get_transformed_evolutions(self.get_processing_resources(), self.arrivals_coming(), action)
        
        if self.reward_function == 'AUC':
            # Create set of unique active cases
            unique_active_cases = {getattr(self, f'processing_r{i}')[0][0] 
                                   for i in range(1, len(self.resources)+1) 
                                   if len(getattr(self, f'processing_r{i}')) > 0}            
            for task in self.task_types:
                unique_active_cases.update(self.waiting_cases[task])

            reward += -len(unique_active_cases) * self.tau # Not needed to multiply by constant tau but kept for consistency with SMDP
        events, probs = zip(*list(transformed_evolutions.items()))
        if self.crn:
            evolution = self.crn.choice(events, weights=probs)
        else:
            evolution = np.random.choice(events, p=probs)
        
        # The state does not return to itself, so we can process the action
        if evolution != 'return_to_state':
            next_task = None
            self.locked_action = None
            # process the action, 'postpone' and 'do nothing', do nothing to the state.
            if action not in ['postpone', 'do_nothing']:       
                if isinstance(action, str):
                    actions = [action]
                else:
                    actions = action
                for action in actions:
                    resource, task = action[0:-1], action[-1]
                    if self.waiting_cases[task]:
                        case_id = self.waiting_cases[task].pop(0)
                        getattr(self, f'processing_{resource}').append((case_id, task))
                        if self.reporter:
                            self.reporter.callback(case_id, task, '<task:start>', self.total_time, resource)
                    else:
                        raise Exception(f'No cases waiting for task {task} to be processed by resource {resource}')
            
            # the total time changes before the evolution happens
            self.total_time += self.tau

            if evolution == 'arrival':
                if self.track_cycle_times:
                    self.arrival_times[self.total_arrivals] = self.total_time

                # sample the first task from the transition matrix
                next_tasks = self.sample_next_task('Start')
                for task in next_tasks:
                    self.waiting_cases[task].append(self.total_arrivals)
                if self.reporter is not None:
                    self.reporter.callback(self.total_arrivals, 'start', '<start_event>', self.total_time)
                self.total_arrivals += 1
            else:
                resource, task = evolution[0:-1], evolution[-1]                
                case_id = getattr(self, f'processing_{resource}').pop(0)[0]
                next_tasks = self.sample_next_task(task, case_id)
                if self.config_type == 'parallel':
                    self.partially_completed_cases.append(case_id)
                if self.reporter:
                    self.reporter.callback(case_id, task, '<task:complete>', self.total_time, resource)
                for next_task in next_tasks:
                    if next_task and next_task != 'Complete':
                        self.waiting_cases[next_task].append(case_id)
                    elif next_task == 'Complete':
                        if self.track_cycle_times:
                            self.cycle_times[case_id] = self.total_time - self.arrival_times[case_id]                      
                        if self.reward_function == 'case_cycle_time':                            
                            reward += -self.cycle_times[case_id]
                        elif self.reward_function == 'inverse_case_cycle_time':
                            reward += 1/(1 + self.cycle_times[case_id])
                        if self.reporter:
                            self.reporter.callback(case_id, 'complete', '<end_event>', self.total_time)
            self.episodic_reward += reward
            return self.observation(), reward, self.is_done(), False, None
        else: #  The state returns to itself, so we lock the action for the next step
            self.locked_action = original_action
            self.total_time += self.tau
            self.episodic_reward += reward
            return self.observation(), reward, self.is_done(), False, None

if __name__ == '__main__':
    from heuristic_policies import fifo_policy, random_policy, greedy_policy, threshold_policy
    nr_replications = 1
    avg_cycle_times = []
    total_rewards = []
    for _ in range(nr_replications):
        #reporter = EventLogReporter("mdp_log_0.5.txt")
        reporter = ProcessReporter()
        tau = 1 / (1/2.0 + 1/1.4 + 1/1.8)
        env = MDP(5000, tau=tau, reporter=reporter, config_type='slow_server')

        done = False
        steps = 0
        total_reward = 0
        while not done:
            action = fifo_policy(env)            
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            time = env.total_time
            steps += 1
        avg_cycle_times.append(np.mean(list(env.cycle_times.values())))
        total_rewards.append(total_reward)
        print(np.mean(list(env.cycle_times.values())), total_reward)
        for task_type in env.task_types:
            print(f'Mean waiting time for task {task_type}: {np.mean([env.waiting_times[case_id][task_type] for case_id in env.waiting_times if task_type in env.waiting_times[case_id]])}')
            for resource in env.resource_pools[task_type]:
                print(f'Mean processing time for task {task_type} with resource {resource}: {np.mean([env.processing_times[case_id][(task_type, resource)] for case_id in env.processing_times if (task_type, resource) in env.processing_times[case_id]])}')
        reporter.close()
        reporter.print_result()
        print('\n')

    # print mean and 95% confidence interval of the average cycle time
    avg_cycle_times = np.array(avg_cycle_times)
    total_rewards = np.array(total_rewards)
    print('tau:', env.tau)
    print('mean CT:', np.mean(avg_cycle_times))
    print('95% CI CT:', [np.mean(avg_cycle_times)-1.96*np.std(avg_cycle_times)/np.sqrt(len(avg_cycle_times)), np.mean(avg_cycle_times)+1.96*np.std(avg_cycle_times)/np.sqrt(len(avg_cycle_times))])
    print('mean reward:', np.mean(total_rewards))
    print('95% CI reward:', [np.mean(total_rewards)-1.96*np.std(total_rewards)/np.sqrt(len(total_rewards)), np.mean(total_rewards)+1.96*np.std(total_rewards)/np.sqrt(len(total_rewards))])
    print('\n')
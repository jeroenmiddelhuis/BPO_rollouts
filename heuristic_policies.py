import random
import numpy as np

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
    if sum(action_mask) == 1: # only do nothing possible
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

    assignment = lowest_processing_times[0]#random.choice(lowest_processing_times)
    return assignment
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
        return env.action_space[action_mask.index(1)]
    possible_actions = [env.action_space[i] for i, action in enumerate(action_mask[:-2]) if action] # Excluding postpone, do_nothing
    possible_tasks = [action[-1] for action in possible_actions if isinstance(action, str)]

    # Identify the case that has been in the system the longest
    all_waiting_cases = sorted([(case_id, task) for task in env.waiting_cases for case_id in env.waiting_cases[task] if task in possible_tasks], key=lambda x: x[0])

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
                        possible_double_assignments_case += [double_assignment 
                                                             for double_assignment in possible_double_assignments 
                                                             if task2 in double_assignment[0] or task2 in double_assignment[1]]
                if len(possible_double_assignments_case) > 0:
                    assignment = random.choice(possible_double_assignments_case)
                    return tuple(assignment)
        else:
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

    # with 0.1 probability select the postpone action
    if random.random() < 0.05:
        return 'postpone'
    if len(possible_double_asignments) > 0:
        return tuple(random.choice(possible_double_asignments))
    else:
        return assignment

def threshold_policy(env, threshold=2):
    action_mask = env.action_mask()  
    if sum(action_mask) == 1: # only do nothing possible
        return env.action_space[action_mask.index(1)]
    if len(env.waiting_cases['a']) < threshold:
        assignment = 'r1a' if action_mask[env.action_space.index('r1a')] else 'postpone'
        return assignment
    else:
        if action_mask[env.action_space.index('r1a')]:
            return 'r1a'
        elif action_mask[env.action_space.index('r2a')]:
            return 'r2a'
        return 'postpone'
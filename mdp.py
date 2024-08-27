from reporters import EventLogReporter
from crn import CRN

PRINT_TRAJECTORY = False

class MDP:
    def __init__(self, nr_arrivals, tau=0.5, reporter=None, crn=None):
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
        self.waiting_cases = []
        self.processing_r1 = []
        self.processing_r2 = []
        self.total_time = 0
        self.total_arrivals = 0
        self.nr_arrivals = nr_arrivals
        self.tau = tau
        if crn is None:
            crn = CRN()
        self.crn = crn
        self.reporter = reporter

    def observation(self):
        return [1-len(self.processing_r1), 1-len(self.processing_r2), len(self.processing_r1), len(self.processing_r2), len(self.waiting_cases)]

    def reset(self):
        self.waiting_cases = []
        self.processing_r1 = []
        self.processing_r2 = []
        self.total_time = 0
        self.total_arrivals = 0
    
    def get_state(self):
        return (self.waiting_cases.copy(), self.processing_r1.copy(), self.processing_r2.copy(), self.total_time, self.total_arrivals, self.nr_arrivals)
    
    def set_state(self, state):
        self.waiting_cases = state[0].copy()
        self.processing_r1 = state[1].copy()
        self.processing_r2 = state[2].copy()
        self.total_time = state[3]
        self.total_arrivals = state[4]
        self.nr_arrivals = state[5]

    def from_state(state):
        smdp = MDP(0)
        smdp.set_state(state)
        return smdp

    def step(self, action):
        intermediate_state = self.observation() # Pointer or object?
        if PRINT_TRAJECTORY: print('intermediate state', intermediate_state)
        # create an intermediate state to calculate the expected reward and the next state
        # we use this state to calculate if the state actually transitions to the next state
        # or if the state remains the same. In latter, no resources are added/removed.
        if action[0]: # (r1, a)
            intermediate_state[0] = intermediate_state[0] - 1
            intermediate_state[2] = intermediate_state[2] + 1
            intermediate_state[4] = intermediate_state[4] - 1
        if action[1]: # (r2, a)
            intermediate_state[1] = intermediate_state[1] - 1
            intermediate_state[3] = intermediate_state[3] + 1
            intermediate_state[4] = intermediate_state[4] - 1
        if PRINT_TRAJECTORY: print(action)
        if PRINT_TRAJECTORY: print('post action state', intermediate_state)

        sum_of_rates = self.arrivals_coming() * 0.5 + 1/1.8 * intermediate_state[2] + 1/10 * intermediate_state[3]
        if sum_of_rates == 0:
            return self.observation(), 0, self.is_done(), False, None
        
        if PRINT_TRAJECTORY: print('arrival coming', self.arrivals_coming(), self.total_arrivals)
        if PRINT_TRAJECTORY: print('sum of rates', sum_of_rates)

        p_arrival = (self.arrivals_coming() * 0.5) / sum_of_rates
        p_r1a_completion = (1/1.8 * intermediate_state[2]) / sum_of_rates
        p_r2a_completion = (1/10 * intermediate_state[3]) / sum_of_rates  
        if PRINT_TRAJECTORY: print('pre probabilities', p_arrival, p_r1a_completion, p_r2a_completion)

        expected_event_time = 1 / sum_of_rates
        expected_reward = -expected_event_time * (intermediate_state[4] + intermediate_state[2] + intermediate_state[3]) # waiting cases + r1 processing + r2 processing = total active cases
        if PRINT_TRAJECTORY: print('expected reward', expected_reward)
        if PRINT_TRAJECTORY: print('expected event time', expected_event_time)

        p_arrival_transformed = self.tau / expected_event_time * p_arrival
        p_r1a_completion_transformed = self.tau / expected_event_time * p_r1a_completion
        p_r2a_completion_transformed = self.tau / expected_event_time * p_r2a_completion
        p_return_to_state = 1 - self.tau / expected_event_time
        if PRINT_TRAJECTORY: print('post probabilities', p_arrival_transformed, p_r1a_completion_transformed, p_r2a_completion_transformed, p_return_to_state)
        reward_rate = expected_reward / expected_event_time
        expected_reward_transformed = reward_rate * self.tau

        evolution = self.crn.choice(['arrival', 'r1a', 'r2a', 'return_to_state'], weights=[p_arrival_transformed, p_r1a_completion_transformed, p_r2a_completion_transformed, p_return_to_state])
        if PRINT_TRAJECTORY: print('evolution', evolution, '\n')
        
        if evolution != 'return_to_state':
            # process the action, 'postpone' and 'do nothing', do nothing to the state.
            if action[0]: # (r1, a)
                self.processing_r1.append(self.waiting_cases.pop(0))
                if self.reporter is not None:
                    self.reporter.callback(self.processing_r1[-1], 'a', '<task:start>', self.total_time)
            elif action[1]: # (r2, a)
                self.processing_r2.append(self.waiting_cases.pop(0))
                if self.reporter is not None:
                    self.reporter.callback(self.processing_r2[-1], 'a', '<task:start>', self.total_time)

            # now calculate the next state and how long it takes to reach that state
            # the time is to the next state is exponentially distributed with rate 
            # min(lambda, mu_(r1, a) * active r1, mu_(r2, a) * active r2) = 
            # exponential(0.5 + 1/1.8 * len(self.nr_processing_r1) + 1/10 * len(self.nr_processing_r2))
            # this is the probability of the next state being the consequence of:
            # an arrival, r1 processing the task or r2 processing the task
            # the probability of one of these evolutions happening is proportional to the rate of that evolution.
            if sum_of_rates == 0:
                return self.observation(), expected_reward, self.is_done(), False, None
            else:
                if evolution == 'arrival':
                    self.waiting_cases.append(self.total_arrivals)
                    if self.reporter is not None:
                        self.reporter.callback(self.total_arrivals, 'start', '<event:complete>', self.total_time)
                    self.total_arrivals += 1
                elif evolution == 'r1a':
                    case_id = self.processing_r1.pop(0)
                    if self.reporter is not None:
                        self.reporter.callback(case_id, 'a', '<task:complete>', self.total_time, 'r1')
                elif evolution == 'r2a':
                    case_id = self.processing_r2.pop(0)
                    if self.reporter is not None:
                        self.reporter.callback(case_id, 'a', '<task:complete>', self.total_time, 'r2')
                return self.observation(), expected_reward_transformed, self.is_done(), False, None
        else:
            self.total_time += self.tau
            return self.observation(), expected_reward_transformed, self.is_done(), False, None

    def arrivals_coming(self):
        return 1 if self.total_arrivals < self.nr_arrivals else 0

    def action_mask(self):
        # (r1, a) is only possible if there is a task waiting and r1 is available
        # (r2, a) is only possible if there is a task waiting and r2 is available
        # postpone is only possible if there is something to postpone, i.e. there is a task waiting and a resource is available
        # do nothing is only possible if nothing can be done, i.e. there is no task waiting or no resource is available
        a1_possible = len(self.waiting_cases) > 0 and len(self.processing_r1) == 0
        a2_possible = len(self.waiting_cases) > 0 and len(self.processing_r2) == 0
        postpone_possible = (self.arrivals_coming() > 0) and (a1_possible or a2_possible)
        do_nothing_possible = not (a1_possible or a2_possible or postpone_possible)
        return [a1_possible, a2_possible, postpone_possible, do_nothing_possible]
    
    def is_done(self):
        """
        The simulation is done if we have reached the maximum number of arrivals and there are no more tasks to process.
        """
        return not self.arrivals_coming() and len(self.waiting_cases) == 0 and len(self.processing_r1) == 0 and len(self.processing_r2) == 0
        

def random_policy(env):
    action_mask = env.action_mask()
    action = [0] * len(action_mask)
    action_index = env.crn.choice([i for i in range(len(action_mask)) if action_mask[i]])
    action[action_index] = 1
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
    reporter = EventLogReporter("test.csv")
    tau = 0.5
    env = MDP(2500, tau=tau, reporter=reporter)

    done = False
    steps = 0
    max_steps = 100000
    while steps < max_steps and not done:        
        action = greedy_policy(env)
        
        state, reward, done, _, _ = env.step(action)
        time = env.total_time

        # print(action, state, reward, time)

        steps += 1
    
    reporter.close()
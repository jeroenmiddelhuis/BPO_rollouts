import unittest
import mdp

# Actions are ['r1a', 'r1b', 'r2a', 'r2b', 'postpone', 'nothing']	
class MDPTests(unittest.TestCase):

    def test_smdp_observation(self):
        env = mdp.MDP(2, 'slow_server')
        self.assertEqual(env.observation(), [1, 1, 0, 0, 0, 0])

    def test_smdp_action_mask_initial_state(self):
        env = mdp.MDP(2, 'slow_server')
        self.assertEqual(env.action_mask(), [False, False, False, False, False, True])
    
    def test_smdp_action_mask_state_a_waiting(self):
        env = mdp.MDP(2, 'slow_server')
        env.waiting_cases = {'a': [1], 'b': []}
        print(env.state_space)
        self.assertEqual(env.action_mask(), [True, False, True, False, True, False])
    
    def test_smdp_action_mask_state_b_waiting(self):
        env = mdp.MDP(2, 'slow_server')
        env.waiting_cases = {'a': [], 'b': [1]}
        self.assertEqual(env.action_mask(), [False, True, False, True, True, False])

    def test_smdp_action_mask_state_one_waiting_r1_processing(self):
        env = mdp.MDP(210, 'slow_server')
        env.waiting_cases = {'a': [2], 'b': []}
        env.processing_r1 = [1]
        self.assertEqual(env.action_mask(), [False, False, True, False, True, False])

    def test_smdp_action_mask_state_one_waiting_r2_processing(self):
        env = mdp.MDP(2, 'slow_server')
        env.waiting_cases = {'a': [2], 'b': []}
        env.processing_r2 = [1]
        self.assertEqual(env.action_mask(), [True, False, False, False, True, False])

    def test_get_state_set_state(self):
        env = mdp.MDP(2, 'slow_server')
        env.waiting_cases = {'a': [2], 'b': []}
        env.processing_r1 = [1]
        env.processing_r2 = [2]
        env.total_time = 10
        env.total_arrivals = 20
        env.nr_arrivals = 30
        state = env.get_state()
        self.assertEqual(state, ({'a': [2], 'b': []}, [1], [2], 10, 20, 30))
        env.reset()
        env.set_state(state)
        self.assertEqual(env.get_state(), state)

    def test_env_reset(self):
        env = mdp.MDP(10, 'slow_server')
        env.waiting_cases = {'a': [2], 'b': [0]}
        env.processing_r1 = [1]
        env.processing_r2 = [1]
        env.total_time = 10
        env.total_arrivals = 20
        env.nr_arrivals = 30 # overwritten nr_arrivals should be reverted to original arrivales
        env.reset()
        self.assertEqual(env.get_state(), ({'a': [], 'b': []}, [], [], 0, 0, 10))

    def test_env_step_start_r1(self):
        env = mdp.MDP(2, 'slow_server')
        env.waiting_cases = {'a': [1], 'b': []}
        env.total_time = 10
        env.total_arrivals = 2
        env.nr_arrivals = 30
        state = env.get_state()
        # Due to randomness, the step may have different results.
        # Let's execute 100 times and check if it leads to:
        # - at least once a state where r1 is processing, r2 is idle and waiting_cases has the next arriving case (with id 2)
        # - at least once a state where r1 is done processing, r2 is idle, and no waiting cases
        found_r1_processing = False
        found_r1_done = False
        for _ in range(100):
            env.set_state(state)
            env.step((1, 0, 0, 0, 0, 0))
            found_r1_processing = found_r1_processing or (env.processing_r1 == [(1, 'a')] and env.processing_r2 == [] and env.waiting_cases['a'] == [2] and env.total_arrivals == 3 and env.total_time > 10)
            found_r1_done = found_r1_done or (env.processing_r1 == [] and env.processing_r2 == [] and env.waiting_cases['a'] == [] and env.waiting_cases['b'] == [1] and env.total_arrivals == 2 and env.total_time > 10)
            if env.processing_r1 == [1]:
                self.assertEqual(env.observation(), [0, 1, 1, 0, 1, 0])
            if env.processing_r1 == []:
                self.assertTrue(env.observation() == [1, 1, 0, 0, 0, 1] or env.observation() == [1, 1, 0, 0, 1, 0]) # done processing or return to state
        self.assertTrue(found_r1_processing)
        self.assertTrue(found_r1_done)

    def test_env_step_start_r2(self):
        env = mdp.MDP(2, 'slow_server')
        env.waiting_cases = {'a': [1], 'b': []}
        env.total_time = 10
        env.total_arrivals = 2
        env.nr_arrivals = 30
        state = env.get_state()
        # Due to randomness, the step may have different results.
        # Let's execute 100 times and check if it leads to:
        # - at least once a state where r2 is processing, r1 is idle and waiting_cases has the next arriving case (with id 2)
        # - at least once a state where r2 is done processing, r1 is idle, and no waiting cases
        found_r2_processing = False
        found_r2_done = False
        for _ in range(100):
            env.set_state(state)
            env.step((0, 0, 1, 0, 0, 0))
            found_r2_processing = found_r2_processing or (env.processing_r1 == [] and env.processing_r2 == [(1, 'a')] and env.waiting_cases['a'] == [2] and env.total_arrivals == 3 and env.total_time > 10)
            found_r2_done = found_r2_done or (env.processing_r1 == [] and env.processing_r2 == [] and env.waiting_cases['a'] == [] and env.waiting_cases['b'] == [1] and env.total_arrivals == 2 and env.total_time > 10)
            if env.processing_r2 == [1]:
                self.assertEqual(env.observation(), [1, 0, 0, 2, 1, 0])
            if env.processing_r2 == []:
                self.assertTrue(env.observation() == [1, 1, 0, 0, 0, 1] or env.observation() == [1, 1, 0, 0, 1, 0]) # done processing or return to state
        self.assertTrue(found_r2_processing)
        self.assertTrue(found_r2_done)
    
    def test_env_step_postpone(self):
        env = mdp.MDP(2, 'slow_server')
        env.waiting_cases = {'a': [1], 'b': []}
        env.total_time = 10
        env.total_arrivals = 2
        env.nr_arrivals = 30
        state = env.get_state()
        for _ in range(100):
            env.set_state(state)
            env.step((0, 0, 0, 0, 1, 0))
            # Because there is nothing else to do, this step must lead to a new arrival
            found_postpone_arrival = (env.processing_r1 == [] and env.processing_r2 == [] and env.waiting_cases['a'] == [1,2] and env.total_arrivals == 3 and env.total_time > 10)
            found_postpone_return = (env.processing_r1 == [] and env.processing_r2 == [] and env.waiting_cases['a'] == [1] and env.total_arrivals == 2 and env.total_time == 10 + env.tau)
        self.assertTrue(found_postpone_arrival or found_postpone_return)
    
    def test_env_step_do_nothing(self):
        env = mdp.MDP(2, 'slow_server')
        env.waiting_cases = {'a': [1], 'b': []}
        env.total_time = 10
        env.total_arrivals = 2
        env.nr_arrivals = 30
        state = env.get_state()
        for _ in range(100):
            env.set_state(state)
            env.step((0, 0, 0, 0, 0, 1))
            # Because there is nothing else to do, this step must lead to a new arrival
            found_postpone_arrival = (env.processing_r1 == [] and env.processing_r2 == [] and env.waiting_cases['a'] == [1,2] and env.total_arrivals == 3 and env.total_time > 10)
            found_postpone_return = (env.processing_r1 == [] and env.processing_r2 == [] and env.waiting_cases['a'] == [1] and env.total_arrivals == 2 and env.total_time == 10 + env.tau)
        self.assertTrue(found_postpone_arrival or found_postpone_return)

    def test_env_step_do_postpone_with_processing_r1(self):
        env = mdp.MDP(2, 'slow_server')
        env.waiting_cases = {'a': [1], 'b': []}
        env.processing_r1 = [(0, 'a')]
        env.total_time = 10
        env.total_arrivals = 2
        env.nr_arrivals = 30
        state = env.get_state()
        # Due to randomness, the step may have different results.
        # Let's execute 100 times and check if it leads to:
        # - at least once a state where r1 is processing, r2 is idle and waiting_cases has the next arriving case (with id 2)
        # - at least once a state where r1 is done processing, r2 is idle, and no waiting cases
        found_r1_processing = False
        found_r1_done = False
        for _ in range(100):
            env.set_state(state)
            env.step((0, 0, 0, 0, 1, 0))
            found_r1_processing = found_r1_processing or (env.processing_r1 == [(0, 'a')] and env.processing_r2 == [] and env.waiting_cases['a'] == [1,2] and env.total_arrivals == 3 and env.total_time > 10)
            found_r1_done = found_r1_done or (env.processing_r1 == [] and env.processing_r2 == [] and env.waiting_cases['a'] == [1] and env.waiting_cases['b'] == [0] and env.total_arrivals == 2 and env.total_time > 10)
        self.assertTrue(found_r1_processing)
        self.assertTrue(found_r1_done)

    # def test_greedy_nothing_possible(self):
    #     env = mdp.MDP(2, 'slow_server')
    #     action = smdp.greedy_policy(env)
    #     self.assertEqual(action, [0, 0, 0, 1])

    # def test_greedy_r1_possible(self):
    #     env = mdp.MDP(2, 'slow_server')
    #     env.waiting_cases = [1]
    #     env.processing_r2 = [0]
    #     action = smdp.greedy_policy(env)
    #     self.assertEqual(action, [1, 0, 0, 0])

    # def test_greedy_r2_possible(self):
    #     env = mdp.MDP(2, 'slow_server')
    #     env.waiting_cases = [1]
    #     env.processing_r1 = [0]
    #     action = smdp.greedy_policy(env)
    #     self.assertEqual(action, [0, 1, 0, 0])

    # def test_greedy_r1_and_r2_possible(self):
    #     env = mdp.MDP(2, 'slow_server')
    #     env.waiting_cases = [1]
    #     action = smdp.greedy_policy(env)
    #     self.assertEqual(action, [1, 0, 0, 0])

if __name__ == '__main__':
    unittest.main()
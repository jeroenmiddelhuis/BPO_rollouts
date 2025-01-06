import unittest
import smdp

# Actions are ['r1a', 'r1b', 'r2a', 'r2b', 'postpone', 'nothing']	
class SMDPTests(unittest.TestCase):

    def test_smdp_observation(self):
        env = smdp.SMDP(2, 'slow_server')
        self.assertEqual(env.observation(), [1, 1, 0, 0, 0, 0, 0, 0])

    def test_smdp_action_mask_initial_state(self):
        env = smdp.SMDP(2, 'slow_server')
        self.assertEqual(env.action_mask(), [False, False, False, False, False, True])
    
    def test_smdp_action_mask_state_a_waiting(self):
        env = smdp.SMDP(2, 'slow_server')
        env.waiting_cases = {'a': [1], 'b': []}
        self.assertEqual(env.action_mask(), [True, False, True, False, True, False])
    
    def test_smdp_action_mask_state_b_waiting(self):
        env = smdp.SMDP(2, 'slow_server')
        env.waiting_cases = {'a': [], 'b': [1]}
        self.assertEqual(env.action_mask(), [False, True, False, True, True, False])

    def test_smdp_action_mask_state_one_waiting_r1_processing(self):
        env = smdp.SMDP(210, 'slow_server')
        env.waiting_cases = {'a': [2], 'b': []}
        env.processing_r1 = [1]
        self.assertEqual(env.action_mask(), [False, False, True, False, True, False])

    def test_smdp_action_mask_state_one_waiting_r2_processing(self):
        env = smdp.SMDP(2, 'slow_server')
        env.waiting_cases = {'a': [2], 'b': []}
        env.processing_r2 = [1]
        self.assertEqual(env.action_mask(), [True, False, False, False, True, False])

    def test_get_state_set_state(self):
        env = smdp.SMDP(2, 'slow_server')
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
        env = smdp.SMDP(10, 'slow_server')
        env.waiting_cases = {'a': [2], 'b': [0]}
        env.processing_r1 = [1]
        env.processing_r2 = [1]
        env.total_time = 10
        env.total_arrivals = 20
        env.nr_arrivals = 30 # overwritten nr_arrivals should be reverted to original arrivales
        env.reset()
        self.assertEqual(env.get_state(), ({'a': [], 'b': []}, [], [], 0, 0, 10))

    def test_env_step_start_r1(self):
        env = smdp.SMDP(2, 'slow_server')
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
                self.assertEqual(env.observation(), [1, 1, 0, 0, 0, 1])
        self.assertTrue(found_r1_processing)
        self.assertTrue(found_r1_done)

    def test_env_step_start_r2(self):
        env = smdp.SMDP(2, 'slow_server')
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
                self.assertEqual(env.observation(), [1, 1, 0, 0, 0, 1])
        self.assertTrue(found_r2_processing)
        self.assertTrue(found_r2_done)
    
    def test_env_step_postpone(self):
        env = smdp.SMDP(2, 'slow_server')
        env.waiting_cases = {'a': [1], 'b': []}
        env.total_time = 10
        env.total_arrivals = 2
        env.nr_arrivals = 30
        env.step((0, 0, 0, 0, 1, 0))
        # Because there is nothing else to do, this step must lead to a new arrival
        found_postpone = (env.processing_r1 == [] and env.processing_r2 == [] and env.waiting_cases['a'] == [1,2] and env.total_arrivals == 3 and env.total_time > 10)
        self.assertTrue(found_postpone)
    
    def test_env_step_do_nothing(self):
        env = smdp.SMDP(2, 'slow_server')
        env.waiting_cases = {'a': [1], 'b': []}
        env.total_time = 10
        env.total_arrivals = 2
        env.nr_arrivals = 30
        env.step((0, 0, 0, 0, 0, 1))
        # Because there is nothing else to do, this step must lead to a new arrival
        found_do_nothing = (env.processing_r1 == [] and env.processing_r2 == [] and env.waiting_cases['a'] == [1,2] and env.total_arrivals == 3 and env.total_time > 10)
        self.assertTrue(found_do_nothing)

    def test_env_step_do_postpone_with_processing_r1(self):
        env = smdp.SMDP(2, 'slow_server')
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
    #     env = smdp.SMDP(2, 'slow_server')
    #     action = smdp.greedy_policy(env)
    #     self.assertEqual(action, [0, 0, 0, 1])

    # def test_greedy_r1_possible(self):
    #     env = smdp.SMDP(2, 'slow_server')
    #     env.waiting_cases = [1]
    #     env.processing_r2 = [0]
    #     action = smdp.greedy_policy(env)
    #     self.assertEqual(action, [1, 0, 0, 0])

    # def test_greedy_r2_possible(self):
    #     env = smdp.SMDP(2, 'slow_server')
    #     env.waiting_cases = [1]
    #     env.processing_r1 = [0]
    #     action = smdp.greedy_policy(env)
    #     self.assertEqual(action, [0, 1, 0, 0])

    # def test_greedy_r1_and_r2_possible(self):
    #     env = smdp.SMDP(2, 'slow_server')
    #     env.waiting_cases = [1]
    #     action = smdp.greedy_policy(env)
    #     self.assertEqual(action, [1, 0, 0, 0])

    def test_slow_server(self):
        env = smdp.SMDP(1, 'slow_server')
        observation = env.observation()
        self.assertEqual(observation, [1, 1, 0, 0, 0, 0, 0, 0])
        # initially nothing is possible and we are waiting for an arrival
        self.assertEqual(env.action_mask(), [False, False, False, False, False, True])
        observation, reward, done, _, _ = env.step((0, 0, 0, 0, 0, 1))
        self.assertEqual(observation, [1, 1, 0, 0, 0, 0, 1, 0])
        # now activity a is possible, postpone is possible but do nothing is not possible
        self.assertEqual(env.action_mask(), [True, False, True, False, True, False])
        observation, reward, done, _, _ = env.step((1, 0, 0, 0, 0, 0))
        self.assertTrue(observation == [1, 1, 0, 0, 0, 0, 0, 1] or observation == [1, 1, 1, 0, 0, 0, 1, 0])
        # now activity b is possible, postpone and do nothing are not possible
        self.assertEqual(env.action_mask(), [False, True, False, True, False, False])
        observation, reward, done, _, _ = env.step((0, 1, 0, 0, 0, 0))
        self.assertEqual(observation, [1, 1, 0, 0, 0, 0])
        # now nothing is possible and we are done
        self.assertEqual(env.action_mask(), [False, False, False, False, False, True])
        self.assertTrue(done)

    def test_single_activity(self):
        env = smdp.SMDP(1, 'single_activity')
        observation = env.observation()
        self.assertEqual(observation, [1, 1, 0, 0, 0])
        # initially nothing is possible and we are waiting for an arrival
        self.assertEqual(env.action_mask(), [False, False, False, True])
        observation, reward, done, _, _ = env.step((0, 0, 0, 1))
        self.assertEqual(observation, [1, 1, 0, 0, 1])
        # now activity a is possible, postpone and do nothing are not possible
        self.assertEqual(env.action_mask(), [True, False, False, False])
        observation, reward, done, _, _ = env.step((1, 0, 0, 0))
        self.assertEqual(observation, [1, 1, 0, 0, 0])
        # now nothing is possible and we are done
        self.assertEqual(env.action_mask(), [False, False, False, True])
        self.assertTrue(done)

    def test_parallel(self):
        env = smdp.SMDP(1, 'parallel')
        observation = env.observation()
        self.assertEqual(observation, [1, 1, 0, 0, 0, 0])
        # initially nothing is possible and we are waiting for an arrival
        self.assertEqual(env.action_mask(), [False, False, False, False, False, True])
        observation, reward, done, _, _ = env.step((0, 0, 0, 0, 0, 1))
        self.assertEqual(observation, [1, 1, 0, 0, 1, 1])
        # now activity a and b are possible, postpone and do nothing are not possible
        self.assertEqual(env.action_mask(), [True, True, True, True, False, False])
        observation, reward, done, _, _ = env.step((1, 0, 0, 0, 0, 0))
        # now activity b is possible, postpone and do nothing are not possible
        self.assertEqual(env.action_mask(), [False, False, False, True, False, False])
        observation, reward, done, _, _ = env.step((0, 1, 0, 0, 0, 0))
        # now nothing is possible and we are done
        self.assertEqual(env.action_mask(), [False, False, False, False, False, True])
        self.assertTrue(done)

    def test_n_system(self):
        env = smdp.SMDP(1, 'n_system')
        observation = env.observation()
        self.assertEqual(observation, [1, 1, 0, 0, 0, 0])
        # initially nothing is possible and we are waiting for an arrival
        self.assertEqual(env.action_mask(), [False, False, False, False, True])
        observation, reward, done, _, _ = env.step((0, 0, 0, 0, 1))
        # now there are two possibilities, either an activity a or an activity b arrives
        # if activity a arrives
        if observation[4] == 1:
            self.assertEqual(observation, [1, 1, 0, 0, 1, 0])
            # only r1 can be assigned
            self.assertEqual(env.action_mask(), [True, False, False, False, False])
            observation, reward, done, _, _ = env.step((1, 0, 0, 0, 0))
            self.assertEqual(observation, [1, 1, 0, 0, 0, 0])
            # after performing a we are done
            self.assertEqual(env.action_mask(), [False, False, False, False, True])
            self.assertTrue(done)
        else:
            self.assertEqual(observation, [1, 1, 0, 0, 0, 1])
            # r1 and r2 can be assigned to b
            self.assertEqual(env.action_mask(), [False, True, True, False, False])
            observation, reward, done, _, _ = env.step((0, 1, 0, 0, 0))
            self.assertEqual(observation, [1, 1, 0, 0, 0, 0])
            # after performing b we are done
            self.assertEqual(env.action_mask(), [False, False, False, False, True])
            self.assertTrue(done)

if __name__ == '__main__':
    unittest.main()
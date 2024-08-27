import unittest
import smdp

class SMDPTests(unittest.TestCase):

    def test_smdp_observation(self):
        env = smdp.SMDP(2)
        self.assertEqual(env.observation(), [1, 1, 0, 0, 0])

    def test_smdp_action_mask_initial_state(self):
        env = smdp.SMDP(2)
        self.assertEqual(env.action_mask(), [False, False, False, True])
    
    def test_smdp_action_mask_state_one_waiting(self):
        env = smdp.SMDP(2)
        env.waiting_cases = [1]
        self.assertEqual(env.action_mask(), [True, True, True, False])
    
    def test_smdp_action_mask_state_one_waiting_r1_processing(self):
        env = smdp.SMDP(2)
        env.waiting_cases = [1]
        env.processing_r1 = [1]
        self.assertEqual(env.action_mask(), [False, True, True, False])

    def test_smdp_action_mask_state_one_waiting_r2_processing(self):
        env = smdp.SMDP(2)
        env.waiting_cases = [1]
        env.processing_r2 = [1]
        self.assertEqual(env.action_mask(), [True, False, True, False])

    def test_get_state_set_state(self):
        env = smdp.SMDP(2)
        env.waiting_cases = [1]
        env.processing_r1 = [1]
        env.processing_r2 = [1]
        env.total_time = 10
        env.total_arrivals = 20
        env.nr_arrivals = 30
        state = env.get_state()
        self.assertEqual(state, ([1], [1], [1], 10, 20, 30))
        env.reset()
        env.set_state(state)
        self.assertEqual(env.get_state(), state)

    def test_env_reset(self):
        env = smdp.SMDP(2)
        env.waiting_cases = [1]
        env.processing_r1 = [1]
        env.processing_r2 = [1]
        env.total_time = 10
        env.total_arrivals = 20
        env.nr_arrivals = 30
        env.reset()
        self.assertEqual(env.get_state(), ([], [], [], 0, 0, 30))

    def test_env_step_start_r1(self):
        env = smdp.SMDP(2)
        env.waiting_cases = [1]
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
            env.step((1, 0, 0, 0))
            found_r1_processing = found_r1_processing or (env.processing_r1 == [1] and env.processing_r2 == [] and env.waiting_cases == [2] and env.total_arrivals == 3 and env.total_time > 10)
            found_r1_done = found_r1_done or (env.processing_r1 == [] and env.processing_r2 == [] and env.waiting_cases == [] and env.total_arrivals == 2 and env.total_time > 10)
            if env.processing_r1 == [1]:
                self.assertEqual(env.observation(), [0, 1, 1, 0, 1])
            if env.processing_r1 == []:
                self.assertEqual(env.observation(), [1, 1, 0, 0, 0])
        self.assertTrue(found_r1_processing)
        self.assertTrue(found_r1_done)

    def test_env_step_start_r2(self):
        env = smdp.SMDP(2)
        env.waiting_cases = [1]
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
            env.step((0, 1, 0, 0))
            found_r2_processing = found_r2_processing or (env.processing_r1 == [] and env.processing_r2 == [1] and env.waiting_cases == [2] and env.total_arrivals == 3 and env.total_time > 10)
            found_r2_done = found_r2_done or (env.processing_r1 == [] and env.processing_r2 == [] and env.waiting_cases == [] and env.total_arrivals == 2 and env.total_time > 10)
            if env.processing_r2 == [1]:
                self.assertEqual(env.observation(), [1, 0, 0, 1, 1])
            if env.processing_r2 == []:
                self.assertEqual(env.observation(), [1, 1, 0, 0, 0])
        self.assertTrue(found_r2_processing)
        self.assertTrue(found_r2_done)
    
    def test_env_step_postpone(self):
        env = smdp.SMDP(2)
        env.waiting_cases = [1]
        env.total_time = 10
        env.total_arrivals = 2
        env.nr_arrivals = 30
        env.step((0, 0, 1, 0))
        # Because there is nothing else to do, this step must lead to a new arrival
        found_postpone = (env.processing_r1 == [] and env.processing_r2 == [] and env.waiting_cases == [1,2] and env.total_arrivals == 3 and env.total_time > 10)
        self.assertTrue(found_postpone)
    
    def test_env_step_do_nothing(self):
        env = smdp.SMDP(2)
        env.waiting_cases = [1]
        env.total_time = 10
        env.total_arrivals = 2
        env.nr_arrivals = 30
        env.step((0, 0, 0, 1))
        # Because there is nothing else to do, this step must lead to a new arrival
        found_do_nothing = (env.processing_r1 == [] and env.processing_r2 == [] and env.waiting_cases == [1,2] and env.total_arrivals == 3 and env.total_time > 10)
        self.assertTrue(found_do_nothing)

    def test_env_step_do_postpone_with_processing_r1(self):
        env = smdp.SMDP(2)
        env.waiting_cases = [1]
        env.processing_r1 = [0]
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
            env.step((0, 0, 1, 0))
            found_r1_processing = found_r1_processing or (env.processing_r1 == [0] and env.processing_r2 == [] and env.waiting_cases == [1,2] and env.total_arrivals == 3 and env.total_time > 10)
            found_r1_done = found_r1_done or (env.processing_r1 == [] and env.processing_r2 == [] and env.waiting_cases == [1] and env.total_arrivals == 2 and env.total_time > 10)
        self.assertTrue(found_r1_processing)
        self.assertTrue(found_r1_done)

    def test_greedy_nothing_possible(self):
        env = smdp.SMDP(2)
        action = smdp.greedy_policy(env)
        self.assertEqual(action, [0, 0, 0, 1])

    def test_greedy_r1_possible(self):
        env = smdp.SMDP(2)
        env.waiting_cases = [1]
        env.processing_r2 = [0]
        action = smdp.greedy_policy(env)
        self.assertEqual(action, [1, 0, 0, 0])

    def test_greedy_r2_possible(self):
        env = smdp.SMDP(2)
        env.waiting_cases = [1]
        env.processing_r1 = [0]
        action = smdp.greedy_policy(env)
        self.assertEqual(action, [0, 1, 0, 0])

    def test_greedy_r1_and_r2_possible(self):
        env = smdp.SMDP(2)
        env.waiting_cases = [1]
        action = smdp.greedy_policy(env)
        self.assertEqual(action, [1, 0, 0, 0])

if __name__ == '__main__':
    unittest.main()
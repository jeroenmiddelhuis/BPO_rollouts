import unittest
import smdp

# Actions are ['r1a', 'r1b', 'r2a', 'r2b', 'postpone', 'nothing']	
class EquivalanceTests(unittest.TestCase):

    def test_smdp_evolutions_r1a_busy(self):
        env = smdp.SMDP(2, 'slow_server')
        env.waiting_cases = {'a': [1], 'b': []}
        env.processing_r1 = [(0, 'a')]
        env.total_time = 10
        env.total_arrivals = 2
        env.nr_arrivals = 30
        evolutions = {'arrival': 0.5,
                      'r1a': 1/1.4}
        sum_of_rates = sum(evolutions.values())
        for evolution, rate in evolutions.items():
            evolutions[evolution] = rate/sum_of_rates

        self.assertEqual(env.get_evolutions(env.processing_r1, env.processing_r2, env.arrivals_coming()), 
                         evolutions)

    def test_mdp_evolutions(self):
        env = smdp.SMDP(2, 'slow_server')
        env.waiting_cases = {'a': [1], 'b': []}
        env.processing_r1 = [(0, 'a')]
        env.total_time = 10
        env.total_arrivals = 2
        env.nr_arrivals = 30
        state = env.get_state()

        self.assertEqual(env.observation(), [1, 1, 0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
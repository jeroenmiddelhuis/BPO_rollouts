import unittest
import smdp, mdp
from smdp import greedy_policy

# Actions are ['r1a', 'r1b', 'r2a', 'r2b', 'postpone', 'nothing']	
class EquivalanceTests(unittest.TestCase):

    def test_evolution_rates_r1a_busy(self):
        env = smdp.SMDP(2, 'slow_server')
        env.waiting_cases = {'a': [1], 'b': []}
        env.processing_r1 = [(0, 'a')]
        env.total_time = 10
        env.total_arrivals = 2
        env.nr_arrivals = 30

        env2 = mdp.MDP(2, 'slow_server')
        env2.waiting_cases = {'a': [1], 'b': []}
        env2.processing_r1 = [(0, 'a')]
        env2.total_time = 10
        env2.total_arrivals = 2
        env2.nr_arrivals = 30

        evolution_rates = { 'arrival': 0.5,
                            'r1a': 1/1.4}

        self.assertEqual(env.get_evolution_rates(env.processing_r1, env.processing_r2, env.arrivals_coming()), 
                         evolution_rates)
        self.assertEqual(env2.get_evolution_rates(env2.processing_r1, env2.processing_r2, env2.arrivals_coming()), 
                         evolution_rates)
        self.assertEqual(env.get_evolution_rates(env.processing_r1, env.processing_r2, env.arrivals_coming()),
                         env2.get_evolution_rates(env2.processing_r1, env2.processing_r2, env2.arrivals_coming()))


    def test_evolution_rates_r2b_busy(self):
        env = smdp.SMDP(2, 'slow_server')
        env.waiting_cases = {'a': [1], 'b': []}
        env.processing_r2 = [(0, 'b')]
        env.total_time = 10
        env.total_arrivals = 2
        env.nr_arrivals = 30

        env2 = mdp.MDP(2, 'slow_server')
        env2.waiting_cases = {'a': [1], 'b': []}
        env2.processing_r2 = [(0, 'b')]
        env2.total_time = 10
        env2.total_arrivals = 2
        env2.nr_arrivals = 30

        evolution_rates = {'arrival': 0.5,
                            'r2b': 1/3.0}

        self.assertEqual(env.get_evolution_rates(env.processing_r1, env.processing_r2, env.arrivals_coming()), 
                         evolution_rates)
        self.assertEqual(env2.get_evolution_rates(env2.processing_r1, env2.processing_r2, env2.arrivals_coming()), 
                         evolution_rates)
        self.assertEqual(env.get_evolution_rates(env.processing_r1, env.processing_r2, env.arrivals_coming()),
                         env2.get_evolution_rates(env2.processing_r1, env2.processing_r2, env2.arrivals_coming()))


    def test_evolutions_r1a_busy(self):
        env = smdp.SMDP(2, 'slow_server')
        env.waiting_cases = {'a': [1], 'b': []}
        env.processing_r1 = [(0, 'a')]
        env.total_time = 10
        env.total_arrivals = 2
        env.nr_arrivals = 30

        env2 = mdp.MDP(2, 'slow_server')
        env2.waiting_cases = {'a': [1], 'b': []}
        env2.processing_r1 = [(0, 'a')]
        env2.total_time = 10
        env2.total_arrivals = 2
        env2.nr_arrivals = 30

        evolution_rates = { 'arrival': 0.5,
                            'r1a': 1/1.4}
        evolutions = {}
        sum_of_rates = sum(evolution_rates.values())
        for evolution, rate in evolution_rates.items():
            evolutions[evolution] = rate/sum_of_rates

        self.assertEqual(env.get_evolutions(env.processing_r1, env.processing_r2, env.arrivals_coming()), 
                         evolutions)
        self.assertEqual(env2.get_evolutions(env2.processing_r1, env2.processing_r2, env2.arrivals_coming()), 
                         evolutions)
        self.assertEqual(env.get_evolutions(env.processing_r1, env.processing_r2, env.arrivals_coming()),
                         env2.get_evolutions(env2.processing_r1, env2.processing_r2, env2.arrivals_coming()))
        

    def test_evolutions_r2b_busy(self):
        env = smdp.SMDP(2, 'slow_server')
        env.waiting_cases = {'a': [1], 'b': []}
        env.processing_r2 = [(0, 'b')]
        env.total_time = 10
        env.total_arrivals = 2
        env.nr_arrivals = 30

        env2 = mdp.MDP(2, 'slow_server')
        env2.waiting_cases = {'a': [1], 'b': []}
        env2.processing_r2 = [(0, 'b')]
        env2.total_time = 10
        env2.total_arrivals = 2
        env2.nr_arrivals = 30

        evolution_rates = {'arrival': 0.5,
                            'r2b': 1/3.0}
        evolutions = {}
        sum_of_rates = sum(evolution_rates.values())
        for evolution, rate in evolution_rates.items():
            evolutions[evolution] = rate/sum_of_rates

        self.assertEqual(env.get_evolutions(env.processing_r1, env.processing_r2, env.arrivals_coming()), 
                         evolutions)
        self.assertEqual(env2.get_evolutions(env2.processing_r1, env2.processing_r2, env2.arrivals_coming()), 
                         evolutions)
        self.assertEqual(env.get_evolutions(env.processing_r1, env.processing_r2, env.arrivals_coming()),
                         env2.get_evolutions(env2.processing_r1, env2.processing_r2, env2.arrivals_coming()))
        
    def test_transformed_evolutions_mdp_r1a_busy(self):
        env2 = mdp.MDP(2, 'slow_server')
        env2.waiting_cases = {'a': [1], 'b': []}
        env2.processing_r1 = [(0, 'a')]
        env2.total_time = 10
        env2.total_arrivals = 2
        env2.nr_arrivals = 30

        evolution_rates = { 'arrival': 0.5,
                            'r1a': 1/1.4}
        evolutions = {}
        sum_of_rates = sum(evolution_rates.values())
        for evolution, rate in evolution_rates.items():
            evolutions[evolution] = rate/sum_of_rates

        self.assertEqual(env2.transformed_evolutions(env2.processing_r1, env2.processing_r2, env2.arrivals_coming()), 
                         evolutions)


    def test_action_count(self):
        env = smdp.SMDP(10000, 'slow_server')

        done = False
        smdp_actions = {}
        while not done:        
            action = greedy_policy(env)
            state, reward, done, _, _ = env.step(action)
            action = env.action_space[action.index(1)]
            if action not in smdp_actions:
                smdp_actions[action] = 0
            smdp_actions[action] += 1

        env = mdp.MDP(10000, 'slow_server')
        state = env.observation()

        done = False
        mdp_actions = {}
        while not done:            
            action = greedy_policy(env)
            new_state, reward, done, _, _ = env.step(action)
            action = env.action_space[action.index(1)]
            if state != new_state: # only real actions. returning to state undoes the action
                if action not in mdp_actions:
                    mdp_actions[action] = 0
                mdp_actions[action] += 1
            state = new_state
        keys = smdp_actions.keys()
        for key in keys:
            self.assertAlmostEqual(smdp_actions[key], mdp_actions[key], delta=100)


if __name__ == '__main__':
    unittest.main()
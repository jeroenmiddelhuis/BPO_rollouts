import unittest
import rollouts
import smdp
from crn import CRN


class RolloutTestCase(unittest.TestCase):

    def test_rollout_with_greedy_policy(self):
        env = smdp.SMDP(2)
        reward = rollouts.rollout(env, smdp.greedy_policy)

        # at the end, the environment should be done
        # two cases must have arrived
        # no cases should be waiting and no resources should be processing
        # the time should be greater than 0
        # the reward should be float-equivalent to -time
        self.assertTrue(env.is_done())
        self.assertEqual(env.total_arrivals, 2)
        self.assertEqual(len(env.waiting_cases), 0)
        self.assertEqual(len(env.processing_r1), 0)
        self.assertEqual(len(env.processing_r2), 0)
        self.assertGreater(env.total_time, 0)
        self.assertAlmostEqual(reward, -env.total_time)

    def test_find_learning_sample(self):
        env = smdp.SMDP(2)
        sample = rollouts.find_learning_sample(env, smdp.greedy_policy, 10)
        # initially, no action is possible, so sample should be None
        self.assertIsNone(sample)
    
    def test_find_learning_sample_with_one_waiting_case(self):
        env = smdp.SMDP(10)
        env.step([0,0,0,1])
        # now a case must have arrived consequently r1, r2, and postpone are all possible
        self.assertEqual(env.action_mask(), [1,1,1,0])
        # since postpone does not take time, both postpone and r1 should be the best actions
        sample = rollouts.find_learning_sample(env, smdp.greedy_policy, 50)
        self.assertEqual(sample[0], [1,1,0,0,1])
        self.assertIn(sample[1], [(1,0,0,0), (0,0,1,0)])

    def test_multiple_rollouts_per_action(self):
        env = smdp.SMDP(1)
        env.step([0,0,0,1])  
        # now a case must have arrived consequently r1, r2 are possible
        # postpone is not possible, because it is the last case and we must process it
        observation, possible_actions, rewards = rollouts.multiple_rollouts_per_action(env, smdp.greedy_policy, 20)
        self.assertEqual(observation, [1,1,0,0,1])
        self.assertEqual(possible_actions, [(1,0,0,0), (0,1,0,0)])
        # assigning a case to r1 should be better than assigning it to r2
        rewards = {action: round(sum(rewards[action])) for action in rewards}
        self.assertGreater(rewards[(1,0,0,0)], rewards[(0,1,0,0)])

    def test_rollout_with_crn(self):
        crn = CRN()
        env = smdp.SMDP(5, crn=crn)
        rollout1 = rollouts.rollout_with_full_information(env, smdp.greedy_policy)
        crn.restart_sequence()
        env.reset()
        rollout2 = rollouts.rollout_with_full_information(env, smdp.greedy_policy)
        self.assertEqual(rollout1, rollout2)

if __name__ == '__main__':
    unittest.main()
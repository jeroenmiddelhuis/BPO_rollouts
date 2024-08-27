import unittest
from crn import CRN

class CRNTests(unittest.TestCase):

    def test_generate_uniform_unique_numbers(self):
        crn = CRN()
        numbers1 = [crn.generate_uniform() for i in range(10)]
        numbers2 = [crn.generate_uniform() for i in range(10)]
        self.assertNotEqual(numbers1, numbers2)

    def test_generate_uniform_same_numbers_after_restart(self):
        crn = CRN()
        numbers1 = [crn.generate_uniform() for i in range(20)]
        crn.restart_sequence()
        numbers2 = [crn.generate_uniform() for i in range(10)]
        numbers3 = [crn.generate_uniform() for i in range(10)]
        self.assertEqual(numbers1[:10], numbers2)
        self.assertEqual(numbers1[10:20], numbers3)

    def test_generate_uniform_different_numbers_after_reset(self):
        crn = CRN()
        numbers1 = [crn.generate_uniform() for i in range(20)]
        crn.restart_sequence()
        numbers2 = [crn.generate_uniform() for i in range(10)]
        crn.reset()
        numbers3 = [crn.generate_uniform() for i in range(10)]
        self.assertEqual(numbers1[:10], numbers2)
        self.assertNotEqual(numbers1[:10], numbers3)
        self.assertNotEqual(numbers1[10:20], numbers3)
        self.assertNotEqual(numbers2, numbers3)

    def test_generate_exponential_unique_numbers(self):
        crn = CRN()
        numbers1 = [crn.generate_exponential() for i in range(10)]
        numbers2 = [crn.generate_exponential() for i in range(10)]
        self.assertNotEqual(numbers1, numbers2)
    
    def test_generate_exponential_same_numbers_after_restart(self):
        crn = CRN()
        numbers1 = [crn.generate_exponential() for i in range(20)]
        crn.restart_sequence()
        numbers2 = [crn.generate_exponential() for i in range(10)]
        numbers3 = [crn.generate_exponential() for i in range(10)]
        self.assertEqual(numbers1[:10], numbers2)
        self.assertEqual(numbers1[10:20], numbers3)

    def test_generate_exponential_different_numbers_after_reset(self):
        crn = CRN()
        numbers1 = [crn.generate_exponential() for i in range(20)]
        crn.restart_sequence()
        numbers2 = [crn.generate_exponential() for i in range(10)]
        crn.reset()
        numbers3 = [crn.generate_exponential() for i in range(10)]
        self.assertEqual(numbers1[:10], numbers2)
        self.assertNotEqual(numbers1[:10], numbers3)
        self.assertNotEqual(numbers1[10:20], numbers3)
        self.assertNotEqual(numbers2, numbers3)
    
    def test_choice_unique_samples(self):
        crn = CRN()
        seq = [1, 2, 3, 4, 5]
        samples1 = [crn.choice(seq) for i in range(10)]
        samples2 = [crn.choice(seq) for i in range(10)]
        self.assertNotEqual(samples1, samples2)
    
    def test_choice_same_samples_after_restart(self):
        crn = CRN()
        seq = [1, 2, 3, 4, 5]
        samples1 = [crn.choice(seq) for i in range(20)]
        crn.restart_sequence()
        samples2 = [crn.choice(seq) for i in range(10)]
        samples3 = [crn.choice(seq) for i in range(10)]
        self.assertEqual(samples1[:10], samples2)
        self.assertEqual(samples1[10:20], samples3)
    
    def test_choice_different_samples_after_reset(self):
        crn = CRN()
        seq = [1, 2, 3, 4, 5]
        samples1 = [crn.choice(seq) for i in range(20)]
        crn.restart_sequence()
        samples2 = [crn.choice(seq) for i in range(10)]
        crn.reset()
        samples3 = [crn.choice(seq) for i in range(10)]
        self.assertEqual(samples1[:10], samples2)
        self.assertNotEqual(samples1[:10], samples3)
        self.assertNotEqual(samples1[10:20], samples3)
        self.assertNotEqual(samples2, samples3)
    
    def test_choice_all_elements(self):
        crn = CRN()
        seq = [1, 2, 3, 4, 5]
        samples = [crn.choice(seq) for i in range(1000)]
        self.assertEqual(set(samples), set(seq))
    
    def test_choice_all_elements_within_tolerance(self):
        crn = CRN()
        seq = [1, 2, 3, 4, 5]
        samples = [crn.choice(seq) for i in range(1000)]
        for i in seq:
            self.assertGreater(samples.count(i), 150)
            self.assertLess(samples.count(i), 250)
    
    def test_choice_weights_within_tolerance(self):
        crn = CRN()
        seq = [1, 2, 3]
        weights = [0.1, 0.45, 0.45]
        samples = [crn.choice(seq, weights) for i in range(1000)]
        self.assertGreater(samples.count(1), 50)
        self.assertLess(samples.count(1), 150)
        self.assertGreater(samples.count(2), 400)
        self.assertLess(samples.count(2), 600)
        self.assertGreater(samples.count(3), 400)
        self.assertLess(samples.count(3), 600)
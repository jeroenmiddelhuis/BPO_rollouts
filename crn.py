import random
import math


class CRN:
    def __init__(self):
        self.__random_generator = random.Random()
        self.__random_numbers = []
        self.__current_random_number = 0        

    def __sample(self):
        return self.__random_generator.random()
        if self.__current_random_number >= len(self.__random_numbers):
            self.__random_numbers.append(self.__random_generator.random())
        self.__current_random_number += 1
        print('self.__random_numbers', len(self.__random_numbers))
        return self.__random_numbers[self.__current_random_number-1]

    def restart_sequence(self):
        """
        Restarts the sequence of random numbers.
        The same sequence of random numbers will be generated again.
        """
        self.__current_random_number = 0

    def reset(self):
        """
        Resets the random number generator.
        A new sequence of random numbers will be generated.
        """
        self.__random_numbers = []
        self.__current_random_number = 0

    def generate_uniform(self, low=0.0, high=1.0):
        """
        Generates a random number from a uniform distribution between low and high (inclusive).
        """
        return low + (high - low) * self.__sample()
    
    def generate_normal(self, mean=0.0, std=1.0):
        """
        Generates a random number from a normal distribution with the given mean and standard deviation.
        """
        # Box-Muller transform
        u1 = self.__sample()
        u2 = self.__sample()
        z0 = (-2 * math.log(u1))**0.5 * math.cos(2 * math.pi * u2)
        return mean + z0 * std

    def generate_exponential(self, rate=1.0):
        """
        Generates a random number from an exponential distribution with the given rate.
        """
        return -math.log(self.__sample()) / rate

    def choice(self, seq, weights=None):
        """
        Chooses a random element from the sequence seq.
        If weights == None, all elements are equally likely.
        If weights != None, the probability of choosing an element is proportional to its weight.
        """
        if weights is None:
            return seq[int(self.__sample() * len(seq))]
        else:
            total_weight = sum(weights)
            r = self.__sample() * total_weight
            for i in range(len(seq)):
                r -= weights[i]
                if r <= 0:
                    return seq[i]
            return seq[-1]
    

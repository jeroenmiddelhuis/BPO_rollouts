import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

class PolicyLearner:
    
    CACHE_SIZE = 10000

    def __init__(self):
        self.model = None
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

    def build_model(self, observations, actions):
        input_dim = len(observations[0])
        output_dim = len(actions[0])

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='softmax')
        ])

        observations = np.array(observations)
        actions = np.array(actions)

        # Train the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.model.fit(observations, actions, epochs=25)

    def predict(self, observation, action_mask):
        observation = np.array(observation)
        observation = observation.reshape(1, -1)
        action_mask = np.array(action_mask)
        action_mask = action_mask.reshape(1, -1)

        action_probs = self.predict_with_cache(observation)
        action_probs = action_probs * action_mask
        action = [0] * len(action_probs[0])
        action[np.argmax(action_probs)] = 1

        return action
    
    def predict_with_cache(self, observation):
        """
        Tries to get the prediction from the observation from cache.
        If it is not in cache, it calculates the prediction and stores it in cache.
        If the cache is full, it removes the oldest entry.
        """
        observation_tuple = tuple(observation[0])
        if observation_tuple in self.cache:
            self.cache_hits += 1
            return self.cache[observation_tuple]

        action_probs = self.model.predict(observation, verbose=0)[0]
        self.cache_misses += 1
        if len(self.cache) >= self.CACHE_SIZE:
            self.cache_evictions += 1
            self.cache.pop(next(iter(self.cache)))  # Remove the oldest entry, not necessarily the best way to do it, but it is simple.
        self.cache[observation_tuple] = action_probs

        return action_probs

    def policy(self, env):
        """
        Is the policy interpretation of the model, 
        i.e. a function that takes an environment and returns an action for the observation of the environment.
        """
        observation = env.observation()
        action_mask = env.action_mask()
        return self.predict(observation, action_mask)
    
    def save(self, filename):
        self.model.save(filename)

    def load(filename):
        self = PolicyLearner()
        self.model = tf.keras.models.load_model(filename)
        return self

    def print_cache_effeciency(self):
        print("Cache hits: ", self.cache_hits)
        print("Cache misses: ", self.cache_misses)
        print("Cache evictions: ", self.cache_evictions)

    def copy(self):
        learner = PolicyLearner()
        learner.model = tf.keras.models.clone_model(self.model)
        return learner
        
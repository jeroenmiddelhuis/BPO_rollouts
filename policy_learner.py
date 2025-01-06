import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from xgboost import XGBClassifier
import xgboost as xgb
from sb3_contrib import MaskablePPO

class PolicyLearner:
    
    CACHE_SIZE = 100000

    def __init__(self):
        self.model = None
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.model_type = None


    def build_model(self, observations, actions, model_type='neural_network'):
        # Convert observations to numpy array and normalize each observation
        observations = np.array(observations, dtype=float)
        observations = np.apply_along_axis(self.normalize_observation, 1, observations)
        if model_type == 'neural_network':
            self.model_type = 'neural_network'
            input_dim = len(observations[0])
            output_dim = len(actions[0])

            inputs = tf.keras.Input(shape=(input_dim,))
            x = tf.keras.layers.Dense(64, activation='relu')(inputs)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            outputs = tf.keras.layers.Dense(output_dim, activation='softmax')(x)

            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Compile the model
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        elif model_type == 'xgboost':
            self.model_type = 'xgboost'
            self.model = XGBClassifier(
                n_estimators=100, # 
                learning_rate=0.1,#
                max_depth=5,#
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0
            )
            self.model.fit(observations, actions)

    def update_model(self, observations, actions):
        # Reset the cache to evict old entries
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

        # Convert observations to numpy array and normalize each observation
        observations = np.array(observations, dtype=float)
        observations = np.apply_along_axis(self.normalize_observation, 1, observations)
        actions = np.array(actions)
        
        if isinstance(self.model, tf.keras.Model):
            self.model.fit(observations, actions, epochs=25, batch_size=64)
        elif isinstance(self.model, XGBClassifier):
            dtrain = xgb.DMatrix(observations, label=actions)
            params = self.model.get_xgb_params()
            self.model = xgb.train(params, dtrain, xgb_model=self.model.get_booster())
    
    def normalize_observation(self, observation):
        if len(observation) == 5: # single_activity
            observation[-1] = np.minimum(1.0, observation[-1] / 100.0)
        else: # Other 2 activity scenarios
            observation[-2:] = np.minimum(1.0, observation[-2:] / 100.0)
        return observation

    def predict(self, observation, action_mask):
        observation = np.array(observation, dtype=float)
        observation = self.normalize_observation(observation)
        observation = observation.reshape(1, -1)
        action_mask = np.array(action_mask)
        action_mask = action_mask.reshape(1, -1)

        action_probs = self.predict_with_cache(observation, action_mask)
        action_probs = action_probs * action_mask
        action = [0] * len(action_probs[0])
        action[np.argmax(action_probs)] = 1

        return action
    
    def predict_with_cache(self, observation, action_mask=None):
        """
        Tries to get the prediction from the observation from cache.
        If it is not in cache, it calculates the prediction and stores it in cache.
        If the cache is full, it removes the oldest entry.
        """
        observation_tuple = tuple(observation[0])
        if observation_tuple in self.cache:
            self.cache_hits += 1
            return self.cache[observation_tuple]

        if isinstance(self.model, tf.keras.Model):
            action_probs = self.model(observation)[0]
        elif isinstance(self.model, xgb.Booster):
            dmatrix = xgb.DMatrix(observation)
            action_probs = self.model.predict(dmatrix)
            action_probs = np.exp(action_probs) / np.sum(np.exp(action_probs))  # Apply softmax
        elif isinstance(self.model, MaskablePPO):
            action_probs = self.model.predict(observation, action_masks=action_mask)[0]
        else:
            raise ValueError("Unsupported model type")

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
        if filename.endswith('.keras'):
            self.model.save(filename)
        elif filename.endswith('.json'):
            self.model.save_model(filename)

    def load(filename):
        if filename.endswith('.keras'):
            self = PolicyLearner()
            self.model = tf.keras.models.load_model(filename)
        elif filename.endswith('.json'):
            self = PolicyLearner()
            self.model = xgb.Booster()
            self.model.load_model(filename)
        elif filename.endswith('.zip'):
            self = PolicyLearner()
            self.model = MaskablePPO.load(filename)
        return self

    def print_cache_effeciency(self):
        print("Cache hits: ", self.cache_hits)
        print("Cache misses: ", self.cache_misses)
        print("Cache evictions: ", self.cache_evictions)

    def copy(self):
        print(type(self.model))
        if isinstance(self.model, tf.keras.Model):
            model = tf.keras.models.clone_model(self.model)
            model.set_weights(self.model.get_weights())
        elif isinstance(self.model, xgb.Booster):
            model = XGBClassifier()
            model._Booster = self.model.get_booster()
            model._le = self.model._le
            model._feature_names = self.model.get_booster().feature_names
            model._feature_types = self.model.get_booster().feature_types

        learner = PolicyLearner()
        learner.model = model
        return learner

class ValueIterationPolicy():
    def __init__(self, env, max_queue, file):
        self.env = env
        self.max_queue = max_queue
        self.value_function = np.load(file)

        self.feature_ranges = []
        for state_label in self.env.state_space:
            if 'is_processing_' in state_label:
                self.feature_ranges.append(2)
            elif 'assigned_' in state_label:
                self.feature_ranges.append(2)
            elif 'waiting_' in state_label:
                self.feature_ranges.append(self.max_queue + 1)

    def observation_to_index(self, observation):
        """
        Convert a state tuple to a unique index.
        
        :param state: A tuple representing the state (e_1, e_2, ..., e_6).
        :param ranges: A list of the number of possible values for each element in the state.
        :return: A unique index for the state.
        """
        assert len(observation) == len(self.feature_ranges), f"State ({len(observation)}) and ranges ({len(self.feature_ranges)}) length must match"
        
        index = 0
        multiplier = 1
        for i in reversed(range(len(observation))):
            index += observation[i] * multiplier
            if i > 0:  # Prepare multiplier for the next element (if any)
                multiplier *= self.feature_ranges[i]        
        return index

    def policy(self, env):
        observation = env.observation()
        observation = self.clip_observation(observation)
        action_mask = env.action_mask()
        return self.predict(observation, action_mask)

    def predict(self, observation, action_mask):
        action = [0] * len(action_mask)
        action[self.value_function[self.observation_to_index(observation)]] = 1
        return action

    def clip_observation(self, observation):
        # Clip the queue
        if observation[-1] > self.max_queue:

            observation[-1] = self.max_queue
        
        # Clip s8 if it exists (non-single activity case)
        if self.env.config_type != 'single_activity':
            if observation[-2] > self.max_queue:
                observation[-2] = self.max_queue
        
        return observation


if __name__ == "__main__":
    # Test the PolicyLearner class
    import random

    # Create some random observations and actions
    observations = [[random.randint(0, 1) for _ in range(6)] + [random.randint(0, 100) for _ in range(2)] for _ in range(1000)]
    print(observations[:10])
    actions = [[random.choice([0, 1]) for _ in range(3)] for _ in range(1000)]

    # Create a PolicyLearner and build a model
    pl = PolicyLearner()
    pl.build_model(observations, actions, model_type='neural_network')
    pl.update_model(observations, actions)

    # Test the predict function
    observation = [random.randint(0, 1) for _ in range(6)] + [random.randint(0, 100) for _ in range(2)]
    action_mask = [random.choice([0, 1]) for _ in range(3)]
    action = pl.predict(observation, action_mask)
    print("Predicted action: ", action)


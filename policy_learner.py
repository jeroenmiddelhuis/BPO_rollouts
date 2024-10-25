import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from xgboost import XGBClassifier
import xgboost as xgb

class PolicyLearner:
    
    CACHE_SIZE = 10000

    def __init__(self):
        self.model = None
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.model_type = None


    def build_model(self, observations, actions, model_type='neural_network'):
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
        
        observations = np.array(observations)
        actions = np.array(actions)
        
        if isinstance(self.model, tf.keras.Model):
            self.model.fit(observations, actions, epochs=25)
        elif isinstance(self.model, XGBClassifier):
            dtrain = xgb.DMatrix(observations, label=actions)
            params = self.model.get_xgb_params()
            self.model = xgb.train(params, dtrain, xgb_model=self.model.get_booster())


    def predict(self, observation, action_mask):
        observation = np.array(observation)
        if self.model_type == 'neural_network' and len(observation) == 6:  # two resources and two activities
            observation[2:4] = observation[2:4].astype(float) / 2  # Normalize between 0 and 1 (max 2 resources)
            observation[4:6] = np.clip(observation[4:6], 0, 100)  # Clip values to be between 0 and 100
            observation[4:6] = np.round(observation[4:6], 2)  # Round to 2 decimals
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

        if isinstance(self.model, tf.keras.Model):
            action_probs = self.model.predict(observation, verbose=0)[0]            
        elif isinstance(self.model, xgb.Booster):
            dmatrix = xgb.DMatrix(observation)
            action_probs = self.model.predict(dmatrix)
            action_probs = np.exp(action_probs) / np.sum(np.exp(action_probs))  # Apply softmax
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
        self.model.save(filename)

    def load(filename, model_type='neural_network'):
        if model_type == 'neural_network':
            self = PolicyLearner()
            self.model = tf.keras.models.load_model(filename)
        else:
            self = PolicyLearner()
            self.model = xgb.Booster()
            self.model.load_model(filename)
        return self

    def print_cache_effeciency(self):
        print("Cache hits: ", self.cache_hits)
        print("Cache misses: ", self.cache_misses)
        print("Cache evictions: ", self.cache_evictions)

    def copy(self):
        learner = PolicyLearner()
        learner.model = tf.keras.models.clone_model(self.model)
        return learner
        
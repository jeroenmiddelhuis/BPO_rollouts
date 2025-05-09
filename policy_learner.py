import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from sb3_contrib import MaskablePPO
import copy as cp


class PolicyLearner():    
    CACHE_SIZE = 50000

    def __init__(self):
        super(PolicyLearner, self).__init__()
        self.model = None
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.model_type = None

    def build_model(self, observations, actions):
        # Convert observations to numpy array and normalize each observation
        observations = np.array(observations, dtype=float)

        self.model_type = 'neural_network'
        input_dim = len(observations[0])
        output_dim = len(actions[0])

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def update_model(self, observations, actions):
        # Reset the cache
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

        # Convert and normalize data
        observations = np.array(observations, dtype=float)
        observations = np.apply_along_axis(self.normalize_observation, 1, observations)
        actions = np.array(actions)

        # Convert to PyTorch tensors
        observations = torch.tensor(observations, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)

        # Training improvements
        batch_size = 64
        epochs = 100
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        # Train the model with early stopping
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            # Mini-batch training
            for i in range(0, len(observations), batch_size):
                batch_obs = observations[i:i + batch_size]
                batch_acts = actions[i:i + batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_obs)
                loss = self.criterion(outputs, batch_acts)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

            avg_loss = total_loss / (len(observations) / batch_size)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    def normalize_observation(self, observation):
        if len(observation) == 3: # single_activity
            observation[-1] = np.minimum(1.0, observation[-1] / 100.0)
        elif len(observation) <= 8: # 2 activity scenarios
            observation[-2:] = np.minimum(1.0, observation[-2:] / 100.0)
        else: # composite model
            observation[-12:] = np.minimum(1.0, observation[-12:] / 100.0)
        return observation

    def predict(self, observation, action_mask):
        if sum(action_mask) == 1:
            return action_mask
        observation = np.array(observation, dtype=float)
        observation = self.normalize_observation(observation)
        action_probs = self.predict_with_cache(observation, action_mask)   
        print(action_probs)
        print(action_mask)
        print("sum of action probs:", sum(action_probs))
        action_probs = action_probs * action_mask
        print(action_probs)
        print("sum of action probs:", sum(action_probs))
        action = [0] * len(action_probs)
        action[np.argmax(action_probs)] = 1
        return action
    
    def predict_with_cache(self, observation, action_mask=None):
        """
        Tries to get the prediction from the observation from cache.
        If it is not in cache, it calculates the prediction and stores it in cache.
        If the cache is full, it removes the oldest entry.
        """
        observation_tuple = tuple(observation)
        cached_value = self.cache.get(observation_tuple)
        if cached_value is not None:
            self.cache_hits += 1
            return cached_value

        # No value has been found in the cache, calculate the prediction
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        action_probs = self.model(observation)[0]
        action_probs = action_probs.detach().numpy()

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
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
        }, filename)

    def load(filename):
        self = PolicyLearner()
        checkpoint = torch.load(filename, weights_only=True)
        input_dim = checkpoint['model_state_dict']['0.weight'].shape[1]
        output_dim = checkpoint['model_state_dict']['4.weight'].shape[0]
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.criterion = nn.CrossEntropyLoss()
        return self

    def print_cache_effeciency(self):
        print("Cache hits: ", self.cache_hits)
        print("Cache misses: ", self.cache_misses)
        print("Cache evictions: ", self.cache_evictions)

    def copy(self):
        learner = PolicyLearner()
        # Create a deep copy of the entire model
        learner.model = cp.deepcopy(self.model)
        learner.optimizer = optim.Adam(learner.model.parameters(), lr=0.001)
        learner.criterion = nn.CrossEntropyLoss()
        return learner

class ValueIterationPolicy():
    def __init__(self, env, max_queue, file):
        self.env = env
        self.max_queue = max_queue
        self.value_function = np.load(file)

        self.feature_ranges = []
        for state_label in self.env.state_space:
            if 'is_available_' in state_label:
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

class PPOPolicy():
    def __init__(self):
        self.model = None

    def normalize_observation(self, observation):
        if len(observation) == 3: # single_activity
            observation[-1] = np.minimum(1.0, observation[-1] / 100.0)
        elif len(observation) <= 8: # 2 activity scenarios
            observation[-2:] = np.minimum(1.0, observation[-2:] / 100.0)
        else: # composite model
            observation[-12:] = np.minimum(1.0, observation[-12:] / 100.0)
        return observation

    def load(filename):
        self = PPOPolicy()
        self.model = MaskablePPO.load(filename)
        return self

    def policy(self, env):
        """
        Is the policy interpretation of the model, 
        i.e. a function that takes an environment and returns an action for the observation of the environment.
        """
        observation = env.observation()
        observation = self.normalize_observation(np.array(observation, dtype=float))
        action_mask = env.action_mask()
        action, _ = self.model.predict(observation, action_masks=action_mask)
        action = int(action)
        action_array = [0] * len(env.action_space)
        action_array[action] = 1
        return action_array


if __name__ == "__main__":

    
    import random



    ######### Check if the models are different
    pl2 = PolicyLearner.load('./models/pi/mdp/low_utilization/low_utilization.best_policy.pth')
    for i in range(1, 11):
        pl = PolicyLearner.load(f'./models/pi/mdp/low_utilization/low_utilization.v{i}.pth')    

        # Compare if the models are different
        are_models_different = False
        for param1, param2 in zip(pl.model.parameters(), pl2.model.parameters()):
            #print(param1, param2)
            if not torch.equal(param1, param2):
                are_models_different = True
                break

        if are_models_different:
            print("The models are different.", i)
        else:
            print("The models are identical.", i)



    ###### Check if the model is copied correctly
    # pl = PolicyLearner.load('./models/pi/smdp/down_stream/down_stream.v1.pth')
    # pl_copy = pl.copy()

    # # Print parameters before update
    # print("Before update:")
    # for (name1, param1), (name2, param2) in zip(pl.model.named_parameters(), pl_copy.model.named_parameters()):
    #     print(f"{name1}: Equal = {torch.equal(param1, param2)}")

    # # Update original model
    # pl.update_model([[random.randint(0, 1) for _ in range(6)] + 
    #                 [random.randint(0, 100) for _ in range(2)] for _ in range(1000)], 
    #                 [[random.choice([0, 1]) for _ in range(6)] for _ in range(1000)])

    # # Print parameters after update
    # print("\nAfter update:")
    # for (name1, param1), (name2, param2) in zip(pl.model.named_parameters(), pl_copy.model.named_parameters()):
    #     print(f"{name1}: Equal = {torch.equal(param1, param2)}")









    # # Create some random observations and actions
    # observations = [[random.randint(0, 1) for _ in range(6)] + [random.randint(0, 100) for _ in range(2)] for _ in range(1000)]
    # print(observations[:10])
    # actions = [[random.choice([0, 1]) for _ in range(3)] for _ in range(1000)]

    # # Update the model with new observations and actions
    # pl.update_model(observations, actions)

    # # Test the predict function
    # observation = [random.randint(0, 1) for _ in range(6)] + [random.randint(0, 100) for _ in range(2)]
    # action_mask = [random.choice([0, 1]) for _ in range(6)]
    # action = pl.predict(observation, action_mask)
    # print("Predicted action: ", action)


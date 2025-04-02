# BPO Rollouts

A Python implementation of the rollout-based algorithm and reward function based on the paper of [Author not given].
The repository contains the code to train policies for Continuous-Time Markov Decision Process (referred to as SMDP) and Markov Decision Process (MDP) models for Business Process Optimization (BPO).

## Project Structure

- `__main__.py` / `__main__mdp.py` - Main entry point for training and evaluating policies
- `smdp.py` / `mdp.py` - Core SMDP/MDP environment implementations 
- `policy_learner.py` - Policy learning algorithms
- `rollouts.py` - Rollout-based learning utilities
- `heuristic_policies.py` - Implementation of baseline policies (FIFO, Random, Greedy)

## Installation

```bash
# Clone the repository
git clone <repository-url>

# Create and activate new environment
conda create --name bpo_rollouts python=3.12
conda activate bpo_rollouts

# Install dependencies
pip install -r requirements.txt
```

## Usage

Train a policy for the CTMDP:
```bash
python __main__.py <config_type>
```

or train a policy for the MDP:
```bash
python __main__mdp.py <config_type>
```

Where `config_type` can be:
- `single_activity`
- `slow_server` 
- `high_utilization`
- `low_utilization`
- `n_system`
- `down_stream`
- `parallel`
- `composite`

## Key Features

- Continuous-Time Markov Decision Process (referred to as SMDP) and MDP modeling of business processes
- Policy learning through rollout-based methods
- Multiple environment configurations for different process scenarios
- Neural network and value iteration based policy models
- Baseline heuristic policies for comparison
- Evaluation tools and metrics

## Project Structure

- `models/` - Saved policy models
- `results/` - Evaluation results
-

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
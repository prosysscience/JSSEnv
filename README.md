Job-Shop Scheduling Environment [![Build Status](https://travis-ci.com/prosysscience/JSSEnv.svg?token=bPABRGzbzQ2JTRTjgRJn&branch=master)](https://travis-ci.com/prosysscience/JSSEnv)
==============================

An optimized OpenAi gym's environment to simulate the [Job-Shop Scheduling problem](https://developers.google.com/optimization/scheduling/job_shop).

![til](./tests/ta01.gif)

If you've found our work useful for your research, you can cite the [paper](https://arxiv.org/abs/2104.03760) as follows:

```
@misc{tassel2021reinforcement,
      title={A Reinforcement Learning Environment For Job-Shop Scheduling}, 
      author={Pierre Tassel and Martin Gebser and Konstantin Schekotihin},
      year={2021},
      eprint={2104.03760},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Getting Started
------------

This repository is available as a pip package:

```shell
pip install JSSEnv
```

Once installed, the environment will be available in your OpenAi's gym environment and can be used to train a reinforcement learning agent:

```python
import gymnasium as gym
import JSSEnv # an ongoing issue with OpenAi's gym causes it to not import automatically external modules, see: https://github.com/openai/gym/issues/2809
# for older version of gym, you have to use 
# env = gym.make('JSSEnv:jss-v1', env_config={'instance_path': 'INSTANCE_PATH'})
env = gym.make('jss-v1', env_config={'instance_path': 'INSTANCE_PATH'})
```

### Important: Your instance must follow [Taillard's specification](http://jobshop.jjvh.nl/explanation.php#taillard_def). 


How To Use
------------

The observation provided by the environment contains both a boolean array indicating if the action is legal or not and the "real" observation

```python 
self.observation_space = gym.spaces.Dict({
            "action_mask": gym.spaces.Box(0, 1, shape=(self.jobs + 1,)),
            "real_obs": gym.spaces.Box(low=0.0, high=1.0, shape=(self.jobs, 7), dtype=np.float),
        })
```

A random agent would have to sample legal action from this `action_mask` array, otherwise, you might take illegal actions.  
In theory, it is not possible to take the same action over and over again as the job will have one of his operations currently on a machine and might not be free for schedule.  

For research purposes, I've made a random loop using RLLib: https://github.com/prosysscience/RL-Job-Shop-Scheduling/blob/0bbe0c0f2b8a742b75cbe67c5f6a825b8cfdf5eb/JSS/randomLoop/random_loop.py

If you don't want to use RLLib, you can write a simple random loop using `numpy.random.choice` function:

```python
import numpy as np
np.random.choice(len(legal_action), 1, p=(legal_action / legal_action.sum()))[0]
```

Where `legal_action` is the array of legal action (i.e., `action_mask`).  
This line of code will randomly sample one legal action from the `action_mask`.

Baseline Policies
-----------------

This repository includes several baseline policies for Job Shop Scheduling that can be used for comparison and benchmarking:

### Available Baselines

1. **RandomPolicy** - Randomly selects from legal actions
2. **SPTPolicy** - Shortest Processing Time dispatching rule
3. **SimulatedAnnealingPolicy** - Metaheuristic optimization with configurable parameters

### Quick Start with Baselines

```python
from baselines import RandomPolicy, SPTPolicy, SimulatedAnnealingPolicy

# Create environment
env = gym.make('JSSEnv/JssEnv-v1')

# Initialize a baseline policy
policy = RandomPolicy(env)
# or
policy = SPTPolicy(env)
# or
policy = SimulatedAnnealingPolicy(env, initial_temp=100.0, cooling_rate=0.95)

# Use the policy
obs, info = env.reset()
action = policy.select_action(obs)
```

### Evaluation Framework

We provide comprehensive evaluation scripts to test and compare baseline performance:

#### Quick Functionality Test
```bash
python test_baselines.py
```

#### Performance Evaluation
```bash
# Quick evaluation
python run_baseline_evaluation.py --quick --runs 3

# Full evaluation with custom instances
python run_baseline_evaluation.py --instances ta01 ta02 ft06 --runs 10

# All options
python run_baseline_evaluation.py --help
```

#### Results and Analysis

The evaluation generates:
- **CSV files** with detailed results for every run
- **JSON summaries** with aggregated statistics
- **Performance comparisons** across policies and instances
- **Statistical analysis** including makespan, runtime, and success rates

Example output:
```
🎯 Overall Performance Summary:
Policy                    Avg_Makespan  Std_Makespan  Min_Makespan  Max_Makespan  Avg_Runtime  Success_Rate
RandomPolicy                   703.000        45.230       650.000       780.000        0.086         1.000
SPTPolicy                      740.670        32.150       695.000       785.000        0.356         1.000
SA_Standard                    729.000         0.000       729.000       729.000        4.234         1.000
```

For detailed documentation on the evaluation framework, see [BASELINE_EVALUATION_README.md](BASELINE_EVALUATION_README.md).

Project Organization
------------

    ├── README.md                      <- The top-level README for developers using this project.
    ├── BASELINE_EVALUATION_README.md  <- Detailed documentation for baseline evaluation
    ├── requirements.txt               <- Python dependencies for baseline policies
    ├── run_baseline_evaluation.py     <- Comprehensive baseline evaluation script
    ├── test_baselines.py              <- Quick functionality tests for baselines
    ├── debug_episode.py               <- Environment debugging utilities
    │
    ├── baselines/                     <- Baseline policy implementations
    │   ├── __init__.py               <- Makes baselines a Python module
    │   ├── base_policy.py            <- Abstract base class for policies
    │   ├── random_policy.py          <- Random action selection policy
    │   ├── spt_policy.py             <- Shortest Processing Time policy
    │   ├── simulated_annealing_policy.py <- Simulated Annealing policy
    │   ├── utils.py                  <- Utility functions for baselines
    │   └── tests/                    <- Tests for baseline policies
    │       ├── __init__.py
    │       └── mock_env.py           <- Mock environment for testing
    │
    ├── JSSEnv
    │   └── envs                      <- Contains the environment.
    │       └── instances             <- Contains some instances from the literature.
    │
    └── tests                 
        │
        ├── test_state.py             <- Unit tests focus on testing the state produced by
        │                                the environment.
        │
        ├── test_rendering.py         <- Unit tests for the rendering, mainly used as an example
        |                                how to render the environment.
        │
        └── test_solutions.py         <- Unit tests to ensure that our environment is correct checking
                                         known solution in the literature leads to the intended make-
                                         span. We also check if all actions provided by the solution are
                                         legal in our environment.
--------

## Question/Need Support?

Open an issue, we will do our best to answer it.

## License

MIT License

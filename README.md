Job-Shop Scheduling Environment [![Tests](https://github.com/prosysscience/JSSEnv/actions/workflows/python-tests.yml/badge.svg)](https://github.com/prosysscience/JSSEnv/actions/workflows/python-tests.yml)
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
# Install the package
pip install JSSEnv

# For development installation with test dependencies
pip install -e ".[dev]"
```

**Requirements**:
- Python 3.8 or newer
- Dependencies are automatically selected based on your Python version:
  - Python 3.8: numpy >= 1.20.0, < 1.24.0 and pandas >= 1.3.0, < 2.0.0
  - Python 3.9-3.10: numpy >= 1.20.0, < 2.0.0 and pandas >= 1.3.0, < 2.1.0
  - Python 3.11-3.12: numpy >= 1.24.0 and pandas >= 2.0.0

Once installed, the environment will be available in your OpenAi's gym environment and can be used to train a reinforcement learning agent:

```python
import gymnasium as gym
import JSSEnv
env = gym.make('jss-v1', env_config={'instance_path': 'INSTANCE_PATH'})

# Full example with a random agent
obs = env.reset()
done = False
cum_reward = 0

while not done:
    # Get legal actions from action mask
    legal_actions = obs["action_mask"]
    
    # Choose a random legal action
    action = np.random.choice(
        len(legal_actions), 1, p=(legal_actions / legal_actions.sum())
    )[0]
    
    # Take action in environment
    obs, reward, done, truncated, _ = env.step(action)
    cum_reward += reward
```

### Important: Your instance must follow [Taillard's specification](http://jobshop.jjvh.nl/explanation.php#taillard_def). 


How To Use
------------

### Basic Usage

The observation provided by the environment contains both a boolean array indicating if the action is legal or not and the "real" observation:

```python 
self.observation_space = gym.spaces.Dict({
            "action_mask": gym.spaces.Box(0, 1, shape=(self.jobs + 1,)),
            "real_obs": gym.spaces.Box(low=0.0, high=1.0, shape=(self.jobs, 7), dtype=float),
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

### Using Dispatching Rules

The package includes common dispatching rules for job shop scheduling that can be used as baselines or heuristics:

```python
import gymnasium as gym
import JSSEnv
from JSSEnv.dispatching import get_rule, compare_rules

# Create environment
env = gym.make('jss-v1', env_config={'instance_path': 'PATH_TO_INSTANCE'})

# Get a specific dispatching rule
spt_rule = get_rule("SPT")  # Shortest Processing Time rule

# Run an episode with the rule
env.reset()
done = False
total_reward = 0

while not done:
    # The rule selects an action based on the current environment state
    action = spt_rule(env)
    
    # Take the action in the environment
    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward

print(f"Makespan: {env.current_time_step}, Total reward: {total_reward}")

# Visualize the schedule
gantt_chart = env.render()
```

#### Available Dispatching Rules

- **SPT** (Shortest Processing Time): Schedules the job with the shortest processing time for its current operation
- **FIFO** (First-In-First-Out): Schedules the job that has been waiting the longest
- **MWR** (Most Work Remaining): Schedules the job with the most total processing time remaining
- **LWR** (Least Work Remaining): Schedules the job with the least total processing time remaining
- **MOR** (Most Operations Remaining): Schedules the job with the most operations remaining
- **LOR** (Least Operations Remaining): Schedules the job with the fewest operations remaining
- **CR** (Critical Ratio): Schedules based on the ratio of time to due date versus remaining work

#### Comparing Multiple Rules

You can compare multiple dispatching rules on your instance:

```python
# Compare all available rules
results = compare_rules(env, num_episodes=5)

# Print results
for rule_name, metrics in results.items():
    print(f"{rule_name}: Avg Reward = {metrics['avg_reward']:.2f}, Avg Makespan = {metrics['avg_makespan']:.2f}")
```

For a complete example, see the `examples/dispatching_rules_example.py` file.

### Generating Visualization GIFs

To create animated GIFs of your schedules like the one shown at the top of this README, you can use the following code:

```python
import gymnasium as gym
import JSSEnv
import imageio

# Create environment
env = gym.make('jss-v1', env_config={'instance_path': 'PATH_TO_INSTANCE'})
env.reset()

# Initialize list to store images
images = []

# Run your scheduling algorithm
done = False
while not done:
    # Your scheduling logic to choose an action
    action = your_scheduling_algorithm(env)
    
    # Take the action
    obs, reward, done, truncated, _ = env.step(action)
    
    # Render and capture the current state as an image
    temp_image = env.render().to_image()
    images.append(imageio.imread(temp_image))

# Save the images as an animated GIF
imageio.mimsave("schedule.gif", images)
```

This will create a GIF that shows how your schedule evolves step by step. You can adjust the GIF quality and frame rate by using additional parameters in `imageio.mimsave()`.

Note: To use this feature, make sure you have the `imageio` package installed:

```bash
pip install imageio
```

Project Organization
------------

    ├── README.md             <- The top-level README for developers using this project.
    ├── JSSEnv
    │   └── envs              <- Contains the environment.
    │       └── instances     <- Contains some intances from the litterature.
    │
    └── tests                 
        │
        ├── test_state.py     <- Unit tests focus on testing the state produced by
        │                        the environment.
        │
        ├── test_rendering.py <- Unit tests for the rendering, mainly used as an example
        |                        how to render the environment.
        │
        └── test_solutions.py <- Unit tests to ensure that our environment is correct checking
                                 known solution in the litterature leads to the intended make-
                                 span. We also check if all actions provided by the solution are
                                 legal in our environment.
--------

## Question/Need Support?

Open an issue, we will do our best to answer it.

## License

MIT License

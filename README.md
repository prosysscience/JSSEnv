Job-Shop Scheduling Environment [![Build Status](https://travis-ci.com/ingambe/JSSEnv.svg?token=bPABRGzbzQ2JTRTjgRJn&branch=master)](https://travis-ci.com/ingambe/JSSEnv)
==============================

An optimized OpenAi gym's environment to simulate the [Job-Shop Scheduling problem](https://developers.google.com/optimization/scheduling/job_shop).

![til](./tests/ta01.gif)

Getting Started
------------

To start working with this environment you simply need to clone and install it:

```shell
git clone LINK_TO_REPOSITORY
cd JSSEnv
python setup.py
```

Once installed, the environment will be available in your OpenAi's gym environment and can be used to train a reinforcement learning agent:

```python
import gym
env = gym.make('JSSEnv:jss-v1', env_config={'instance_path': 'INSTANCE_PATH'})
```

### Important: Your instance must follow [Taillard's specification](http://jobshop.jjvh.nl/explanation.php#taillard_def). 

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

## License

MIT License

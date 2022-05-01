# -*- coding: utf-8 -*-
"""

This python script outlines the _validate_env logic for Rllib (v.1.9.1) and
can be used for debugging issues in environment configuration.

@author: Philipp Willms
"""
import gym
import numpy as np
import JSSEnv            # import of the installed pip package needed   

# test the environment
instance_path = 'JSSEnv/envs/instances/dmu20'
env = gym.make('JSSEnv-v1', env_config={'instance_path': instance_path})

   
if isinstance(env, gym.Env) :
    # Make sure the gym.Env has the two space attributes properly set.
    assert hasattr(env, "observation_space") and hasattr(
        env, "action_space")
    # Get a dummy observation by resetting the env.
    dummy_obs = env.reset()
    # Convert lists to np.ndarrays.
    if type(dummy_obs) is list and isinstance(env.observation_space,
                                              gym.spaces.Box):
        dummy_obs = np.array(dummy_obs)
        print("Dummy obs after np array conversion: ")
        print(dummy_obs)
    # Ignore float32/float64 diffs.
    if isinstance(env.observation_space, gym.spaces.Box) and \
            env.observation_space.dtype != dummy_obs.dtype:
        dummy_obs = dummy_obs.astype(env.observation_space.dtype)
        print("Dummy obs after ignore float diffs ")
        print(dummy_obs)
    # Check, if observation is ok (part of the observation space). If not,
    # error.
    
    determined_obs_space = env.observation_space
    
    # original code from box.py
    #    def contains(self, x):
    #        if not isinstance(x, np.ndarray):
    #            logger.warn("Casting input x to numpy array.")
    #            x = np.asarray(x, dtype=self.dtype)

    #        return (
    #            np.can_cast(x.dtype, self.dtype)
    #            and x.shape == self.shape
    #            and np.all(x >= self.low)
    #            and np.all(x <= self.high)
    #        )
    
    # action_mask check
    x = determined_obs_space["action_mask"]
    y = dummy_obs["action_mask"]
    print(x)
    print(y)
    print(np.can_cast(x.dtype, y.dtype))
    print(x.shape == y.shape)
    print(np.all(y >= x.low))
    print(np.all(y <= x.high))
    
    # state check
    x = determined_obs_space["real_obs"]
    y = dummy_obs["real_obs"]
    print(x)
    print(y)
    print(np.can_cast(x.dtype, y.dtype))
    print(x.shape == y.shape)
    print(np.all(y >= x.low))
    print(np.all(y <= x.high))
    
# original code from dict.py
#    def contains(self, x):
#        if not isinstance(x, dict) or len(x) != len(self.spaces):
#            return False
#        for k, space in self.spaces.items():
#            if k not in x:
#                return False
#            if not space.contains(x[k]):
#                return False
#        return True
    
    x = determined_obs_space
    y = dummy_obs
    print("Dict check")
    print(isinstance(y, dict))
    print("Length observation space: " + str(len(x.spaces)))
    print("Length dummy observation: " + str(len(y)))
    print(len(y) == len(x.spaces))
    for k, space in x.spaces.items():
            print(k)
            print(space)
            if k not in y:
                #return False
                print("Element not found in dummy observation")
                print(k)
            if not space.contains(y[k]):
                print("Contains check failed")
                print(y[k])
               # return False
    
    # If there is a hard nut to crack with specific observation state, use the following
    x = determined_obs_space["real_obs"]
    mal_state = y[k]
    print(np.can_cast(x.dtype, mal_state.dtype))
    print(x.shape == mal_state.shape)
    print(np.all(mal_state >= x.low))
    print(np.all(mal_state  <= x.high))
    print(isinstance(y[k], np.ndarray))
    print(x.contains(mal_state))
    
    # Copied from rollout_worker.py
    if not env.observation_space.contains(dummy_obs):
        print(
            f"Env's `observation_space` {env.observation_space} does not "
            f"contain returned observation after a reset ({dummy_obs})!")
    else:
        print("All checks passed")
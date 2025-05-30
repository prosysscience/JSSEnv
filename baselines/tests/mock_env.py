import numpy as np
import gymnasium as gym # For space definitions
from gymnasium import spaces # Consistent import for older gymnasium

class MockJSSEnv(gym.Env):
    def __init__(self, num_jobs, num_features=1, instance_name="mock_instance"):
        super().__init__()
        self.num_jobs = num_jobs
        self.num_features = num_features
        self.action_space = spaces.Discrete(num_jobs)
        
        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(0, 1, shape=(num_jobs,), dtype=np.int8),
            "real_obs": spaces.Box(-np.inf, np.inf, shape=(num_jobs, self.num_features), dtype=np.float32)
        })
        
        self.current_action_mask = np.ones(num_jobs, dtype=np.int8)
        self.current_real_obs = np.zeros((num_jobs, num_features), dtype=np.float32)
        self.is_done = False
        self.makespan_to_return = 0 
        self.steps_taken_in_eval = 0
        self.max_steps_in_eval = num_jobs # Default: SA eval takes num_jobs steps

    def set_observation_parts(self, action_mask, real_obs):
        self.current_action_mask = np.array(action_mask, dtype=np.int8)
        self.current_real_obs = np.array(real_obs, dtype=np.float32)
        if self.current_real_obs.ndim == 1:
            # If 1D real_obs is provided, reshape to (num_jobs, 1) if num_features is 1
            if self.num_features == 1 and self.current_real_obs.shape[0] == self.num_jobs:
                self.current_real_obs = self.current_real_obs.reshape((self.num_jobs, 1))
            else:
                # This case might indicate a mismatch, handle as error or specific logic
                pass # Or raise error if shape is not compatible with num_features

    def get_current_observation(self):
        return {"action_mask": self.current_action_mask, "real_obs": self.current_real_obs}

    def set_evaluation_behavior(self, makespan_to_return, max_steps_in_eval=None):
        self.makespan_to_return = makespan_to_return
        if max_steps_in_eval is not None:
            self.max_steps_in_eval = max_steps_in_eval
        else:
            self.max_steps_in_eval = self.num_jobs # Default if not specified

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.is_done = False
        self.steps_taken_in_eval = 0
        # Return a default observation or the preset one
        return self.get_current_observation(), {"info_key": "reset_info"}

    def step(self, action):
        # This step is primarily for SA's _evaluate_schedule_makespan
        self.steps_taken_in_eval += 1
        
        # Update action mask based on action (simple mock: selected job is no longer available)
        # This is a very basic simulation, real JSS state is much more complex.
        if 0 <= action < self.num_jobs:
             self.current_action_mask[action] = 0 
        
        # If all actions in mask are 0 or max_steps_in_eval reached, then done
        if np.sum(self.current_action_mask) == 0 or self.steps_taken_in_eval >= self.max_steps_in_eval:
            self.is_done = True
            
        terminated = self.is_done
        truncated = False # Not handling truncation explicitly in this mock for now
        
        # For SA evaluation, it needs 'makespan' in info when done
        info = {'makespan': self.makespan_to_return} if terminated else {}
        reward = -1 # Dummy reward
        
        return self.get_current_observation(), reward, terminated, truncated, info

    def render(self): pass
    def close(self): pass

    @property
    def spec(self): # Mock spec if policies check for it
        class MockSpec: id = "MockJSSEnv-v0"
        return MockSpec()

import numpy as np
from .base_policy import BaselinePolicy

class RandomPolicy(BaselinePolicy):
    """
    A policy that selects actions randomly from the set of legal actions.
    """
    def __init__(self, env):
        """
        Initializes the RandomPolicy.

        Args:
            env: The OpenAI Gym environment.
        """
        super().__init__(env)

    def select_action(self, observation):
        """
        Selects a random action from the legal actions.

        Args:
            observation: The current observation from the environment.
                         Expected format: {"action_mask": array, "real_obs": array}

        Returns:
            A randomly selected legal action (job index).
        """
        action_mask = observation["action_mask"]
        # Ensure action_mask is a numpy array for np.where
        if not isinstance(action_mask, np.ndarray):
            action_mask = np.array(action_mask)
            
        legal_actions = np.where(action_mask == 1)[0]
        
        if not legal_actions.size:
            # This case should ideally not happen in a well-behaved JSS environment
            # if the episode is not 'done'.
            # If it occurs, it implies no actions are possible, which might mean
            # the episode should have already terminated or there's an issue
            # with the environment's state representation or action mask generation.
            # Raising an error might be appropriate if this state is unexpected.
            # For now, let numpy.random.choice raise an error if legal_actions is empty,
            # as this indicates a potentially problematic state.
            # Consider adding specific error handling or logging if this becomes an issue.
            # For example:
            # raise ValueError("No legal actions available based on the action mask.")
            pass
            
        selected_action = np.random.choice(legal_actions)
        return selected_action

    def reset(self):
        """
        Resets the policy's internal state.
        For RandomPolicy, there is no state to reset.
        """
        pass

import abc

class BaselinePolicy(abc.ABC):
    """
    Abstract base class for all baseline policies.
    """
    def __init__(self, env):
        """
        Initializes the policy with the environment.

        Args:
            env: The OpenAI Gym environment.
        """
        self.env = env
    
    @abc.abstractmethod
    def select_action(self, observation):
        """
        Selects an action given the current observation.

        This method must be implemented by subclasses.

        Args:
            observation: The current observation from the environment.
                         Expected format: {"action_mask": array, "real_obs": array}

        Returns:
            The selected action (job index).
        """
        pass
    
    def reset(self):
        """
        Resets the policy's internal state if needed.

        This method can be overridden by subclasses if they maintain state.
        """
        pass

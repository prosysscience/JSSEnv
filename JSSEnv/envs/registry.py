from typing import Any, Dict, Optional
from gymnasium.envs.registration import EnvRegistry, EnvSpec

# Initialize the environment registry
registry = EnvRegistry()


def register(id: str, **kwargs: Any) -> None:
    """
    Register an environment with the JSSEnv registry.
    
    Args:
        id: Unique identifier for the environment
        **kwargs: Additional arguments to pass to the environment constructor
    
    Returns:
        None
    """
    return registry.register(id, **kwargs)


def make(id: str, **kwargs: Any) -> Any:
    """
    Create an instance of a registered environment.
    
    Args:
        id: Identifier of a registered environment
        **kwargs: Additional arguments to pass to the environment constructor
    
    Returns:
        An instance of the environment
    """
    return registry.make(id, **kwargs)


def spec(id: str) -> EnvSpec:
    """
    Returns the specification for the specified environment.
    
    Args:
        id: Identifier of a registered environment
    
    Returns:
        The specification for the specified environment
    """
    return registry.spec(id)

from typing import Any, Dict, Type, Union
import numpy as np


def assign_env_config(self: Any, kwargs: Dict[str, Any]) -> None:
    """
    Assign environment configuration parameters to an object.
    
    This utility function sets attributes on an object from a dictionary of parameters.
    It handles type conversion based on existing attribute types and supports nested
    configuration via env_config.
    
    Args:
        self: The object to configure
        kwargs: Dictionary of configuration parameters
    """
    for key, value in kwargs.items():
        setattr(self, key, value)
    
    if hasattr(self, "env_config"):
        for key, value in self.env_config.items():
            # Check types based on default settings
            if hasattr(self, key):
                if isinstance(getattr(self, key), np.ndarray):
                    setattr(self, key, value)
                else:
                    setattr(self, key, type(getattr(self, key))(value))
            else:
                setattr(self, key, value)


def create_env(config: Union[Dict[str, Any], str], *args: Any, **kwargs: Any) -> Type:
    """
    Create an environment instance based on configuration.
    
    This function helps integrate with Ray by providing a way to create environment
    instances from configuration dictionaries or strings.
    
    Args:
        config: Either a dictionary containing an 'env' key or a string with the environment name
        *args: Additional positional arguments to pass to the environment
        **kwargs: Additional keyword arguments to pass to the environment
        
    Returns:
        Environment class that can be instantiated
        
    Raises:
        NotImplementedError: If the environment name is not recognized
    """
    if isinstance(config, dict):
        env_name = config["env"]
    else:
        env_name = config
        
    if env_name == "jss-v1":
        from JSSEnv.envs.jss_env import JssEnv as env
    else:
        raise NotImplementedError(f"Environment {env_name} not recognized.")
        
    return env

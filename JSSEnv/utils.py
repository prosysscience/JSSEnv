import numpy as np


def assign_env_config(self, kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)
    if hasattr(self, "env_config"):
        for key, value in self.env_config.items():
            # Check types based on default settings
            if hasattr(self, key):
                if type(getattr(self, key)) == np.ndarray:
                    setattr(self, key, value)
                else:
                    setattr(self, key, type(getattr(self, key))(value))
            else:
                setattr(self, key, value)


# Get Ray to work with gym registry
def create_env(config, *args, **kwargs):
    if type(config) == dict:
        env_name = config["env"]
    else:
        env_name = config
    if env_name == "jss-v1":
        from JSSEnv.envs.jss_env import JssEnv as env
    else:
        raise NotImplementedError("Environment {} not recognized.".format(env_name))
    return env

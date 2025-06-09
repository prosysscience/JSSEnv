__version__ = "1.1.0"

from gymnasium.envs.registration import register

register(
    id="JSSEnv/JssEnv-v1",
    entry_point="JSSEnv.envs:JssEnv",
    max_episode_steps=300
)
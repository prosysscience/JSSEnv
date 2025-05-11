# from gym.envs.registration import register
from JSSEnv.envs.jss_env import JssEnv
from gymnasium.envs.registration import register

register(
    id="JSSEnv/JssEnv-v1",
    entry_point="JSSEnv.envs:JssEnv",
    max_episode_steps=300
)
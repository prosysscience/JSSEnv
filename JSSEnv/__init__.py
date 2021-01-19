from gym.envs.registration import register

register(
    id='jss-v0',
    entry_point='JSSEnv.envs:JssEnv',
)
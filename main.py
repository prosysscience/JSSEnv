import gymnasium as gym
import JSSEnv # an ongoing issue with OpenAi's gym causes it to not import automatically external modules, see: https://github.com/openai/gym/issues/2809
# for older version of gym, you have to use 
# env = gym.make('JSSEnv:jss-v1', env_config={'instance_path': 'INSTANCE_PATH'})

import numpy
import traceback
from colorama import Fore
import imageio.v2 as imageio

from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# gym.pprint_registry()
# env = gym.make("LunarLander-v3", render_mode="human")
env = gym.make(
    'JSSEnv/JssEnv-v1',
    env_config={
        "instance_path": f"JSSEnv/envs/instances/ta02"
    }
).unwrapped

check_env(env)

env.action_space.seed(42)
observation, info = env.reset(seed=42)

print(f"env.observation_space.shape = {env.observation_space.shape}")
print(f"env.action_space.shape = {env.action_space.shape}")

# Solution sequence for rendering (example sequence)
solution_sequence = [
    [7, 11, 9, 10, 8, 3, 12, 2, 14, 5, 1, 6, 4, 0, 13],
    [2, 8, 7, 14, 6, 13, 9, 11, 4, 5, 12, 3, 10, 1, 0],
    [11, 9, 3, 0, 4, 12, 8, 7, 5, 2, 6, 14, 13, 10, 1],
    [6, 5, 0, 9, 12, 7, 11, 10, 14, 1, 13, 2, 3, 4, 8],
    [10, 13, 0, 4, 1, 5, 14, 3, 7, 6, 12, 8, 2, 9, 11],
    [5, 7, 3, 12, 13, 10, 1, 11, 8, 4, 2, 6, 0, 9, 14],
    [9, 0, 4, 8, 3, 11, 13, 14, 6, 12, 10, 2, 1, 7, 5],
    [4, 6, 7, 10, 0, 11, 1, 9, 3, 5, 13, 14, 8, 2, 12],
    [13, 4, 6, 2, 9, 14, 12, 11, 7, 10, 0, 1, 3, 8, 5],
    [9, 3, 2, 4, 13, 11, 12, 1, 0, 7, 8, 5, 14, 10, 6],
    [8, 14, 4, 3, 11, 12, 9, 0, 10, 13, 5, 1, 6, 2, 7],
    [7, 9, 8, 5, 6, 0, 2, 3, 1, 13, 14, 12, 4, 10, 11],
    [6, 0, 7, 11, 5, 14, 10, 2, 4, 13, 8, 9, 3, 12, 1],
    [13, 10, 7, 9, 5, 3, 11, 1, 12, 14, 2, 4, 0, 6, 8],
    [13, 11, 6, 8, 7, 4, 1, 5, 3, 10, 0, 14, 9, 2, 12],
]

# Initialize variables
done = False
job_nb = len(solution_sequence[0])
machine_nb = len(solution_sequence)
index_machine = [0 for _ in range(machine_nb)]
step_nb = 0
images = []

verbose = True
print_every_n_iter = 10

try:
    for i in range(1000):  # Loop for a fixed number of steps
        # Take a random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        # render image if available
        fig = env.render()
        if fig is None:
            # nothing to draw (e.g. right after reset), skip this frame
            print("Nothing to draw.")
            continue
        temp_image = fig.to_image()
        images.append(imageio.imread(temp_image))
        
        # Print verbose information every `print_every_n_iter` steps
        if verbose and i % print_every_n_iter == 0:
            print(f"Iteration {i}:")
            print(f"\nReward: {reward}")
            print(f"\nLast time step: {env.last_time_step}")
            print(f"Current time step: {env.current_time_step}")
            print(f"Next time step: {env.next_time_step}")
            print(f"\nNumber of legal machines: {env.nb_machine_legal}")
            print(f"Number of legal actions: {env.nb_legal_actions}")
            print(f"\nAction: {action}")
            print(f"Jobs: {env.jobs}")
            print("------")
            
        if (action >= env.jobs):
            print("------")
            print(f"Action >= jobs: bottleneck at iteration {i}")
            print("------")
            
        if env.nb_legal_actions == 0:
            print(f"Env should've stopped at iteration {i}")
            print(f"\n\tFinal Iteration {i}:")
            print(f"\t\tReward: {reward}")
            print(f"\t\tLast time step: {env.last_time_step}")
            print(f"\t\tCurrent time step: {env.current_time_step}")
            print(f"\t\tNext time step: {env.next_time_step}")
            print(f"\t\tNumber of legal machines: {env.nb_machine_legal}")
            print(f"\t\tNumber of legal actions: {env.nb_legal_actions}\n")
            print(f"\n\tAction: {action}")
            print(f"\tJobs: {env.jobs}")
            print("------")

        # Check if the episode is done
        if terminated or truncated:
            if verbose:
                print("------")
                print(f"RESETTING ENVIRONMENT at iteration {i}!")
                print("------")
            observation, info = env.reset(seed=42)
except Exception as e:
    print(Fore.RED + f"An error occured: {e}" + Fore.RESET)
    traceback.print_exc()
    print(Fore.RED + f"======" + Fore.RESET)
finally:
    # Save the rendered frames as a GIF
    try:
        imageio.mimsave("ta01.gif", images)
        print("Rendering completed. GIF saved as 'ta01.gif'.")
        env.close()
    except Exception as e:
        print(Fore.RED + f"An error occured: {e}" + Fore.RESET)
        traceback.print_exc()
        print(Fore.RED + f"======" + Fore.RESET)
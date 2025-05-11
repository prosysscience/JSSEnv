import gymnasium as gym
import JSSEnv # an ongoing issue with OpenAi's gym causes it to not import automatically external modules, see: https://github.com/openai/gym/issues/2809
# for older version of gym, you have to use 
# env = gym.make('JSSEnv:jss-v1', env_config={'instance_path': 'INSTANCE_PATH'})

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from colorama import Fore
import imageio.v2 as imageio

from gymnasium.utils.env_checker import check_env
from gymnasium.utils import RecordConstructorArgs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class IdleTrackingWrapper(RecordConstructorArgs, gym.Wrapper):
    """
    A Gymnasium wrapper that tracks idle time per machine by monkey-patching
    the environment's increase_time_step method. Inherits from RecordConstructorArgs
    so it can be recreated by gymnasium's env checker.
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        RecordConstructorArgs.__init__(self)

        # track number of machines and their idle times
        self.n_machines = env.machines
        self.machine_idle = np.zeros(self.n_machines, dtype=float)
        
        # keep a reference to original increase_time_step
        self._orig_increase = env.increase_time_step

        def tracked_increase():
            # if no pending completions, no idle time
            if not env.next_time_step:
                return 0
            
            # compute time delta until next event
            next_time = env.next_time_step[0]
            diff = next_time - env.current_time_step

            # idle time per machine = max(0, diff - busy_time)
            idle_per_machine = np.maximum(
                0, diff - env.time_until_available_machine
            )
            
            # accumulate idle times
            self.machine_idle += idle_per_machine
            
            # call original time advancement
            return self._orig_increase()

        # monkey-patch environment
        env.increase_time_step = tracked_increase

    def __getattr__(self, name):
        # lookup attribute in the wrapped environment
        return getattr(self.env, name)
        
raw = gym.make(
    'JSSEnv/JssEnv-v1',
    env_config={'instance_path': 'JSSEnv/envs/instances/ta02'}
).unwrapped
check_env(raw)

env = IdleTrackingWrapper(raw)

env.action_space.seed(42)
observation, info = env.reset(seed=42)

print(f"env.observation_space.shape = {env.observation_space.shape}")
print(f"env.action_space.shape = {env.action_space.shape}")

# Initialize variables
images = []
records = []

verbose = True
print_every_n_iter = 50

terminated = False
truncated = False

num_iters = 1000

try:
    for step in range(num_iters):  # Loop for a fixed number of steps
        # Take a random action
        if verbose and step % print_every_n_iter == 0:
            print(f"Iteration {step}")
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        # render image if available
        env.env._last_iteration = step
        fig = env.render()
        
        if fig is None:
            continue # nothing to draw, skip
        
        temp_image = fig.to_image()
        images.append(imageio.imread(temp_image))

        records.append({
            'step':         step,
            'reward':       reward,
            'time_step':    env.current_time_step,
            'legal_machines': env.nb_machine_legal,
            'legal_actions':  env.nb_legal_actions
        })

        # Check if the episode is done
        if terminated or truncated:
            observation, info = env.reset(seed=42)
except Exception as e:
    print(Fore.RED + f"An error occured: {e}" + Fore.RESET)
    traceback.print_exc()
    print(Fore.RED + f"======" + Fore.RESET)
finally:
    # Save the rendered frames as a GIF
    try:
        print("Attempting to render...")
        imageio.mimsave("output/ta01.gif", images)
        print("Rendering completed. GIF saved as 'ta01.gif'.")
        
        if records:
            # Create output folder
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            df = pd.DataFrame(records)
            df.to_csv(os.path.join(output_dir, "jss_run.csv"), index=False)
            print("→ Metrics saved to jss_run.csv")

            # Plotting and saving each figure
            plt.figure()
            plt.plot(df['step'], df['reward'])
            plt.xlabel('Step'); plt.ylabel('Reward'); plt.title('Reward per Step')
            plt.savefig(os.path.join(output_dir, "reward_per_step.png"))
            # plt.show()
            plt.close()

            plt.figure()
            plt.plot(df['step'], df['reward'].cumsum())
            plt.xlabel('Step'); plt.ylabel('Cumulative Reward'); plt.title('Cumulative Reward')
            plt.savefig(os.path.join(output_dir, "cumulative_reward.png"))
            # plt.show()
            plt.close()

            plt.figure()
            plt.plot(df['step'], df['legal_machines'])
            plt.xlabel('Step'); plt.ylabel('Legal Machines'); plt.title('Machine Availability')
            plt.savefig(os.path.join(output_dir, "legal_machines_over_time.png"))
            # plt.show()
            plt.close()

            plt.figure()
            plt.plot(df['step'], df['legal_actions'])
            plt.xlabel('Step'); plt.ylabel('Legal Actions'); plt.title('Action-Mask Size')
            plt.savefig(os.path.join(output_dir, "legal_actions_over_time.png"))
            # plt.show()
            plt.close()

            # TO-DO: Idle time per machine
            plt.figure()
            plt.bar(range(env.n_machines), env.machine_idle)
            plt.xlabel('Machine ID'); plt.ylabel('Total Idle Time'); plt.title('Idle Time per Machine')
            plt.savefig(os.path.join(output_dir, "idle_time_per_machine.png"))
            # plt.show()
            plt.close()

            print(f"All figures saved to '{output_dir}/'")
            
        env.close()
    except Exception as e:
        print(Fore.RED + f"An error occured: {e}" + Fore.RESET)
        traceback.print_exc()
        print(Fore.RED + f"======" + Fore.RESET)
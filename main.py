import argparse
import gymnasium as gym
import JSSEnv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from colorama import Fore
import imageio.v2 as imageio
from gymnasium.utils.env_checker import check_env
from gymnasium.utils import RecordConstructorArgs
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3 import PPO, DQN
from baselines import RandomPolicy, SPTPolicy, SimulatedAnnealingPolicy, LWKRPolicy, CriticalPathPolicy

# Wrapper to track idle time per machine
class IdleTrackingWrapper(RecordConstructorArgs, gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        RecordConstructorArgs.__init__(self)

        self.n_machines = env.machines
        self.machine_idle = np.zeros(self.n_machines, dtype=float)
        self._orig_increase = env.increase_time_step

        def tracked_increase():
            if not env.next_time_step:
                return 0
            next_time = env.next_time_step[0]
            diff = next_time - env.current_time_step
            idle_per_machine = np.maximum(0, diff - env.time_until_available_machine)
            self.machine_idle += idle_per_machine
            return self._orig_increase()

        env.increase_time_step = tracked_increase

    def __getattr__(self, name):
        return getattr(self.env, name)


# Function to create a policy instance
def create_policy(policy_name, env):
    if policy_name == "RandomPolicy":
        return RandomPolicy(env)
    elif policy_name == "SPTPolicy":
        return SPTPolicy(env)
    elif policy_name == "SimulatedAnnealingPolicy":
        return SimulatedAnnealingPolicy(env, initial_temp=100.0, cooling_rate=0.95, max_iter_per_restart=100, num_restarts=5, seed=42)
    elif policy_name == "LWKRPolicy":
        return LWKRPolicy(env)
    elif policy_name == "CriticalPathPolicy":
        return CriticalPathPolicy(env)
    
    ### TO-DO: Fix PPO and DQN models, which don't render anything
    ### WARNING: 
    ### PPO and DQN were trained on instance ta80, make sure you change the instance we are running on by using the --instance arg.
    # elif policy_name == "PPO":
    #     if not model_path:
    #         raise ValueError("Model path must be provided for PPO.")
    #     print(f"Loading PPO model from: {model_path}")
    #     return PPO.load(model_path, env=env)
    # elif policy_name == "DQN":
    #     if not model_path:
    #         raise ValueError("Model path must be provided for DQN.")
    #     print(f"Loading DQN model from: {model_path}")
    #     return DQN.load(model_path, env=env)
    # else:
    #     raise ValueError(f"Unknown policy: {policy_name}")


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run Job Shop Scheduling simulation with a selected policy.")
parser.add_argument("--policy", type=str, default="RandomPolicy", help="Name of the policy to use (e.g., RandomPolicy, SPTPolicy, SimulatedAnnealingPolicy, LWKRPolicy, CriticalPathPolicy)")
parser.add_argument("--instance_dir", type=str, default="JSSEnv/envs/instances", help="Directory containing instance files")
parser.add_argument("--instance", type=str, default="ta02", help="Name of the instance file (e.g., ta02)")
parser.add_argument("--num_iters", type=int, default=1000, help="Number of iterations to simulate")
parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
parser.add_argument("--print_freq", type=int, default=50, help="Frequency of printing progress (in iterations)")
parser.add_argument("--model_path", type=str, default=None, help="Path to the trained PPO or DQN model (required for PPO and DQN policies)")
args = parser.parse_args()

# Extract arguments
policy_name = args.policy
instance_dir = args.instance_dir
instance_name = args.instance
num_iters = args.num_iters
output_dir = args.output_dir
print_freq = args.print_freq
model_path = args.model_path

# Fallback to defaults if arguments are missing
instance_path = os.path.join(instance_dir, f"{instance_name}")

print(f"Using policy: {policy_name}")
print(f"Instance: {instance_path}")
print(f"Number of iterations: {num_iters}")
print(f"Output directory: {output_dir}")
print(f"Print frequency: {print_freq}")

# Create output directory if it doesn't exist
policy_output_dir = os.path.join(output_dir, policy_name)
os.makedirs(policy_output_dir, exist_ok=True)

# Initialize the environment
env_config = {'instance_path': instance_path}
env = gym.make('JSSEnv/JssEnv-v1', env_config=env_config).unwrapped
env = IdleTrackingWrapper(env)  # Apply the wrapper

# Create policy
policy = create_policy(policy_name, env)

# Initialize variables
images = []
records = []
resets = []
reset_counter = 0

verbose = True
terminated = False
truncated = False

# Flag to track if rendering is working
rendering_enabled = True

try:
    # Initialize the environment and get the initial observation
    observation, info = env.reset(seed=42)  # Reset the environment
    
    for step in range(num_iters):  # Loop for a fixed number of steps
        # Select an action using the policy
        if verbose and step % print_freq == 0:
            print(f"Iteration {step}")
            
        if policy_name in ["PPO", "DQN"]:
            # Use the RL model to predict the action
            action, _ = policy.predict(observation, deterministic=True)
        else:
            action = policy.select_action(env.reset(seed=42)[0] if step == 0 else observation)

        # Take the action in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Try to render image if available and rendering is enabled
        if rendering_enabled:
            try:
                fig = env.render()
                if fig is not None:
                    temp_image = fig.to_image()
                    images.append(imageio.imread(temp_image))
            except Exception as render_error:
                print(f"Rendering disabled due to error: {render_error}")
                rendering_enabled = False
                images = []  # Clear any partial images

        # Check if the episode is done
        if terminated or truncated:
            resets.append(step)
            reset_counter += 1
            observation, info = env.reset(seed=42)
            
        records.append({
            'step':           step,
            'reward':         reward,
            'time_step':      env.current_time_step,
            'legal_machines': env.nb_machine_legal,
            'legal_actions':  env.nb_legal_actions,
            'resets':         reset_counter
        })
except Exception as e:
    print(Fore.RED + f"An error occurred: {e}" + Fore.RESET)
    traceback.print_exc()
    print(Fore.RED + f"======" + Fore.RESET)
finally:
    # Save the rendered frames as a GIF (only if we have images)
    try:
        print("Attempting to save results...")
        
        if images and rendering_enabled:
            gif_path = os.path.join(policy_output_dir, f"{policy_name}_simulation.gif")
            imageio.mimsave(gif_path, images)
            print(f"Rendering completed. GIF saved as '{gif_path}'.")
        else:
            print("Skipping GIF creation due to rendering issues.")
        
        if records:
            # Save metrics to CSV
            csv_path = os.path.join(policy_output_dir, f"{policy_name}_metrics.csv")
            df = pd.DataFrame(records)
            df.to_csv(csv_path, index=False)
            print(f"→ Metrics saved to {csv_path}")

            # Plotting and saving each figure
            plt.figure()
            plt.plot(df['step'], df['reward'])
            plt.xlabel('Step'); plt.ylabel('Reward'); plt.title('Reward per Step')
            plt.savefig(os.path.join(policy_output_dir, f"{policy_name}_reward_per_step.png"))
            plt.close()

            plt.figure()
            plt.plot(df['step'], df['reward'].cumsum())
            plt.xlabel('Step'); plt.ylabel('Cumulative Reward'); plt.title('Cumulative Reward')
            plt.savefig(os.path.join(policy_output_dir, f"{policy_name}_cumulative_reward.png"))
            plt.close()

            plt.figure()
            plt.plot(df['step'], df['legal_machines'])
            plt.xlabel('Step'); plt.ylabel('Legal Machines'); plt.title('Machine Availability')
            plt.savefig(os.path.join(policy_output_dir, f"{policy_name}_legal_machines_over_time.png"))
            plt.close()

            plt.figure()
            plt.plot(df['step'], df['legal_actions'])
            plt.xlabel('Step'); plt.ylabel('Legal Actions'); plt.title('Action-Mask Size')
            plt.savefig(os.path.join(policy_output_dir, f"{policy_name}_legal_actions_over_time.png"))
            plt.close()

            plt.figure()
            plt.bar(range(env.n_machines), env.machine_idle)
            plt.xlabel('Machine ID'); plt.ylabel('Total Idle Time'); plt.title('Idle Time per Machine')
            plt.savefig(os.path.join(policy_output_dir, f"{policy_name}_idle_time_per_machine.png"))
            plt.close()

            plt.figure()
            plt.plot(df['step'], df['resets'])
            plt.xlabel('Step'); plt.ylabel('Resets'); plt.title('Resets Over Time')
            plt.savefig(os.path.join(policy_output_dir, f"{policy_name}_resets_over_time.png"))
            plt.close()

            print(f"All figures saved to '{policy_output_dir}/'")
            
        print(f"Resets: {resets}")
        env.close()
        print("Execution completed successfully!")
    except Exception as e:
        print(Fore.RED + f"An error occurred: {e}" + Fore.RESET)
        traceback.print_exc()
        print(Fore.RED + f"======" + Fore.RESET)
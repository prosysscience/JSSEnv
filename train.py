import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from colorama import Fore
import imageio.v2 as imageio

from gymnasium.utils.env_checker import check_env
from gymnasium.utils import RecordConstructorArgs
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env

import JSSEnv  # Ensure JSSEnv module is discoverable

# Wrapper to track idle times
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
            diff = env.next_time_step[0] - env.current_time_step
            idle_per_machine = np.maximum(0, diff - env.time_until_available_machine)
            self.machine_idle += idle_per_machine
            return self._orig_increase()

        env.increase_time_step = tracked_increase

    def __getattr__(self, name):
        return getattr(self.env, name)


def evaluate_model(model, env, n_steps=1000, seed=42):
    obs, info = env.reset(seed=seed)
    cum_reward = 0.0
    for step in range(n_steps):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        cum_reward += reward
        if terminated or truncated:
            break
    return cum_reward, env.machine_idle


def run_random_play(env, num_iters=1000):
    obs, info = env.reset(seed=42)
    cum_reward = 0.0
    records = []
    images = []

    for step in range(num_iters):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        cum_reward += reward

        # render frames
        env.env._last_iteration = step
        fig = env.render()
        if fig:
            temp = fig.to_image()
            images.append(imageio.imread(temp))

        records.append({
            'step': step,
            'reward': reward,
            'time_step': env.current_time_step,
            'legal_machines': env.nb_machine_legal,
            'legal_actions': env.nb_legal_actions,
            'resets': env.reset_count if hasattr(env, 'reset_count') else 0
        })

        if terminated or truncated:
            obs, info = env.reset(seed=42)

    return cum_reward, records, images


if __name__ == '__main__':
    # -- Environment setup --
    raw_env = gym.make(
        'JSSEnv/JssEnv-v1',
        env_config={'instance_path': 'JSSEnv/envs/instances/ta02'}
    ).unwrapped
    check_env(raw_env)
    env = IdleTrackingWrapper(raw_env)

    # -- Training with PPO --
    train_env = make_vec_env(lambda: gym.make(
        'JSSEnv/JssEnv-v1',
        env_config={'instance_path': 'JSSEnv/envs/instances/ta02'}
    ), n_envs=4)
    ppo_model = PPO("MultiInputPolicy", train_env, verbose=1)
    ppo_model.learn(total_timesteps=100000)
    ppo_model.save("ppo_jss_model")
    print("PPO training complete and model saved as 'ppo_jss_model.zip'")

    # -- Training with DQN --
    dqn_model = DQN("MultiInputPolicy", train_env, verbose=1, exploration_fraction=0.1)
    dqn_model.learn(total_timesteps=100000)
    dqn_model.save("dqn_jss_model")
    print("DQN training complete and model saved as 'dqn_jss_model.zip'")

    # -- Evaluation --
    eval_env = IdleTrackingWrapper(raw_env)
    ppo_reward, ppo_idle = evaluate_model(ppo_model, eval_env)
    print(f"PPO evaluation cumulative reward: {ppo_reward}")

    # Reset raw_env for fresh evaluation
    raw_env = gym.make(
        'JSSEnv/JssEnv-v1',
        env_config={'instance_path': 'JSSEnv/envs/instances/ta02'}
    ).unwrapped
    eval_env = IdleTrackingWrapper(raw_env)
    dqn_reward, dqn_idle = evaluate_model(dqn_model, eval_env)
    print(f"DQN evaluation cumulative reward: {dqn_reward}")

    # -- Optional: random play for baseline comparison --
    random_reward, records, images = run_random_play(env)
    print(f"Random agent cumulative reward: {random_reward}")

    # Save metrics and plots
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "jss_run.csv"), index=False)

    # Reward plot
    plt.figure()
    plt.plot(df['step'], df['reward'])
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward per Step')
    plt.savefig(os.path.join(output_dir, "reward_per_step.png"))
    plt.close()

    # Cumulative reward plot
    plt.figure()
    plt.plot(df['step'], df['reward'].cumsum())
    plt.xlabel('Step')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward over Time')
    plt.savefig(os.path.join(output_dir, "cumulative_reward.png"))
    plt.close()

    # Idle time visualization
    plt.figure()
    plt.bar(range(eval_env.n_machines), ppo_idle)
    plt.xlabel('Machine ID')
    plt.ylabel('Total Idle Time (PPO)')
    plt.title('Idle Time per Machine (PPO Evaluation)')
    plt.savefig(os.path.join(output_dir, "ppo_idle_time_per_machine.png"))
    plt.close()

    plt.figure()
    plt.bar(range(eval_env.n_machines), dqn_idle)
    plt.xlabel('Machine ID')
    plt.ylabel('Total Idle Time (DQN)')
    plt.title('Idle Time per Machine (DQN Evaluation)')
    plt.savefig(os.path.join(output_dir, "dqn_idle_time_per_machine.png"))
    plt.close()

    # Save GIF from random play
    if images:
        imageio.mimsave(os.path.join(output_dir, "random_play.gif"), images)

    print("All artifacts saved to 'output/' directory.")

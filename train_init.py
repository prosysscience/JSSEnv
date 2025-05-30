import gymnasium as gym
import JSSEnv
from stable_baselines3 import PPO, DQN
import os

# Create output directory for models
output_dir = "output/models"
os.makedirs(output_dir, exist_ok=True)

# Initialize the environment
env_config = {"instance_path": "JSSEnv/envs/instances/ta80"}  # Update path if needed
env = gym.make("JSSEnv/JssEnv-v1", env_config=env_config)

# Train PPO model
print("Training PPO model...")
ppo_model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=1e-4, n_steps=2048, batch_size=64, gae_lambda=0.95, gamma=0.99)
ppo_model.learn(total_timesteps=1000000)
ppo_model.save(os.path.join(output_dir, "ppo_model"))
print("PPO model trained and saved.")

# Train DQN model
print("Training DQN model...")
dqn_model = DQN("MultiInputPolicy", env, verbose=1, learning_rate=1e-4, buffer_size=50000, batch_size=32, train_freq=4, target_update_interval=500)
dqn_model.learn(total_timesteps=1000000)
dqn_model.save(os.path.join(output_dir, "dqn_model"))
print("DQN model trained and saved.")

env.close()
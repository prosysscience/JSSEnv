import os
import gymnasium as gym
import JSSEnv
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# ---- Configuration ----
SEED = 42
NUM_ENVS = 4
INSTANCE_PATH = "JSSEnv/envs/instances/ta80"
LOG_DIR = "output/dqn/logs"
MODEL_DIR = "output/dqn/models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---- Environment Factory ----
def make_env():
    env = gym.make("JSSEnv/JssEnv-v1", env_config={"instance_path": INSTANCE_PATH})
    return Monitor(env, LOG_DIR)  # logs per-episode stats

# ---- Vectorized and Normalized Env ----
train_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
train_env.seed(SEED)
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)  # online normalization

# ---- Eval Callback ----
eval_env = DummyVecEnv([make_env])
eval_env.seed(SEED)
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=5_000,
    deterministic=True,
)

# ---- DQN Agent ----
dqn = DQN(
    policy="MultiInputPolicy",
    env=train_env,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=64,
    train_freq=4,
    target_update_interval=1_000,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    tensorboard_log=LOG_DIR,
    verbose=1,
    seed=SEED,
    device="auto",
)

# ---- Training ----
dqn.learn(total_timesteps=500_000, callback=eval_callback)
dqn.save(os.path.join(MODEL_DIR, "dqn_model"))
train_env.save(os.path.join(MODEL_DIR, "vecnormalize.pkl"))
print("DQN training complete!")


import os
import argparse
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import JSSEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.set_num_threads(os.cpu_count())


class QNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        jobs, feat = obs_shape
        input_size = jobs * feat

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, real_obs: torch.Tensor):
        B = real_obs.shape[0]
        x = real_obs.view(B, -1)  # flatten
        return self.net(x)


def train_dqn(
    train_instances,
    total_steps,
    batch_size,
    buffer_capacity,
    lr,
    gamma,
    eps_start,
    eps_final,
    eps_decay_steps,
    target_update_freq,
    train_interval,
    eval_interval,
    device,
    log_dir
):
    writer = SummaryWriter(log_dir) if log_dir is not None else None

    # Initialize replay buffer
    replay_buffer = deque(maxlen=buffer_capacity)

    # Create a dummy env on the first instance to get obs_shape & action mask size
    first_path = os.path.join("JSSEnv", "envs", "instances", train_instances[0])
    env0 = gym.make("JSSEnv/JssEnv-v1", env_config={"instance_path": first_path})
    obs0, _ = env0.reset()
    obs_shape = obs0["real_obs"].shape      # (jobs, feat)
    n_actions = obs0["action_mask"].shape[0]  # number of possible actions
    env0.close()

    # Build Q-networks
    q_net = QNetwork(obs_shape, n_actions).to(device)
    target_net = QNetwork(obs_shape, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    step_count = 0
    update_count = 0
    eps = eps_start

    episode_returns = []
    episode_lengths = []

    pbar = tqdm(total=total_steps, desc="DQN steps", unit="step")

    # Main training loop
    while step_count < total_steps:
        # 1) Pick a random training instance
        instance_name = random.choice(train_instances)
        inst_path = os.path.join("JSSEnv", "envs", "instances", instance_name)
        env = gym.make("JSSEnv/JssEnv-v1", env_config={"instance_path": inst_path})
        obs_dict, _ = env.reset()
        real_obs = obs_dict["real_obs"].astype(np.float32)      # numpy array (jobs, feat)
        action_mask = obs_dict["action_mask"].astype(np.bool_)  # numpy array (n_actions,)

        episode_return = 0.0
        episode_len = 0
        done = False
        truncated = False

        # To penalize loops, track visited states
        visited = set()
        # Represent a state by the bytes of its arrays
        state_key = (real_obs.tobytes(), action_mask.tobytes())
        visited.add(state_key)

        while not (done or truncated) and step_count < total_steps:
            # 2) Epsilon-greedy action selection
            ro_tensor = torch.tensor(real_obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, jobs, feat)
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)  # (1, n_actions)

            with torch.no_grad():
                q_values = q_net(ro_tensor).squeeze(0)  # (n_actions,)
            q_values[~mask_tensor.squeeze(0)] = -1e8

            if random.random() < eps:
                valid_ids = np.where(action_mask)[0]
                action = int(random.choice(valid_ids))
            else:
                action = int(q_values.argmax().item())

            # 3) Step the environment
            next_obs_dict, reward, done, truncated, info = env.step(action)
            next_real_obs = next_obs_dict["real_obs"].astype(np.float32)
            next_action_mask = next_obs_dict["action_mask"].astype(np.bool_)

            # 4) Reward shaping: time-step penalty
            reward += -0.01

            # 5) Loop penalty: if we revisit the exact same state
            next_key = (next_real_obs.tobytes(), next_action_mask.tobytes())
            if next_key in visited:
                reward += -0.1
            else:
                visited.add(next_key)

            episode_return += reward
            episode_len += 1
            step_count += 1
            pbar.update(1)

            # 6) Store transition in replay buffer
            replay_buffer.append(
                (real_obs, action_mask, action, float(reward), next_real_obs, next_action_mask, float(done or truncated))
            )

            # 7) Advance to next state
            real_obs = next_real_obs
            action_mask = next_action_mask

            # 8) Decay epsilon linearly
            eps = max(eps_final, eps - (eps_start - eps_final) / eps_decay_steps)

            # 9) Double DQN update every train_interval steps
            if len(replay_buffer) >= batch_size and step_count % train_interval == 0:
                batch = random.sample(replay_buffer, batch_size)
                ro_b, am_b, a_b, r_b, nro_b, nam_b, d_b = zip(*batch)

                # Stack numpy arrays
                ro_arr = np.stack(ro_b, axis=0)     # (B, jobs, feat)
                nro_arr = np.stack(nro_b, axis=0)   # (B, jobs, feat)
                am_arr = np.stack(am_b, axis=0)     # (B, n_actions)
                a_arr = np.array(a_b, dtype=np.int64)   # (B,)
                r_arr = np.array(r_b, dtype=np.float32) # (B,)
                d_arr = np.array(d_b, dtype=np.float32) # (B,)

                # Convert to tensors
                ro_t = torch.tensor(ro_arr, dtype=torch.float32, device=device)    # (B, jobs, feat)
                am_t = torch.tensor(am_arr, dtype=torch.bool, device=device)      # (B, n_actions)
                nro_t = torch.tensor(nro_arr, dtype=torch.float32, device=device) # (B, jobs, feat)
                a_t = torch.tensor(a_arr, dtype=torch.int64, device=device).unsqueeze(1)  # (B,1)
                r_t = torch.tensor(r_arr, dtype=torch.float32, device=device)     # (B,)
                d_t = torch.tensor(d_arr, dtype=torch.float32, device=device)     # (B,)

                # --- Q_current = Q_online(s,a) ---
                q_all = q_net(ro_t)                      # (B, n_actions)
                q_current = q_all.gather(1, a_t).squeeze(1)  # (B,)

                # --- Double DQN target computation ---
                with torch.no_grad():
                    # 1) Get best next action from online network
                    q_next_online = q_net(nro_t)            # (B, n_actions)
                    mask_next = torch.full_like(q_next_online, -1e8)
                    for i in range(batch_size):
                        valid_indices = np.where(nam_b[i])[0]
                        mask_next[i, valid_indices] = 0.0
                    next_actions = (q_next_online + mask_next).argmax(dim=1)  # (B,)

                    # 2) Evaluate that action with target network
                    q_next_target = target_net(nro_t)      # (B, n_actions)
                    q_next = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (B,)

                    # 3) Build target: R + gamma * q_next * (1 - done)
                    q_target = r_t + gamma * q_next * (1.0 - d_t)

                # --- Compute Huber loss and update ---
                loss = F.smooth_l1_loss(q_current, q_target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()

                update_count += 1
                if update_count % target_update_freq == 0:
                    target_net.load_state_dict(q_net.state_dict())

                if writer:
                    writer.add_scalar("train/loss", loss.item(), update_count)
                    writer.add_scalar("train/epsilon", eps, update_count)

            # 10) Periodic logging of episode return/length
            if writer and eval_interval > 0 and step_count % eval_interval == 0:
                writer.add_scalar("train/episode_return", episode_return, step_count)
                writer.add_scalar("train/episode_length", episode_len, step_count)

            if (done or truncated):
                break

        # End of one episode
        if writer:
            writer.add_scalar("episode/return", episode_return, step_count)
            writer.add_scalar("episode/length", episode_len, step_count)

        episode_returns.append(episode_return)
        episode_lengths.append(episode_len)
        env.close()

    pbar.close()

    # Save the trained networks
    os.makedirs("models", exist_ok=True)
    torch.save(q_net.state_dict(), "models/dqn_model.pth")
    torch.save(target_net.state_dict(), "models/dqn_target_model.pth")
    if writer:
        writer.close()

    print("Training finished.")
    print(f"Average return (last 10 eps): {np.mean(episode_returns[-10:]):.2f}")
    print(f"Average length (last 10 eps): {np.mean(episode_lengths[-10:]):.2f}")
    print("Saved models to models/dqn_model.pth and models/dqn_target_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Double DQN on JSSenv")
    parser.add_argument(
        "--train-instances", nargs="+",
        default=["ta01", "ta02", "ta03", "ta04", "ta05"],
        help="List of ta instances for training"
    )
    parser.add_argument(
        "--total-steps", type=int, default=300000,
        help="Total env steps to train"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for DQN updates"
    )
    parser.add_argument(
        "--buffer-capacity", type=int, default=100000,
        help="Replay buffer capacity"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate for Adam"
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0,
        help="Discount factor"
    )
    parser.add_argument(
        "--eps-start", type=float, default=1.0,
        help="Initial epsilon for exploration"
    )
    parser.add_argument(
        "--eps-final", type=float, default=0.05,
        help="Final epsilon after decay"
    )
    parser.add_argument(
        "--eps-decay-steps", type=int, default=150000,
        help="Number of steps over which epsilon decays"
    )
    parser.add_argument(
        "--target-update-freq", type=int, default=1000,
        help="Number of gradient updates between copying to target network"
    )
    parser.add_argument(
        "--train-interval", type=int, default=4,
        help="Perform a DQN update every train_interval environment steps"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=0,
        help="If >0, log training metrics every `eval_interval` steps"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device to train on"
    )
    parser.add_argument(
        "--log-dir", type=str, default="runs/dqn",
        help="TensorBoard log directory (set to '' to disable)"
    )

    args = parser.parse_args()

    train_dqn(
        train_instances=args.train_instances,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        lr=args.lr,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_final=args.eps_final,
        eps_decay_steps=args.eps_decay_steps,
        target_update_freq=args.target_update_freq,
        train_interval=args.train_interval,
        eval_interval=args.eval_interval,
        device=args.device,
        log_dir=(args.log_dir if args.log_dir else None)
    )

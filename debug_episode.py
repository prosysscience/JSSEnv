#!/usr/bin/env python3
"""
Debug script to understand episode behavior
"""

import gymnasium as gym
import JSSEnv
import numpy as np

def debug_episode():
    env = gym.make('JSSEnv/JssEnv-v1')
    obs, info = env.reset()
    
    print("=== Episode Debug ===")
    print(f"Initial info: {info}")
    print(f"Action space: {env.action_space}")
    print(f"Initial legal actions: {np.where(obs['action_mask'] == 1)[0]}")
    print(f"Real obs shape: {obs['real_obs'].shape}")
    
    done = False
    truncated = False
    steps = 0
    max_steps = 300
    
    while not (done or truncated) and steps < max_steps:
        legal_actions = np.where(obs['action_mask'] == 1)[0]
        if len(legal_actions) == 0:
            print(f"Step {steps}: No legal actions available!")
            break
            
        action = np.random.choice(legal_actions)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        
        print(f"Step {steps}: action={action}, reward={reward}, done={done}, truncated={truncated}")
        print(f"  Legal actions remaining: {len(np.where(obs['action_mask'] == 1)[0])}")
        if done or truncated:
            print(f"  Final info: {info}")
            print(f"  Makespan: {info.get('makespan', 'not found')}")
    
    if steps >= max_steps:
        print(f"Episode didn't finish in {max_steps} steps")
        print(f"Final info: {info}")
        print(f"Makespan: {info.get('makespan', 'not found')}")
        print(f"Environment current_time_step: {getattr(env, 'current_time_step', 'not found')}")
        print(f"Environment last_time_step: {getattr(env, 'last_time_step', 'not found')}")
        print(f"Done: {done}, Truncated: {truncated}")
    
    env.close()

if __name__ == "__main__":
    debug_episode() 
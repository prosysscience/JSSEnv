"""
Example of using built-in dispatching rules for Job Shop Scheduling.

This script demonstrates how to use the built-in dispatching rules to solve
instances of the Job Shop Scheduling problem. It compares multiple rules on
a given instance and visualizes the results.

Usage:
    python dispatching_rules_example.py [instance_path]

    If no instance_path is provided, it uses the default ta01 instance.
"""

import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
import JSSEnv
from JSSEnv.dispatching import DISPATCHING_RULES, compare_rules

from pathlib import Path


def run_example(instance_path=None):
    """Run the dispatching rules example on the given instance."""
    if instance_path is None:
        # Use default instance
        instance_path = str(Path(__file__).parent.parent / "JSSEnv" / "envs" / "instances" / "ta01")
    
    print(f"Running dispatching rules example on instance: {instance_path}")
    
    # Create environment
    env = gym.make('jss-v1', env_config={"instance_path": instance_path})
    
    # Compare all dispatching rules
    print("\nComparing all dispatching rules...")
    start_time = time.time()
    results = compare_rules(env, num_episodes=5)
    end_time = time.time()
    
    print(f"Comparison took {end_time - start_time:.2f} seconds")
    
    # Print results
    print("\nResults summary:")
    print("=" * 60)
    print(f"{'Rule':<8} {'Avg Reward':<15} {'Avg Makespan':<15}")
    print("-" * 60)
    
    # Sort by average makespan (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_makespan'])
    
    for rule_name, metrics in sorted_results:
        print(f"{rule_name:<8} {metrics['avg_reward']:<15.2f} {metrics['avg_makespan']:<15.2f}")
    
    print("=" * 60)
    
    # Plot results
    plot_results(results)
    
    # Run the best rule once and visualize the Gantt chart
    best_rule = sorted_results[0][0]
    print(f"\nRunning best rule ({best_rule}) once to generate a Gantt chart...")
    
    env.reset()
    rule = DISPATCHING_RULES[best_rule]
    
    # Run episode
    obs = env.reset()
    done = False
    
    while not done:
        action = rule(env)
        obs, reward, done, truncated, _ = env.step(action)
    
    # Render the Gantt chart
    fig = env.render()
    fig.update_layout(
        title=f"Gantt Chart for {rule.get_name()} - Makespan: {env.current_time_step}",
        xaxis_title="Time",
        yaxis_title="Machine",
    )
    
    # Save the figure
    gantt_path = Path(__file__).parent / f"gantt_{rule.get_name()}.html"
    fig.write_html(str(gantt_path))
    print(f"Gantt chart saved to {gantt_path}")
    
    # Example of detailed rule usage
    print("\nExample of using a dispatching rule directly:")
    print("-" * 60)
    rule = DISPATCHING_RULES["SPT"]
    print(f"Rule: {rule.get_name()} - {rule.get_description()}")
    
    env.reset()
    obs = env.reset()
    done = False
    steps = 0
    total_reward = 0
    
    while not done and steps < 10:  # Only show first 10 steps
        action = rule(env)
        if steps < 5:  # Show details only for first 5 steps
            print(f"Step {steps}: Selected job {action}")
            
            # Show number of legal actions
            legal_actions = env.get_legal_actions()
            num_legal = np.sum(legal_actions[:-1])  # Exclude no-op
            print(f"  Legal actions: {num_legal}")
            
            # Show processing times of legal jobs
            for job in range(env.jobs):
                if legal_actions[job]:
                    current_op = env.todo_time_step_job[job]
                    process_time = env.instance_matrix[job][current_op][1]
                    print(f"  Job {job}: Processing time = {process_time}")
        
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
    
    print("...")  # Indicate that more steps would follow
    print(f"Episode finished with reward {total_reward:.2f} and makespan {env.current_time_step}")
    

def plot_results(results):
    """Plot the results of the dispatching rules comparison."""
    rule_names = list(results.keys())
    
    # Extract the metrics
    makespans = [results[rule]['avg_makespan'] for rule in rule_names]
    rewards = [results[rule]['avg_reward'] for rule in rule_names]
    
    # Sort by makespan
    indices = np.argsort(makespans)
    sorted_rules = [rule_names[i] for i in indices]
    sorted_makespans = [makespans[i] for i in indices]
    sorted_rewards = [rewards[i] for i in indices]
    
    # Create a dataframe for easier plotting
    df = pd.DataFrame({
        'Rule': sorted_rules,
        'Makespan': sorted_makespans,
        'Reward': sorted_rewards
    })
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot makespans
    ax1.bar(df['Rule'], df['Makespan'], color='skyblue')
    ax1.set_title('Average Makespan by Dispatching Rule')
    ax1.set_xlabel('Dispatching Rule')
    ax1.set_ylabel('Makespan (lower is better)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot rewards
    ax2.bar(df['Rule'], df['Reward'], color='lightgreen')
    ax2.set_title('Average Reward by Dispatching Rule')
    ax2.set_xlabel('Dispatching Rule')
    ax2.set_ylabel('Reward (higher is better)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = Path(__file__).parent / "dispatching_results.png"
    plt.savefig(plot_path)
    print(f"Results plot saved to {plot_path}")


if __name__ == "__main__":
    # Get instance path from command line or use default
    instance_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    run_example(instance_path)
#!/usr/bin/env python3
"""
Comprehensive Baseline Evaluation Script for Job Shop Scheduling

This script evaluates all implemented baseline policies and compares their performance
across multiple instances and runs, providing detailed statistics and analysis.
"""

import os
import sys
import time
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym

# Import JSSEnv to register the environment
import JSSEnv

# Add the current directory to Python path to import baselines
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from baselines import RandomPolicy, SPTPolicy, SimulatedAnnealingPolicy
    from baselines.utils import format_results_table
except ImportError as e:
    print(f"Error importing baselines: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

# Default test instances - modify these based on your JSSEnv setup
DEFAULT_INSTANCES = [
    "ta01", "ta02", "ta03", "ta04", "ta05",
    "ft06", "ft10", "ft20"
]

# Simulated Annealing configurations to test
SA_CONFIGS = {
    "SA_Quick": {
        'initial_temp': 100.0,
        'cooling_rate': 0.95,
        'max_iter_per_restart': 50,
        'num_restarts': 3,
        'seed': 42
    },
    "SA_Standard": {
        'initial_temp': 100.0,
        'cooling_rate': 0.95,
        'max_iter_per_restart': 100,
        'num_restarts': 5,
        'seed': 42
    },
    "SA_Intensive": {
        'initial_temp': 200.0,
        'cooling_rate': 0.98,
        'max_iter_per_restart': 200,
        'num_restarts': 10,
        'seed': 42
    }
}

class BaselineEvaluator:
    def __init__(self, instances=None, num_runs=5, output_dir="evaluation_results"):
        self.instances = instances or DEFAULT_INSTANCES
        self.num_runs = num_runs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.results = {}
        self.detailed_results = []
        self.errors = []
        
        # Timing
        self.start_time = None
        self.total_time = 0
        
    def create_environment(self, instance_id):
        """Create environment for the given instance."""
        try:
            # Try different environment creation patterns
            env_configs = [
                {'instance_path': instance_id},
                {'instance_name': instance_id},
                {'instance': instance_id}
            ]
            
            for config in env_configs:
                try:
                    env = gym.make('JSSEnv/JssEnv-v1', env_config=config)
                    print(f"  ✓ Created environment for {instance_id} with config: {config}")
                    return env
                except Exception as e:
                    continue
            
            # If all configs fail, try without env_config
            env = gym.make('JSSEnv/JssEnv-v1')
            print(f"  ✓ Created default environment (instance: {instance_id})")
            return env
            
        except Exception as e:
            error_msg = f"Failed to create environment for instance '{instance_id}': {e}"
            print(f"  ✗ {error_msg}")
            self.errors.append(error_msg)
            return None
    
    def run_single_episode(self, env, policy, policy_name, instance_id, run_id):
        """Run a single episode with the given policy."""
        start_time = time.time()
        
        try:
            # Reset policy if it has a reset method
            if hasattr(policy, 'reset'):
                policy.reset()
            
            # Reset environment
            obs, info = env.reset()
            done = False
            truncated = False
            step_count = 0
            max_steps = 10000  # Safety limit
            
            while not (done or truncated) and step_count < max_steps:
                try:
                    action = policy.select_action(obs)
                    obs, reward, done, truncated, info = env.step(action)
                    step_count += 1
                except Exception as e:
                    error_msg = f"Error during step {step_count} with {policy_name} on {instance_id}: {e}"
                    print(f"    ✗ {error_msg}")
                    self.errors.append(error_msg)
                    return None
            
            runtime = time.time() - start_time
            
            # Get makespan from info or environment's current_time_step
            makespan = info.get('makespan', None)
            if makespan is None:
                # If makespan not in info, try to get it from the environment
                if hasattr(env.unwrapped, 'current_time_step'):
                    makespan = env.unwrapped.current_time_step
                elif hasattr(env.unwrapped, 'last_time_step'):
                    makespan = env.unwrapped.last_time_step
                else:
                    makespan = float('inf')
            
            if step_count >= max_steps:
                print(f"    ⚠ {policy_name} on {instance_id} exceeded max steps ({max_steps})")
                makespan = float('inf')
            
            result = {
                'policy': policy_name,
                'instance': instance_id,
                'run_id': run_id,
                'makespan': makespan,
                'runtime': runtime,
                'steps': step_count,
                'success': makespan != float('inf') and (done or truncated)
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Error running {policy_name} on {instance_id}, run {run_id}: {e}"
            print(f"    ✗ {error_msg}")
            self.errors.append(error_msg)
            traceback.print_exc()
            return None
    
    def create_policy(self, policy_name, env, config=None):
        """Create a policy instance."""
        try:
            if policy_name == "RandomPolicy":
                return RandomPolicy(env)
            elif policy_name == "SPTPolicy":
                return SPTPolicy(env)
            elif policy_name.startswith("SA_"):
                sa_config = config or SA_CONFIGS.get(policy_name, SA_CONFIGS["SA_Standard"])
                return SimulatedAnnealingPolicy(env, **sa_config)
            else:
                raise ValueError(f"Unknown policy: {policy_name}")
        except Exception as e:
            error_msg = f"Error creating {policy_name}: {e}"
            print(f"    ✗ {error_msg}")
            self.errors.append(error_msg)
            return None
    
    def evaluate_policy_on_instance(self, policy_name, instance_id, config=None):
        """Evaluate a single policy on a single instance."""
        print(f"  📊 Evaluating {policy_name} on {instance_id}")
        
        env = self.create_environment(instance_id)
        if env is None:
            return []
        
        policy = self.create_policy(policy_name, env, config)
        if policy is None:
            env.close()
            return []
        
        results = []
        for run_id in range(self.num_runs):
            print(f"    🔄 Run {run_id + 1}/{self.num_runs}", end=" ")
            
            result = self.run_single_episode(env, policy, policy_name, instance_id, run_id)
            if result:
                results.append(result)
                print(f"✓ Makespan: {result['makespan']:.2f}, Time: {result['runtime']:.3f}s")
            else:
                print("✗ Failed")
        
        env.close()
        return results
    
    def run_evaluation(self):
        """Run the complete evaluation."""
        self.start_time = time.time()
        print("🚀 Starting Baseline Evaluation")
        print(f"📋 Instances: {self.instances}")
        print(f"🔢 Runs per instance: {self.num_runs}")
        print(f"📁 Output directory: {self.output_dir}")
        print("=" * 80)
        
        # Define all policies to test
        policies_to_test = [
            ("RandomPolicy", None),
            ("SPTPolicy", None),
        ]
        
        # Add SA configurations
        for sa_name, sa_config in SA_CONFIGS.items():
            policies_to_test.append((sa_name, sa_config))
        
        # Run evaluation for each policy and instance
        for policy_name, config in policies_to_test:
            print(f"\n🎯 Testing {policy_name}")
            print("-" * 40)
            
            policy_results = []
            for instance_id in self.instances:
                instance_results = self.evaluate_policy_on_instance(policy_name, instance_id, config)
                policy_results.extend(instance_results)
            
            if policy_results:
                self.results[policy_name] = policy_results
                self.detailed_results.extend(policy_results)
                
                # Print summary for this policy
                makespans = [r['makespan'] for r in policy_results if r['makespan'] != float('inf')]
                if makespans:
                    print(f"  📈 {policy_name} Summary:")
                    print(f"    Average Makespan: {np.mean(makespans):.2f}")
                    print(f"    Std Dev: {np.std(makespans):.2f}")
                    print(f"    Min: {np.min(makespans):.2f}")
                    print(f"    Max: {np.max(makespans):.2f}")
                    print(f"    Success Rate: {len(makespans)}/{len(policy_results)} ({100*len(makespans)/len(policy_results):.1f}%)")
        
        self.total_time = time.time() - self.start_time
        print(f"\n⏱ Total evaluation time: {self.total_time:.2f} seconds")
    
    def analyze_results(self):
        """Analyze and summarize the results."""
        if not self.detailed_results:
            print("❌ No results to analyze!")
            return
        
        print("\n" + "=" * 80)
        print("📊 DETAILED ANALYSIS")
        print("=" * 80)
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.detailed_results)
        
        # Overall summary
        print("\n🎯 Overall Performance Summary:")
        print("-" * 40)
        
        summary_stats = []
        for policy in df['policy'].unique():
            policy_data = df[df['policy'] == policy]
            valid_makespans = policy_data[policy_data['makespan'] != float('inf')]['makespan']
            
            if len(valid_makespans) > 0:
                stats = {
                    'Policy': policy,
                    'Avg_Makespan': valid_makespans.mean(),
                    'Std_Makespan': valid_makespans.std(),
                    'Min_Makespan': valid_makespans.min(),
                    'Max_Makespan': valid_makespans.max(),
                    'Avg_Runtime': policy_data['runtime'].mean(),
                    'Success_Rate': len(valid_makespans) / len(policy_data),
                    'Total_Runs': len(policy_data)
                }
                summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        if not summary_df.empty:
            summary_df = summary_df.sort_values('Avg_Makespan')
            print(summary_df.to_string(index=False, float_format='%.3f'))
        
        # Per-instance analysis
        print("\n📋 Per-Instance Performance:")
        print("-" * 40)
        
        for instance in df['instance'].unique():
            print(f"\n🏭 Instance: {instance}")
            instance_data = df[df['instance'] == instance]
            
            instance_summary = []
            for policy in instance_data['policy'].unique():
                policy_instance_data = instance_data[instance_data['policy'] == policy]
                valid_makespans = policy_instance_data[policy_instance_data['makespan'] != float('inf')]['makespan']
                
                if len(valid_makespans) > 0:
                    instance_summary.append({
                        'Policy': policy,
                        'Avg_Makespan': valid_makespans.mean(),
                        'Std_Makespan': valid_makespans.std(),
                        'Avg_Runtime': policy_instance_data['runtime'].mean(),
                        'Success_Rate': len(valid_makespans) / len(policy_instance_data)
                    })
            
            if instance_summary:
                instance_df = pd.DataFrame(instance_summary)
                instance_df = instance_df.sort_values('Avg_Makespan')
                print(instance_df.to_string(index=False, float_format='%.3f'))
        
        # Best performing policy per instance
        print("\n🏆 Best Policy Per Instance:")
        print("-" * 40)
        
        best_policies = []
        for instance in df['instance'].unique():
            instance_data = df[df['instance'] == instance]
            best_makespan = float('inf')
            best_policy = None
            
            for policy in instance_data['policy'].unique():
                policy_data = instance_data[instance_data['policy'] == policy]
                valid_makespans = policy_data[policy_data['makespan'] != float('inf')]['makespan']
                
                if len(valid_makespans) > 0:
                    avg_makespan = valid_makespans.mean()
                    if avg_makespan < best_makespan:
                        best_makespan = avg_makespan
                        best_policy = policy
            
            if best_policy:
                best_policies.append({
                    'Instance': instance,
                    'Best_Policy': best_policy,
                    'Best_Makespan': best_makespan
                })
        
        if best_policies:
            best_df = pd.DataFrame(best_policies)
            print(best_df.to_string(index=False, float_format='%.3f'))
    
    def save_results(self):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as CSV
        if self.detailed_results:
            df = pd.DataFrame(self.detailed_results)
            csv_file = self.output_dir / f"baseline_results_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"\n💾 Detailed results saved to: {csv_file}")
        
        # Save summary as JSON
        summary = {
            'evaluation_info': {
                'timestamp': timestamp,
                'instances': self.instances,
                'num_runs': self.num_runs,
                'total_time': self.total_time,
                'num_errors': len(self.errors)
            },
            'results': self.results,
            'errors': self.errors
        }
        
        json_file = self.output_dir / f"baseline_summary_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"💾 Summary saved to: {json_file}")
        
        # Save errors if any
        if self.errors:
            error_file = self.output_dir / f"baseline_errors_{timestamp}.txt"
            with open(error_file, 'w') as f:
                f.write(f"Evaluation Errors ({len(self.errors)} total)\n")
                f.write("=" * 50 + "\n\n")
                for i, error in enumerate(self.errors, 1):
                    f.write(f"{i}. {error}\n\n")
            print(f"⚠️  Errors saved to: {error_file}")
    
    def print_errors(self):
        """Print any errors that occurred during evaluation."""
        if self.errors:
            print(f"\n⚠️  {len(self.errors)} errors occurred during evaluation:")
            print("-" * 40)
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline policies for Job Shop Scheduling")
    parser.add_argument('--instances', nargs='+', default=DEFAULT_INSTANCES,
                        help='List of instance identifiers to test')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs per instance per policy')
    parser.add_argument('--output-dir', default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick evaluation with fewer SA iterations')
    
    args = parser.parse_args()
    
    # Modify SA configs for quick evaluation
    if args.quick:
        for config in SA_CONFIGS.values():
            config['max_iter_per_restart'] = 20
            config['num_restarts'] = 2
        print("🚀 Running quick evaluation mode")
    
    # Create evaluator and run
    evaluator = BaselineEvaluator(
        instances=args.instances,
        num_runs=args.runs,
        output_dir=args.output_dir
    )
    
    try:
        evaluator.run_evaluation()
        evaluator.analyze_results()
        evaluator.save_results()
        evaluator.print_errors()
        
        print("\n" + "=" * 80)
        print("✅ Evaluation completed successfully!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        evaluator.save_results()
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        traceback.print_exc()
        evaluator.save_results()
        sys.exit(1)


if __name__ == "__main__":
    main() 
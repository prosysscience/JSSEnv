#!/usr/bin/env python3
"""
Comprehensive Baseline + DQN Evaluation Script for Job Shop Scheduling

This script evaluates all implemented baseline policies (Random, SPT, SA variants)
and also a pretrained DQN agent on a collection of JSSEnv instances. It then
prints per-instance summaries and overall statistics, and saves detailed results
as CSV/JSON in the specified output directory.

Usage:
    python dqn_vs_baselines_test.py --instances ta01 ta02 ta03 ta04 ta05 --runs 5 --dqn-model-path models/dqn_improved_model.pth --output-dir evaluation_results
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
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import JSSEnv to register the environment
import JSSEnv

# Ensure that we can import our baselines folder from project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

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

# ─── Q‐NETWORK DEFINITION (same architecture used in train_dqn.py) ───
class QNetwork(nn.Module):
    """
    Dueling Q-Network: separate streams for state-value V(s) and advantage A(s,a).
    """
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        jobs, feat = obs_shape
        input_size = jobs * feat

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # Dueling heads
        self.value_head = nn.Linear(128, 1)
        self.adv_head = nn.Linear(128, n_actions)

    def forward(self, real_obs: torch.Tensor):
        B = real_obs.shape[0]
        x = real_obs.view(B, -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value_head(x)             # shape (B, 1)
        adv   = self.adv_head(x)               # shape (B, n_actions)
        adv_mean = adv.mean(dim=1, keepdim=True)  # shape (B, 1)

        q = value + (adv - adv_mean)           # Broadcasting: (B,1) + (B,n_actions)
        return q



class BaselineEvaluator:
    def __init__(self, instances=None, num_runs=5, output_dir="evaluation_results",
                 dqn_model_path=None, device="cpu"):
        self.instances = instances or DEFAULT_INSTANCES
        self.num_runs = num_runs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Optional DQN
        self.dqn_model_path = dqn_model_path
        self.device = torch.device(device)
        self.q_net = None
        self.obs_shape = None
        self.n_actions = None

        # Results storage
        self.results = {}           # keyed by policy name → list of episode‐level dicts
        self.detailed_results = []  # flat list of all episodes
        self.errors = []

        # Timing
        self.start_time = None
        self.total_time = 0

        # If a DQN path is provided, defer loading until we know obs/action shapes
        # (we'll call _load_dqn_model() as soon as we can)
        if self.dqn_model_path is not None:
            # We delay the actual loading until after __init__, because we need to know
            # the obs/action shape (via a dummy reset). So we just leave self.q_net = None
            # for now, and explicitly call _load_dqn_model() in run_evaluation().
            pass

    def _get_instance_path(self, instance_name: str) -> str:
        """
        Build and return the full filesystem path to a given instance folder,
        essentially: <project_root>/JSSEnv/envs/instances/<instance_name>
        """
        inst_rel_path = os.path.join("JSSEnv", "envs", "instances", instance_name)
        # First try absolute from project_root
        candidate = os.path.join(project_root, inst_rel_path)
        if os.path.exists(candidate):
            return candidate
        # If not, maybe the user is already running from within project directory.
        if os.path.exists(inst_rel_path):
            return inst_rel_path
        raise FileNotFoundError(f"Instance '{instance_name}' not found at either '{candidate}' or '{inst_rel_path}'")

    def _load_dqn_model(self):
        """
        Instantiate a dummy env (first instance) to get obs_shape & action mask size,
        then load the QNetwork state_dict.
        """
        inst0 = self.instances[0]
        try:
            inst0_path = self._get_instance_path(inst0)
        except FileNotFoundError as e:
            print(f"✗ {self._get_instance_path.__name__}: {e}")
            sys.exit(1)

        tmp_env = gym.make("JSSEnv/JssEnv-v1", env_config={"instance_path": inst0_path})
        # Gymnasium reset returns (obs, info)
        obs0, _ = tmp_env.reset()
        # obs0["real_obs"] is something like an array of shape (jobs, feat)
        self.obs_shape = obs0["real_obs"].shape         # e.g. (jobs, 7)
        self.n_actions = obs0["action_mask"].shape[0]   # e.g. jobs+1
        tmp_env.close()

        # Build QNetwork and load weights
        self.q_net = QNetwork(self.obs_shape, self.n_actions).to(self.device)
        self.q_net.load_state_dict(torch.load(self.dqn_model_path, map_location=self.device))
        self.q_net.eval()
        print(f"✓ Loaded DQN model from `{self.dqn_model_path}` onto {self.device}.\n")

    def create_environment(self, instance_id: str):
        """Create and return a fresh JSSEnv for a given instance ID."""
        try:
            instance_path_full = self._get_instance_path(instance_id)
            env = gym.make(
                "JSSEnv/JssEnv-v1",
                env_config={"instance_path": instance_path_full}
            )
            return env
        except Exception as e:
            msg = f"Failed to create environment for '{instance_id}': {e}"
            print(f"  ✗ {msg}")
            self.errors.append(msg)
            return None

    def run_single_episode(self, env, policy, policy_name, instance_id, run_id):
        """
        Run one episode with `policy.select_action(obs)` returning an action int.
        Policy may be any object with a `select_action(obs)` method.
        """
        start = time.time()
        try:
            if hasattr(policy, 'reset'):
                policy.reset()

            # Gymnasium reset
            obs_dict, info = env.reset()
            done = False
            truncated = False
            steps = 0
            max_steps = 10000

            while not (done or truncated) and steps < max_steps:
                action = policy.select_action(obs_dict)
                obs_dict, reward, done, truncated, info = env.step(action)
                steps += 1

            duration = time.time() - start
            # Extract makespan
            if "makespan" in info:
                ms = info["makespan"]
            else:
                # Fallback: dig through wrappers
                raw = env
                while hasattr(raw, "env"):
                    raw = raw.env
                ms = getattr(raw, "current_time_step", float("inf"))

            success = info.get("all_jobs_completed", False) and ms < float("inf")
            if steps >= max_steps:
                ms = float("inf")

            return {
                'policy': policy_name,
                'instance': instance_id,
                'run_id': run_id,
                'makespan': ms,
                'runtime': duration,
                'steps': steps,
                'success': success
            }

        except Exception as e:
            msg = f"Error during {policy_name} @ {instance_id} run {run_id}: {e}"
            print(f"    ✗ {msg}")
            self.errors.append(msg)
            return None

    def create_policy(self, policy_name, env, config=None):
        """
        Instantiate the appropriate baseline policy object for `policy_name`.
        """
        try:
            if policy_name == "RandomPolicy":
                return RandomPolicy(env)
            elif policy_name == "SPTPolicy":
                return SPTPolicy(env)
            elif policy_name.startswith("SA_"):
                sa_cfg = config or SA_CONFIGS.get(policy_name)
                return SimulatedAnnealingPolicy(env, **sa_cfg)
            else:
                raise ValueError(f"Unknown policy: {policy_name}")
        except Exception as e:
            msg = f"Error creating {policy_name}: {e}"
            print(f"    ✗ {msg}")
            self.errors.append(msg)
            return None

    def evaluate_policy_on_instance(self, policy_name, instance_id, config=None):
        """
        Evaluate a single baseline policy on a single instance for `self.num_runs` episodes.
        Returns a list of episode‐level result dicts.
        """
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
            print(f"    🔄 Run {run_id+1}/{self.num_runs}", end=" ")
            res = self.run_single_episode(env, policy, policy_name, instance_id, run_id)
            if res:
                results.append(res)
                print(f"✓ Makespan: {res['makespan']:.2f}, Time: {res['runtime']:.3f}s")
            else:
                print("✗ Failed")
        env.close()
        return results

    def evaluate_dqn_on_instance(self, instance_id: str):
        """
        Evaluate the pretrained DQN on one instance for `self.num_runs` episodes.
        Returns a list of episode‐level result dicts with policy="DQN".
        """
        if self.q_net is None:
            return []

        print(f"  📊 Evaluating DQN on {instance_id}")
        try:
            inst_path = self._get_instance_path(instance_id)
        except FileNotFoundError as e:
            msg = f"DQN evaluation failed: {e}"
            print(f"    ✗ {msg}")
            self.errors.append(msg)
            return []

        base_env = gym.make("JSSEnv/JssEnv-v1", env_config={"instance_path": inst_path})
        env = gym.wrappers.RecordEpisodeStatistics(base_env)

        results = []
        for run_id in range(self.num_runs):
            print(f"    🔄 Run {run_id+1}/{self.num_runs}", end=" ")
            obs_dict, info = env.reset(seed=300 + run_id)
            done = False
            truncated = False
            steps = 0
            start = time.time()

            while not (done or truncated):
                ro = torch.tensor(obs_dict["real_obs"], dtype=torch.float32, device=self.device).unsqueeze(0)
                mask = torch.tensor(obs_dict["action_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    qv = self.q_net(ro)  # (1, n_actions)
                qv[~mask] = -1e8
                action = int(qv.argmax(dim=1).item())

                obs_dict, reward, done, truncated, info = env.step(action)
                steps += 1

            duration = time.time() - start
            if "makespan" in info:
                ms = info["makespan"]
            else:
                raw = env
                while hasattr(raw, "env"):
                    raw = raw.env
                ms = getattr(raw, "current_time_step", float("inf"))
            success = info.get("all_jobs_completed", False) and ms < float("inf")

            results.append({
                'policy': "DQN",
                'instance': instance_id,
                'run_id': run_id,
                'makespan': ms,
                'runtime': duration,
                'steps': steps,
                'success': success
            })
            print(f"✓ Makespan: {ms:.2f}, Time: {duration:.3f}s")

        env.close()
        return results

    def run_evaluation(self):
        """Run baseline policies, then optionally DQN, across all self.instances."""
        self.start_time = time.time()
        print("🚀 Starting Evaluation")
        print(f"📋 Instances: {self.instances}")
        print(f"🔢 Runs per instance: {self.num_runs}")
        print(f"📁 Output directory: {self.output_dir}")
        if self.dqn_model_path:
            # Now that we know instances, load the DQN model (so obs_shape/n_actions are set)
            self._load_dqn_model()
            print(f"🤖 DQN model: {self.dqn_model_path} (on {self.device})")
        print("=" * 80)

        # 1) Evaluate all baseline policies
        baseline_policies = [
            ("RandomPolicy", None),
            ("SPTPolicy", None),
        ]
        # Add SA variations
        for sa_name, sa_cfg in SA_CONFIGS.items():
            baseline_policies.append((sa_name, sa_cfg))

        for (policy_name, cfg) in baseline_policies:
            print(f"\n🎯 Testing {policy_name}")
            print("-" * 40)
            all_results_for_policy = []
            for inst in self.instances:
                res_list = self.evaluate_policy_on_instance(policy_name, inst, cfg)
                all_results_for_policy.extend(res_list)
            if all_results_for_policy:
                self.results[policy_name] = all_results_for_policy
                self.detailed_results.extend(all_results_for_policy)

                # Print a quick summary for this baseline
                makespans = [r['makespan'] for r in all_results_for_policy if np.isfinite(r['makespan'])]
                if makespans:
                    print(f"  📈 {policy_name} Summary:")
                    print(f"    Avg Makespan: {np.mean(makespans):.2f}")
                    print(f"    Std Dev:       {np.std(makespans):.2f}")
                    print(f"    Min:           {np.min(makespans):.2f}")
                    print(f"    Max:           {np.max(makespans):.2f}")
                    success_count = sum(1 for r in all_results_for_policy if r['makespan'] < float('inf'))
                    total_count = len(all_results_for_policy)
                    print(f"    Success Rate:  {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")

        # 2) Evaluate DQN (if provided)
        if self.q_net is not None:
            print(f"\n🎯 Testing DQN")
            print("-" * 40)
            all_dqn_results = []
            for inst in self.instances:
                res_list = self.evaluate_dqn_on_instance(inst)
                all_dqn_results.extend(res_list)
            if all_dqn_results:
                self.results["DQN"] = all_dqn_results
                self.detailed_results.extend(all_dqn_results)

                # Print DQN summary
                makespans = [r['makespan'] for r in all_dqn_results if np.isfinite(r['makespan'])]
                if makespans:
                    print(f"    DQN Summary:")
                    print(f"    Avg Makespan: {np.mean(makespans):.2f}")
                    print(f"    Std Dev:       {np.std(makespans):.2f}")
                    print(f"    Min:           {np.min(makespans):.2f}")
                    print(f"    Max:           {np.max(makespans):.2f}")
                    success_count = sum(1 for r in all_dqn_results if r['makespan'] < float('inf'))
                    total_count = len(all_dqn_results)
                    print(f"    Success Rate:  {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")

        self.total_time = time.time() - self.start_time
        print(f"\n Total evaluation time: {self.total_time:.2f} sec")

    def analyze_results(self):
        """Analyze and summarize all collected results."""
        if not self.detailed_results:
            print("No results to analyze!")
            return

        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS")
        print("=" * 80)

        df = pd.DataFrame(self.detailed_results)

        # Overall summary by policy
        print("\n Overall Performance Summary:")
        print("-" * 40)
        summary_stats = []
        for pol in df['policy'].unique():
            pdata = df[df['policy'] == pol]
            valid_ms = pdata[pdata['makespan'] != float('inf')]['makespan']
            if len(valid_ms) > 0:
                stats = {
                    'Policy': pol,
                    'Avg_Makespan': float(valid_ms.mean()),
                    'Std_Makespan': float(valid_ms.std()),
                    'Min_Makespan': float(valid_ms.min()),
                    'Max_Makespan': float(valid_ms.max()),
                    'Avg_Runtime': float(pdata['runtime'].mean()),
                    'Success_Rate': float(len(valid_ms) / len(pdata)),
                    'Total_Runs': len(pdata)
                }
                summary_stats.append(stats)

        summary_df = pd.DataFrame(summary_stats)
        if not summary_df.empty:
            summary_df = summary_df.sort_values('Avg_Makespan')
            print(summary_df.to_string(index=False, float_format='%.3f'))

        # Per‐instance performance
        print("\n Per-instance Performance:")
        print("-" * 40)
        for inst in df['instance'].unique():
            print(f"\n🏭 Instance: {inst}")
            inst_df = df[df['instance'] == inst]
            inst_summary = []
            for pol in inst_df['policy'].unique():
                pol_df = inst_df[inst_df['policy'] == pol]
                valid_ms = pol_df[pol_df['makespan'] != float('inf')]['makespan']
                if len(valid_ms) > 0:
                    inst_summary.append({
                        'Policy': pol,
                        'Avg_Makespan': float(valid_ms.mean()),
                        'Std_Makespan': float(valid_ms.std()),
                        'Avg_Runtime': float(pol_df['runtime'].mean()),
                        'Success_Rate': float(len(valid_ms) / len(pol_df))
                    })
            if inst_summary:
                inst_summary_df = pd.DataFrame(inst_summary).sort_values('Avg_Makespan')
                print(inst_summary_df.to_string(index=False, float_format='%.3f'))

        # Best policy per instance
        print("\n🏆 Best Policy Per Instance:")
        print("-" * 40)
        best_list = []
        for inst in df['instance'].unique():
            inst_df = df[df['instance'] == inst]
            best_ms = float('inf')
            best_pol = None
            for pol in inst_df['policy'].unique():
                pol_df = inst_df[inst_df['policy'] == pol]
                valid_ms = pol_df[pol_df['makespan'] != float('inf')]['makespan']
                if len(valid_ms) > 0:
                    avg_ms = float(valid_ms.mean())
                    if avg_ms < best_ms:
                        best_ms = avg_ms
                        best_pol = pol
            if best_pol is not None:
                best_list.append({'Instance': inst, 'Best_Policy': best_pol, 'Best_Makespan': best_ms})
        if best_list:
            best_df = pd.DataFrame(best_list)
            print(best_df.to_string(index=False, float_format='%.3f'))

    def save_results(self):
        """Save detailed results + summary to CSV/JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results as CSV
        if self.detailed_results:
            df = pd.DataFrame(self.detailed_results)
            csv_file = self.output_dir / f"evaluation_results_{timestamp}.csv"
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
        json_file = self.output_dir / f"evaluation_summary_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"💾 Summary saved to: {json_file}")

        # If there were any errors, log them out
        if self.errors:
            err_file = self.output_dir / f"evaluation_errors_{timestamp}.txt"
            with open(err_file, 'w') as f:
                f.write(f"Errors ({len(self.errors)}):\n")
                f.write("="*50 + "\n")
                for i, e in enumerate(self.errors, 1):
                    f.write(f"{i}. {e}\n")
            print(f" Errors saved to: {err_file}")

    def print_errors(self):
        """Print any captured errors to console."""
        if self.errors:
            print(f"\n  {len(self.errors)} errors occurred during evaluation:")
            print("-" * 40)
            for i, e in enumerate(self.errors, 1):
                print(f"{i}. {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline policies (and optionally DQN) for JSSP"
    )
    parser.add_argument(
        '--instances', nargs='+', default=DEFAULT_INSTANCES,
        help='List of instance identifiers (folder names under JSSEnv/envs/instances)'
    )
    parser.add_argument(
        '--runs', type=int, default=5,
        help='Number of episodes per instance per policy'
    )
    parser.add_argument(
        '--dqn-model-path', type=str, default=None,
        help='Path to pretrained DQN model (.pth). If provided, DQN is evaluated too.'
    )
    parser.add_argument(
        '--device', type=str, default="cpu", choices=["cpu", "cuda"],
        help='Device for DQN inference (if used)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='evaluation_results',
        help='Directory where results (CSV/JSON) will be saved'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick SA variants (fewer iterations)'
    )

    args = parser.parse_args()

    # If quick mode, shrink SA configs
    if args.quick:
        for cfg in SA_CONFIGS.values():
            cfg['max_iter_per_restart'] = 20
            cfg['num_restarts'] = 2
        print("🚀 Running quick evaluation mode (SA variants downscaled)\n")

    evaluator = BaselineEvaluator(
        instances=args.instances,
        num_runs=args.runs,
        output_dir=args.output_dir,
        dqn_model_path=args.dqn_model_path,
        device=args.device
    )

    try:
        evaluator.run_evaluation()
        evaluator.analyze_results()
        evaluator.save_results()
        evaluator.print_errors()

        print("\n" + "="*80)
        print("Evaluation completed successfully!")
        print("="*80)

    except KeyboardInterrupt:
        print("\n Evaluation interrupted by user")
        evaluator.save_results()
    except Exception as e:
        print(f"\n Evaluation failed with error: {e}")
        traceback.print_exc()
        evaluator.save_results()
        sys.exit(1)


if __name__ == "__main__":
    main()

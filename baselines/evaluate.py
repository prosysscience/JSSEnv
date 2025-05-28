import gymnasium as gym
import time
import numpy as np
import importlib # To check for JSSEnv

# Attempt to import JSSEnv and provide a helpful message if not found
try:
    importlib.import_module('JSSEnv')
except ImportError:
    print("Error: JSSEnv module not found. Please ensure it is installed.")
    print("You might need to install it, e.g., via 'pip install JSSEnv' or 'pip install -e .'")
    print("if it's a local package.")
    # Optionally, exit here if JSSEnv is critical for the script to run
    # import sys
    # sys.exit(1)

from . import RandomPolicy, SPTPolicy, SimulatedAnnealingPolicy 
from .lwkr_policy import LWKRPolicy
from .critical_path_policy import CriticalPathPolicy
from .utils import format_results_table

# These are placeholder instance identifiers.
# The user MUST ensure these are valid for their JSSEnv setup.
# JSSEnv might load instances by short names (e.g., 'ta01', 'ft06') if they are bundled,
# or it might require full file paths.
# Example: DEFAULT_INSTANCE_IDENTIFIERS = ["ta01", "ta02", "ft06"] # If short names work
# Example: DEFAULT_INSTANCE_IDENTIFIERS = [
#    "path/to/instances/ta01.txt", 
#    "path/to/instances/ft06.txt"
# ] # If full paths are needed
DEFAULT_INSTANCE_IDENTIFIERS = ["ta01", "ta02"] # Modify as needed

def run_episode(env, policy, policy_name="Policy", seed=None):
    """
    Runs a single episode with the given policy on the environment.
    Args:
        env: The Gym environment instance.
        policy: The policy instance to use.
        policy_name (str): Name of the policy for logging.
        seed (int, optional): Seed for environment reset. Policies should be seeded at init.
    Returns:
        dict: A dictionary containing 'makespan' and 'runtime'.
    """
    start_time = time.time()
    
    # Policy reset (important for stateful policies like SA)
    if hasattr(policy, 'reset'):
        policy.reset() # Calls the reset method defined in BasePolicy and overridden if needed

    # Environment reset
    # Newer gym versions allow passing seed directly to env.reset()
    # obs, info = env.reset(seed=seed)
    # For broader compatibility, assuming seeding is handled at env/policy creation
    # or via a global seed if necessary for non-SA policies.
    try:
        obs, info = env.reset() # Add options={'seed': seed} if using gym 0.26+ and want per-episode env seed
    except TypeError: # Older gym might not support options dict
        obs, info = env.reset()


    done = False
    truncated = False 
    
    step_count = 0
    max_steps = 10000 # Safety break for non-terminating episodes

    while not (done or truncated):
        if step_count > max_steps:
            print(f"Warning: {policy_name} on {env.spec.id if env.spec else 'env'} exceeded max_steps ({max_steps}). Truncating.")
            break
        try:
            action = policy.select_action(obs)
        except Exception as e:
            print(f"Error during {policy_name}.select_action(): {e}")
            print(f"Observation was: {obs}")
            # Potentially re-raise or break, or try a fallback action
            # For now, let's break and this episode will likely result in inf makespan.
            break 
            
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated # Standard way to check for episode end
        except Exception as e:
            print(f"Error during env.step() with action {action} from {policy_name}: {e}")
            break # Exit loop on step error

        step_count += 1

    runtime = time.time() - start_time
    makespan = info.get('makespan', float('inf'))
    
    return {"makespan": makespan, "runtime": runtime}


def main(instance_identifiers=None, num_runs_per_instance=3):
    if instance_identifiers is None:
        instance_identifiers = DEFAULT_INSTANCE_IDENTIFIERS

    if not instance_identifiers:
        print("No instance identifiers provided. Exiting evaluation.")
        return

    # SA parameters can be tuned. Using defaults from issue.
    sa_config = {
        'initial_temp': 100.0, 
        'cooling_rate': 0.95, 
        'max_iter_per_restart': 100, # Might need to be higher for good SA performance
        'num_restarts': 5,           # Might need to be higher
        'seed': 42                   # For reproducibility of SA's internal randomness
    }
    
    print("Starting evaluation script...")
    print(f"Instances to evaluate: {instance_identifiers}")
    print(f"Number of runs per instance: {num_runs_per_instance}")
    print(f"Simulated Annealing Parameters: {sa_config}")
    print("Ensure JSSEnv is installed and instance identifiers are correctly configured for your JSSEnv setup.")
    print("-" * 60)

    all_results_raw = {} # policy -> instance -> list_of_makespans, list_of_runtimes
    final_summary_metrics = {} # policy -> {avg_makespan, avg_runtime}

    # Define policy constructors
    # Env is passed at instantiation time for each instance
    policy_constructors = {
        "RandomPolicy": lambda env_instance: RandomPolicy(env_instance),
        "SPTPolicy": lambda env_instance: SPTPolicy(env_instance),
        "LWKRPolicy": lambda env_instance: LWKRPolicy(env_instance),
        "CriticalPathPolicy": lambda env_instance: CriticalPathPolicy(env_instance),
        "SimulatedAnnealingPolicy": lambda env_instance: SimulatedAnnealingPolicy(env_instance, **sa_config)
    }
    policy_names_ordered = ["RandomPolicy", "SPTPolicy", "LWKRPolicy", "CriticalPathPolicy", "SimulatedAnnealingPolicy"]


    for instance_id in instance_identifiers:
        print(f"\n--- Evaluating on Instance: {instance_id} ---")
        
        try:
            # Create environment: env_config structure depends on JSSEnv.
            # Common patterns: {'instance_path': path_to_file}, {'instance_name': name}
            env = gym.make('jss-v1', env_config={'instance_path': instance_id})
            # Or, if JSSEnv expects a name for bundled instances:
            # env = gym.make('jss-v1', env_config={'instance_name': instance_id})
        except Exception as e:
            print(f"  Error creating environment for instance '{instance_id}': {e}. Skipping.")
            # Store None or error marker for this instance if needed for table consistency
            for policy_name in policy_names_ordered:
                if policy_name not in all_results_raw: all_results_raw[policy_name] = {}
                all_results_raw[policy_name][instance_id] = {'makespans': [float('inf')], 'runtimes': [0.0]}
            continue

        for policy_name in policy_names_ordered:
            if policy_name not in all_results_raw:
                all_results_raw[policy_name] = {}
            
            all_results_raw[policy_name][instance_id] = {'makespans': [], 'runtimes': []}

            print(f"  - Running {policy_name}...")
            try:
                policy = policy_constructors[policy_name](env)
            except Exception as e:
                print(f"    Error instantiating {policy_name}: {e}. Skipping policy for this instance.")
                all_results_raw[policy_name][instance_id]['makespans'].append(float('inf'))
                all_results_raw[policy_name][instance_id]['runtimes'].append(0.0)
                continue

            for i in range(num_runs_per_instance):
                # print(f"    Run {i+1}/{num_runs_per_instance} for {policy_name} on {instance_id}...")
                # Seed for each run can be passed to run_episode if needed,
                # but SA is seeded at construction. Random/SPT use global np.random
                # unless modified to take seeds.
                run_data = run_episode(env, policy, policy_name=policy_name) # seed=i if seeding runs
                all_results_raw[policy_name][instance_id]['makespans'].append(run_data["makespan"])
                all_results_raw[policy_name][instance_id]['runtimes'].append(run_data["runtime"])
        env.close()

    # Process results for summary table
    for policy_name in policy_names_ordered:
        instance_avg_makespans = []
        instance_avg_runtimes = []
        
        for instance_id in instance_identifiers:
            if instance_id in all_results_raw.get(policy_name, {}):
                makespans = all_results_raw[policy_name][instance_id]['makespans']
                runtimes = all_results_raw[policy_name][instance_id]['runtimes']
                
                avg_m = np.mean(makespans) if makespans else float('inf')
                avg_r = np.mean(runtimes) if runtimes else float('inf')
                
                instance_avg_makespans.append(avg_m)
                instance_avg_runtimes.append(avg_r)
        
        # Overall average across instances for this policy
        overall_avg_makespan = np.mean(instance_avg_makespans) if instance_avg_makespans else float('inf')
        overall_avg_runtime = np.mean(instance_avg_runtimes) if instance_avg_runtimes else float('inf')
        
        final_summary_metrics[policy_name] = {
            "makespan": overall_avg_makespan, # Key for format_results_table
            "runtime": overall_avg_runtime   # Key for format_results_table
        }

    print("\n" + "="*60)
    print("Overall Average Performance Across All Instances:")
    print("="*60)
    print(format_results_table(policy_names_ordered, final_summary_metrics, 
                               instance_name="All Tested Instances (Average)"))

    # Optional: Print detailed per-instance average results
    print("\n" + "="*60)
    print("Average Performance Per Instance:")
    print("="*60)
    for instance_id in instance_identifiers:
        per_instance_summary = {}
        valid_instance_data = False
        for policy_name in policy_names_ordered:
            if instance_id in all_results_raw.get(policy_name, {}):
                makespans = all_results_raw[policy_name][instance_id]['makespans']
                runtimes = all_results_raw[policy_name][instance_id]['runtimes']
                avg_m = np.mean(makespans) if makespans else float('inf')
                avg_r = np.mean(runtimes) if runtimes else float('inf')
                per_instance_summary[policy_name] = {"makespan": avg_m, "runtime": avg_r}
                if avg_m != float('inf'): valid_instance_data = True
        
        if valid_instance_data:
            print(format_results_table(policy_names_ordered, per_instance_summary, instance_name=instance_id))
        else:
            print(f"No valid results to display for instance: {instance_id}\n")


if __name__ == "__main__":
    # For deterministic behavior of RandomPolicy and SPTPolicy (if they use np.random directly)
    # and for SA's initial solution generation if its internal seed is not solely relied upon.
    # np.random.seed(42) # Global seed
    
    # Example of how to run with custom instances:
    # my_instances = ["path/to/my/instance1.txt", "path/to/my/instance2.txt"]
    # main(instance_identifiers=my_instances, num_runs_per_instance=5)
    main()

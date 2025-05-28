import numpy as np

def get_current_processing_times(observation, num_jobs):
    """
    Placeholder utility function to extract current processing times for all jobs.
    
    This function assumes a specific structure for observation['real_obs']
    and may need to be adapted based on the actual JSSEnv specification.
    
    Args:
        observation (dict): The environment observation.
                            Expected: observation['real_obs']
        num_jobs (int): The total number of jobs.
        
    Returns:
        np.array: An array of processing times for the current operation of each job.
                  Returns None or raises error if structure is not as expected.
                  Jobs that are not schedulable or have finished might have a
                  conventional value like 0, inf, or a specific marker.
    """
    real_obs = observation.get("real_obs")
    if real_obs is None:
        # Or raise an error, or return a default array (e.g., all zeros/infs)
        print("Warning: 'real_obs' not found in observation.")
        return np.full(num_jobs, float('inf')) # Default to infinity if not found

    if not isinstance(real_obs, np.ndarray):
        real_obs = np.array(real_obs)

    # Assumption: real_obs[job_idx, 0] is processing time, as used in SPTPolicy
    # This part needs to be robust and match JSSEnv's actual spec.
    # This is a simplified placeholder. A real implementation would need to know
    # the exact structure of real_obs (e.g., if it's flat, or which column to use).
    
    processing_times = np.full(num_jobs, float('inf')) # Default for non-active/finished jobs

    # Example: If real_obs is (num_jobs, features) and feature 0 is proc_time
    if real_obs.ndim == 2 and real_obs.shape[0] == num_jobs and real_obs.shape[1] > 0:
        # This assumes that real_obs always pertains to the *current* operation
        # and its processing time. If a job is finished, its proc_time here
        # should reflect that (e.g., 0 or some other indicator).
        # The SPTPolicy directly uses action_mask to only consider relevant jobs.
        # This util function might just report what's in real_obs.
        for i in range(num_jobs):
            processing_times[i] = real_obs[i, 0] 
        return processing_times
    elif real_obs.ndim == 1 and real_obs.shape[0] == num_jobs:
        # Fallback: if real_obs is 1D, assume it's a list of proc times for current ops
        return real_obs.copy()
    else:
        print(f"Warning: 'real_obs' has unexpected shape {real_obs.shape}. Expected ({num_jobs}, features) or ({num_jobs},).")
        # Return default (all inf) or handle error as appropriate
        return processing_times


# Example of another utility that might be useful for evaluation:
def format_results_table(policy_names, metrics, instance_name=""):
    """
    Formats policy performance metrics into a string table.
    
    Args:
        policy_names (list of str): Names of the policies.
        metrics (dict of dicts): 
            Example: {"RandomPolicy": {"makespan": 100, "runtime": 0.1},
                      "SPTPolicy": {"makespan": 80, "runtime": 0.05}}
        instance_name (str, optional): Name of the instance being reported.
        
    Returns:
        str: A formatted string representing the results table.
    """
    if not policy_names or not metrics:
        return "No results to display."

    header = f"Results for {instance_name}:" if instance_name else "Results:"
    table_str = f"{header}\n"
    table_str += f"{'Policy':<30} | {'Makespan':<10} | {'Runtime (s)':<12}\n"
    table_str += "-" * 55 + "\n"

    for name in policy_names:
        if name in metrics:
            makespan = metrics[name].get('makespan', 'N/A')
            runtime = metrics[name].get('runtime', 'N/A')
            runtime_str = f"{runtime:.4f}" if isinstance(runtime, float) else str(runtime)
            makespan_str = f"{makespan:.2f}" if isinstance(makespan, float) else str(makespan)
            table_str += f"{name:<30} | {makespan_str:<10} | {runtime_str:<12}\n"
        else:
            table_str += f"{name:<30} | {'N/A':<10} | {'N/A':<12}\n"
            
    return table_str

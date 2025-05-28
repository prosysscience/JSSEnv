import numpy as np
from .base_policy import BaselinePolicy

class SPTPolicy(BaselinePolicy):
    """
    Shortest Processing Time (SPT) dispatching rule policy.
    Selects the legal action (job) that has the shortest processing time 
    for its current available operation.
    """
    def __init__(self, env):
        """
        Initializes the SPTPolicy.

        Args:
            env: The OpenAI Gym environment.
        """
        super().__init__(env)
        # Assuming self.env.num_jobs can give us the number of jobs if needed,
        # or observation["real_obs"] will have a consistent shape.

    def select_action(self, observation):
        """
        Selects an action based on the Shortest Processing Time rule.

        Args:
            observation: The current observation from the environment.
                         Expected format: {
                             "action_mask": np.array of shape (num_jobs,),
                             "real_obs": np.array of shape (num_jobs, num_features)
                                         where real_obs[job_idx, 0] is assumed to be
                                         the processing time of the current operation for job_idx.
                                         This assumption might need verification against
                                         the actual JSSEnv observation spec.
                         }

        Returns:
            The selected action (job index) with the shortest processing time.
            Ties are broken randomly.
        """
        action_mask = observation["action_mask"]
        real_obs = observation["real_obs"]

        # Ensure action_mask is a numpy array
        if not isinstance(action_mask, np.ndarray):
            action_mask = np.array(action_mask)
        
        # Ensure real_obs is a numpy array
        if not isinstance(real_obs, np.ndarray):
            real_obs = np.array(real_obs)

        legal_job_indices = np.where(action_mask == 1)[0]

        if not legal_job_indices.size:
            # Should not happen if episode is not done.
            # Let it raise an error if no action is possible.
            raise ValueError("No legal actions available in SPTPolicy.")

        min_processing_time = float('inf')
        candidate_actions = []

        for job_idx in legal_job_indices:
            # ASSUMPTION: real_obs[job_idx][0] is the processing time of the
            # current operation for job_idx. This is a critical assumption.
            # If JSSEnv's real_obs has a different structure, this line MUST be updated.
            # For example, if real_obs is flat, or if the processing time
            # needs to be looked up via job_op_idx from a proc_times matrix
            # also present in the observation.
            # For now, proceeding with a common convention.
            
            # Check if real_obs has enough dimensions and elements
            if real_obs.ndim == 2 and real_obs.shape[0] > job_idx and real_obs.shape[1] > 0:
                current_op_processing_time = real_obs[job_idx, 0] 
            elif real_obs.ndim == 1 and real_obs.shape[0] > job_idx: 
                # Fallback: if real_obs is 1D, assume it's a list of proc times for current ops
                # This is less standard but a possible simple obs structure.
                # Or, it could be that real_obs itself IS the processing time array.
                # The problem states "extracting processing time information from observation",
                # suggesting it's a component.
                # If real_obs is just [proc_time_job0, proc_time_job1, ...], then:
                current_op_processing_time = real_obs[job_idx]
            else:
                # This indicates an unexpected structure for real_obs or an invalid job_idx
                raise ValueError(
                    f"real_obs has unexpected shape {real_obs.shape} or job_idx {job_idx} is out of bounds. "
                    f"Cannot extract processing time for SPT."
                )

            if current_op_processing_time < min_processing_time:
                min_processing_time = current_op_processing_time
                candidate_actions = [job_idx]
            elif current_op_processing_time == min_processing_time:
                candidate_actions.append(job_idx)
        
        if not candidate_actions:
             # This could happen if all legal jobs have inf processing time (unlikely)
             # or if legal_job_indices was empty (handled above).
             # Or if real_obs structure assumption was wrong and proc times were not found.
            raise ValueError("SPTPolicy could not find any valid candidate actions.")

        # Break ties randomly
        selected_action = np.random.choice(candidate_actions)
        return selected_action

    def reset(self):
        """
        Resets the policy's internal state.
        For SPTPolicy, there is no state to reset beyond what's in the observation.
        """
        pass

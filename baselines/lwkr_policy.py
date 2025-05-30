import numpy as np
from .base_policy import BaselinePolicy

class LWKRPolicy(BaselinePolicy):
    def __init__(self, env):
        super().__init__(env)

    def select_action(self, observation):
        action_mask = observation["action_mask"]
        real_obs = observation["real_obs"]

        if not isinstance(action_mask, np.ndarray):
            action_mask = np.array(action_mask)
        if not isinstance(real_obs, np.ndarray):
            real_obs = np.array(real_obs)

        legal_job_indices = np.where(action_mask == 1)[0]

        if not legal_job_indices.size:
            raise ValueError("No legal actions available in LWKRPolicy.")

        min_remaining_work = float('inf')
        candidate_actions = []

        for job_idx in legal_job_indices:
            if job_idx >= real_obs.shape[0]:
                # This check ensures that the job_idx is valid for real_obs.
                # It might be relevant if action_mask can include indices
                # not represented in real_obs (e.g. special system actions).
                continue

            # Assuming real_obs[job_idx, 1] is total remaining work
            if real_obs.ndim != 2 or real_obs.shape[1] < 2:
                raise ValueError(
                    f"real_obs has unexpected shape {real_obs.shape}. "
                    f"Expected at least 2 columns for LWKRPolicy (current op time, total remaining work). "
                    f"Cannot extract total remaining work for job {job_idx}."
                )
            
            current_job_remaining_work = real_obs[job_idx, 1]

            if current_job_remaining_work < min_remaining_work:
                min_remaining_work = current_job_remaining_work
                candidate_actions = [job_idx]
            elif current_job_remaining_work == min_remaining_work:
                candidate_actions.append(job_idx)
        
        if not candidate_actions:
            # This could happen if all legal jobs were out of bounds for real_obs
            # or had issues with data extraction.
            # Fallback to any legal job if action_mask had entries but they couldn't be processed.
            if legal_job_indices.size > 0:
                 candidate_actions = legal_job_indices.tolist()
            else:
                # This case should ideally be caught by the first check for legal_job_indices.size
                raise ValueError("LWKRPolicy could not find any valid candidate actions from legal_job_indices.")

        selected_action = np.random.choice(candidate_actions)
        return selected_action

    def reset(self):
        pass

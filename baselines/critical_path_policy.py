import numpy as np
from .base_policy import BaselinePolicy

class CriticalPathPolicy(BaselinePolicy):
    def __init__(self, env):
        super().__init__(env)
        self.buffer_factor = 1.3

    def select_action(self, observation):
        action_mask = observation["action_mask"]
        real_obs = observation["real_obs"]

        if not isinstance(action_mask, np.ndarray):
            action_mask = np.array(action_mask)
        if not isinstance(real_obs, np.ndarray):
            real_obs = np.array(real_obs)

        legal_job_indices = np.where(action_mask == 1)[0]

        if not legal_job_indices.size:
            raise ValueError("No legal actions available in CriticalPathPolicy.")

        min_critical_ratio = float('inf')
        candidate_actions = []

        for job_idx in legal_job_indices:
            if job_idx >= real_obs.shape[0]:
                # This check ensures that the job_idx is valid for real_obs.
                continue

            if real_obs.ndim != 2 or real_obs.shape[1] < 2:
                raise ValueError(
                    f"real_obs has unexpected shape {real_obs.shape}. "
                    f"Expected at least 2 columns for CriticalPathPolicy (current op time, total remaining work). "
                    f"Cannot extract data for job {job_idx}."
                )
            
            remaining_work = real_obs[job_idx, 1]
            current_op_processing_time = real_obs[job_idx, 0]

            if current_op_processing_time == 0:
                # Revised logic:
                # If current_op_processing_time is 0:
                #   - If remaining_work > 0, this job is blocked or has a zero-duration task
                #     that doesn't consume time on the current machine but is part of the path.
                #     It should be deprioritized by this rule, as CR focuses on time consumption.
                #   - If remaining_work == 0, this job is done.
                # In both cases, assign a high ratio to deprioritize.
                critical_ratio = float('inf')
            else:
                # remaining_work includes current_op_processing_time for the current operation
                # plus all subsequent operations. The critical path rule often looks at
                # total remaining work / current operation processing time.
                # The budget factor is applied to the total remaining work.
                remaining_time_budget = remaining_work * self.buffer_factor
                critical_ratio = remaining_time_budget / current_op_processing_time
            
            if critical_ratio < min_critical_ratio:
                min_critical_ratio = critical_ratio
                candidate_actions = [job_idx]
            elif critical_ratio == min_critical_ratio:
                candidate_actions.append(job_idx)
        
        if not candidate_actions:
            # This could happen if all legal jobs had current_op_processing_time = 0,
            # or were out of bounds for real_obs, or had issues with data extraction.
            # Fallback to any legal job if action_mask had entries but they couldn't be processed
            # by the primary logic.
            if legal_job_indices.size > 0:
                 candidate_actions = legal_job_indices.tolist()
            else:
                # This case should ideally be caught by the first check for legal_job_indices.size
                raise ValueError("CriticalPathPolicy could not find any valid candidate actions from legal_job_indices.")

        selected_action = np.random.choice(candidate_actions)
        return selected_action

    def reset(self):
        pass

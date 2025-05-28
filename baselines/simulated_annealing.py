import numpy as np
import copy
from .base_policy import BaselinePolicy

class SimulatedAnnealingPolicy(BaselinePolicy):
    """
    A policy that uses Simulated Annealing to find a good job permutation (schedule)
    and then follows this permutation to select actions.
    """
    def __init__(self, env, initial_temp=100.0, cooling_rate=0.95, 
                 max_iter_per_restart=100, num_restarts=5, seed=None):
        """
        Initializes the SimulatedAnnealingPolicy.

        Args:
            env: The OpenAI Gym environment.
            initial_temp (float): Initial temperature for SA.
            cooling_rate (float): Cooling rate for SA (e.g., 0.95 for exponential cooling).
            max_iter_per_restart (int): Number of iterations per SA restart.
            num_restarts (int): Number of SA restarts.
            seed (int, optional): Seed for numpy random number generator.
        """
        super().__init__(env)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter_per_restart = max_iter_per_restart
        self.num_restarts = num_restarts
        
        self.np_random = np.random.RandomState(seed)
        
        # Assuming env.action_space.n gives the number of jobs.
        # This needs to be true for JSSEnv.
        # If not, this needs to be obtained differently (e.g., from obs shape).
        if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
            self.num_jobs = env.action_space.n
        else:
            # Fallback: try to infer from an initial observation if possible,
            # though this is less robust. For now, raise an error.
            # obs_space = env.observation_space
            # if isinstance(obs_space, gym.spaces.Dict) and 'action_mask' in obs_space.spaces:
            #    self.num_jobs = obs_space.spaces['action_mask'].shape[0]
            # else:
            raise ValueError("Cannot determine num_jobs from environment. Ensure env.action_space.n is available.")

        self.best_job_permutation = None
        self.is_optimized = False # Flag to check if SA optimization has run for the current episode

    def _generate_initial_solution(self):
        """Generates a random permutation of job indices."""
        return self.np_random.permutation(self.num_jobs)

    def _get_neighbor(self, current_solution):
        """Generates a neighbor solution by swapping two jobs in the permutation."""
        neighbor = current_solution.copy()
        idx1, idx2 = self.np_random.choice(self.num_jobs, 2, replace=False)
        neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        return neighbor

    def _evaluate_schedule_makespan(self, job_permutation):
        """
        Evaluates a job permutation by simulating it in a copy of the environment
        and returning the makespan.
        """
        temp_env = copy.deepcopy(self.env) # Critical: use a deep copy
        obs, _ = temp_env.reset() # Reset the copied environment
        done = False
        total_reward = 0

        while not done:
            action_mask = obs["action_mask"]
            if not isinstance(action_mask, np.ndarray): # Ensure numpy array
                action_mask = np.array(action_mask)

            selected_action = -1
            for job_idx in job_permutation:
                if action_mask[job_idx] == 1:
                    selected_action = job_idx
                    break
            
            if selected_action == -1:
                # If permutation doesn't yield a valid action, pick any valid one
                # This can happen if the permutation is poor or action_mask is restrictive
                legal_actions = np.where(action_mask == 1)[0]
                if legal_actions.size > 0:
                    selected_action = self.np_random.choice(legal_actions)
                else:
                    # No legal actions available, episode should be ending or error
                    # print("Warning: No legal action found during SA schedule evaluation.")
                    break 
            
            obs, reward, done, truncated, info = temp_env.step(selected_action)
            done = done or truncated # Consider truncated as done for makespan calculation
            total_reward += reward # Not directly used for makespan but good to track

        # Makespan is typically available in info at the end of an episode
        makespan = info.get('makespan', float('inf')) 
        # If makespan is not in info, this environment is not suitable or
        # the episode didn't terminate correctly for makespan calculation.
        # For JSSEnv, 'makespan' is expected.
        return makespan

    def run_sa_optimization(self):
        """
        Runs the main Simulated Annealing optimization algorithm to find the best job permutation.
        """
        overall_best_permutation = None
        overall_best_makespan = float('inf')

        for _ in range(self.num_restarts):
            current_solution = self._generate_initial_solution()
            current_makespan = self._evaluate_schedule_makespan(current_solution)
            
            best_solution_this_restart = current_solution
            best_makespan_this_restart = current_makespan
            
            temp = self.initial_temp

            for _ in range(self.max_iter_per_restart):
                neighbor_solution = self._get_neighbor(current_solution)
                neighbor_makespan = self._evaluate_schedule_makespan(neighbor_solution)

                if neighbor_makespan < current_makespan:
                    acceptance_probability = 1.0
                else:
                    if temp > 1e-6: # Avoid division by zero or underflow
                        acceptance_probability = np.exp((current_makespan - neighbor_makespan) / temp)
                    else:
                        acceptance_probability = 0.0
                
                if self.np_random.rand() < acceptance_probability:
                    current_solution = neighbor_solution
                    current_makespan = neighbor_makespan

                if current_makespan < best_makespan_this_restart:
                    best_solution_this_restart = current_solution
                    best_makespan_this_restart = current_makespan
                
                temp *= self.cooling_rate
            
            if best_makespan_this_restart < overall_best_makespan:
                overall_best_makespan = best_makespan_this_restart
                overall_best_permutation = best_solution_this_restart
        
        self.best_job_permutation = overall_best_permutation
        self.is_optimized = True
        # print(f"SA optimization complete. Best makespan: {overall_best_makespan}, Permutation: {self.best_job_permutation}")


    def select_action(self, observation):
        """
        Selects an action based on the pre-optimized job permutation.
        If optimization hasn't run for this episode, it runs it first.
        """
        if not self.is_optimized:
            # print("Running SA optimization for the first time this episode...")
            self.run_sa_optimization()

        action_mask = observation["action_mask"]
        if not isinstance(action_mask, np.ndarray):
            action_mask = np.array(action_mask)

        if self.best_job_permutation is None:
            # Fallback if SA failed to find any solution (should not happen ideally)
            # print("Warning: SA did not produce a job permutation. Falling back to random.")
            legal_actions = np.where(action_mask == 1)[0]
            return self.np_random.choice(legal_actions) if legal_actions.size > 0 else 0 # Default to 0 if no legal actions

        for job_idx in self.best_job_permutation:
            if action_mask[job_idx] == 1:
                return job_idx
        
        # Fallback: if the preferred permutation doesn't offer a currently legal action
        # (e.g., all high-priority jobs are blocked). Pick any legal one.
        # This ensures the policy always returns a valid action if one exists.
        # print("Warning: SA permutation did not yield a valid action. Picking from available.")
        legal_actions = np.where(action_mask == 1)[0]
        if legal_actions.size > 0:
            return self.np_random.choice(legal_actions)
        else:
            # This case implies no legal actions are available at all.
            # The environment should handle this (e.g., episode ends).
            # Or, if it's an error state, an error should be raised by the env.
            # Returning a default or raising an error here depends on env spec.
            # For now, if this happens, it indicates an issue.
            # Let's assume action_mask will not be all zeros if not done.
            # If it can, then env behavior for env.step(??) needs to be defined.
            # This path implies the episode should have ended or there's an error.
            # For robustness, if an action is *required*, one might return a dummy (e.g., 0)
            # but this is risky if 0 is not always a valid masked action.
            # The environment should ideally not call select_action if no actions are possible.
            raise ValueError("SA Policy: No legal actions available based on action mask, and permutation fallback failed.")


    def reset(self):
        """
        Resets the policy's state, indicating that SA optimization needs to be re-run
        for the new episode.
        """
        # print("SA Policy Reset: Optimization will run on next select_action or if called directly.")
        self.best_job_permutation = None
        self.is_optimized = False
        # Potentially re-run SA optimization here if preferred,
        # but doing it lazily on first select_action is also fine.
        # self.run_sa_optimization() # Uncomment if SA should run on env.reset()

import bisect
import datetime
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import gymnasium as gym
import numpy as np
import plotly.figure_factory as ff
from plotly.graph_objects import Figure


class JssEnv(gym.Env):
    """
    Job Shop Scheduling Environment.
    
    This environment models the job shop scheduling problem as a single agent problem:

    - The actions correspond to a job allocation plus one action for no allocation at this time step (NOPE action)
    - We keep track of time with next possible time steps
    - Each time we allocate a job, the end time of the job is added to the stack of time steps
    - If we don't have a legal action (i.e., we can't allocate a job),
      we automatically go to the next time step until we have a legal action
    """
    
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Job Shop Scheduling environment.
        
        Args:
            env_config: Dictionary of configuration parameters. 
                        Can contain 'instance_path' key with path to problem instance file.
        """
        if env_config is None:
            env_config = {
                "instance_path": Path(__file__).parent.absolute() / "instances" / "ta80"
            }
        instance_path = env_config["instance_path"]

        # initial values for variables used for instance
        self.jobs = 0
        self.machines = 0
        self.instance_matrix = None
        self.jobs_length = None
        self.max_time_op = 0
        self.max_time_jobs = 0
        self.nb_legal_actions = 0
        self.nb_machine_legal = 0
        # initial values for variables used for solving (to reinitialize when reset() is called)
        self.solution = None
        self.last_solution = None
        self.last_time_step = float("inf")
        self.current_time_step = float("inf")
        self.next_time_step = list()
        self.next_jobs = list()
        self.legal_actions = None
        self.time_until_available_machine = None
        self.time_until_finish_current_op_jobs = None
        self.todo_time_step_job = None
        self.total_perform_op_time_jobs = None
        self.needed_machine_jobs = None
        self.total_idle_time_jobs = None
        self.idle_time_jobs_last_op = None
        self.state = None
        self.illegal_actions = None
        self.action_illegal_no_op = None
        self.machine_legal = None
        # initial values for variables used for representation
        self.start_timestamp = datetime.datetime.now().timestamp()
        self.sum_op = 0
        with open(instance_path, "r") as instance_file:
            for line_cnt, line_str in enumerate(instance_file, start=1):
                split_data = list(map(int, line_str.split()))

                if line_cnt == 1:
                    self.jobs, self.machines = split_data
                    self.instance_matrix = np.zeros((self.jobs, self.machines), dtype=(int, 2))
                    self.jobs_length = np.zeros(self.jobs, dtype=int)
                else:
                    assert len(split_data) % 2 == 0 and len(split_data) // 2 == self.machines
                    job_nb = line_cnt - 2
                    for i in range(0, len(split_data), 2):
                        machine, time = split_data[i], split_data[i + 1]
                        self.instance_matrix[job_nb][i // 2] = (machine, time)
                        self.max_time_op = max(self.max_time_op, time)
                        self.jobs_length[job_nb] += time
                        self.sum_op += time
        self.max_time_jobs = max(self.jobs_length)
        # check the parsed data are correct
        assert self.max_time_op > 0
        assert self.max_time_jobs > 0
        assert self.jobs > 0
        assert self.machines > 1, "We need at least 2 machines"
        assert self.instance_matrix is not None
        # allocate a job + one to wait
        self.action_space = gym.spaces.Discrete(self.jobs + 1)
        # used for plotting
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.machines)
        ]
        """
        matrix with the following attributes for each job:
            -Legal job
            -Left over time on the current op
            -Current operation %
            -Total left over time
            -When next machine available
            -Time since IDLE: 0 if not available, time otherwise
            -Total IDLE time in the schedule
        """
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.Box(0, 1, shape=(self.jobs + 1,)),
                "real_obs": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.jobs, 7), dtype=float
                ),
            }
        )

    def _get_current_state_representation(self) -> Dict[str, np.ndarray]:
        """
        Get the current state representation as a dictionary.
        
        Returns:
            Dict containing state representation with keys:
                - 'real_obs': Normalized state matrix for each job
                - 'action_mask': Boolean mask of legal actions
        """
        self.state[:, 0] = self.legal_actions[:-1]
        return {
            "real_obs": self.state,
            "action_mask": self.legal_actions,
        }

    def get_legal_actions(self) -> np.ndarray:
        """
        Get the mask of legal actions.
        
        Returns:
            Boolean array where True indicates a legal action
        """
        return self.legal_actions

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment to an initial state.
        
        Returns:
            Initial observation as a dictionary with keys:
                - 'real_obs': Normalized state matrix for each job
                - 'action_mask': Boolean mask of legal actions
        """
        self.current_time_step = 0
        self.next_time_step = list()
        self.next_jobs = list()
        self.nb_legal_actions = self.jobs
        self.nb_machine_legal = 0
        # represent all the legal actions
        self.legal_actions = np.ones(self.jobs + 1, dtype=bool)
        self.legal_actions[self.jobs] = False
        # used to represent the solution
        self.solution = np.full((self.jobs, self.machines), -1, dtype=int)
        self.time_until_available_machine = np.zeros(self.machines, dtype=int)
        self.time_until_finish_current_op_jobs = np.zeros(self.jobs, dtype=int)
        self.todo_time_step_job = np.zeros(self.jobs, dtype=int)
        self.total_perform_op_time_jobs = np.zeros(self.jobs, dtype=int)
        self.needed_machine_jobs = np.zeros(self.jobs, dtype=int)
        self.total_idle_time_jobs = np.zeros(self.jobs, dtype=int)
        self.idle_time_jobs_last_op = np.zeros(self.jobs, dtype=int)
        self.illegal_actions = np.zeros((self.machines, self.jobs), dtype=bool)
        self.action_illegal_no_op = np.zeros(self.jobs, dtype=bool)
        self.machine_legal = np.zeros(self.machines, dtype=bool)
        for job in range(self.jobs):
            needed_machine = self.instance_matrix[job][0][0]
            self.needed_machine_jobs[job] = needed_machine
            if not self.machine_legal[needed_machine]:
                self.machine_legal[needed_machine] = True
                self.nb_machine_legal += 1
        self.state = np.zeros((self.jobs, 7), dtype=float)
        return self._get_current_state_representation()

    def _prioritization_non_final(self) -> None:
        """
        Prioritize non-final operations over final operations based on processing times.
        
        This method implements a heuristic that makes non-final operations higher priority
        than final operations when they require the same machine and the final operation
        takes longer to complete.
        
        The optimization logic works as follows:
        1. Identify all final operations (last operation of a job) and non-final operations 
           (operations with subsequent operations) that compete for the same machine
        2. For non-final operations, find the minimum processing time among them
        3. If a final operation takes longer than this minimum time, make it illegal
           to prioritize the faster non-final operations
        
        This improves overall makespan by prioritizing operations that will unlock
        subsequent operations sooner, leading to better resource utilization.
        """
        # Only proceed if there's at least one legal machine available
        if self.nb_machine_legal >= 1:
            for machine in range(self.machines):
                if self.machine_legal[machine]:
                    # Separate jobs needing this machine into final and non-final operations
                    final_job = list()  # Jobs on their last operation
                    non_final_job = list()  # Jobs with operations remaining after this one
                    min_non_final = float("inf")  # Track minimum processing time of non-final operations
                    
                    # Categorize all legal jobs that need this machine
                    for job in range(self.jobs):
                        if (
                            self.needed_machine_jobs[job] == machine
                            and self.legal_actions[job]
                        ):
                            # Check if this is the job's final operation
                            if self.todo_time_step_job[job] == (self.machines - 1):
                                final_job.append(job)
                            else:
                                # For non-final operations, calculate processing time
                                current_time_step_non_final = self.todo_time_step_job[job]
                                time_needed_legal = self.instance_matrix[job][
                                    current_time_step_non_final
                                ][1]
                                
                                # Get the machine needed for the next operation of this job
                                machine_needed_nextstep = self.instance_matrix[job][
                                    current_time_step_non_final + 1
                                ][0]
                                
                                # Only consider this job for prioritization if its next machine is available
                                # This is critical for maximizing throughput - we only want to prioritize
                                # jobs that can immediately continue to their next operation
                                if (
                                    self.time_until_available_machine[machine_needed_nextstep] == 0
                                ):
                                    # Keep track of the minimum processing time among non-final operations
                                    min_non_final = min(min_non_final, time_needed_legal)
                                    non_final_job.append(job)
                    
                    # OPTIMIZATION: If we have non-final operations, prioritize them over longer final operations
                    # This improves makespan by focusing on operations that will unlock subsequent work
                    if len(non_final_job) > 0:
                        for job in final_job:
                            current_time_step_final = self.todo_time_step_job[job]
                            time_needed_legal = self.instance_matrix[job][
                                current_time_step_final
                            ][1]
                            
                            # If this final operation takes longer than the fastest non-final operation,
                            # make it illegal to encourage scheduling the faster non-final operation first
                            if time_needed_legal > min_non_final:
                                self.legal_actions[job] = False
                                self.nb_legal_actions -= 1

    def _check_no_op(self) -> None:
        """
        Determine if a no-operation action (waiting) is legal.
        
        This method checks if it's beneficial to wait for the next event rather than
        allocating a job now. It sets self.legal_actions[self.jobs] to True if waiting
        is a good decision based on future resource availability and waiting jobs.
        
        The optimization logic works as follows:
        1. Start by assuming waiting is not beneficial (no-op action is illegal)
        2. Check if there are future events (next_time_step) and only a small number of
           legal actions and machines (to limit the computational burden of this check)
        3. Calculate time horizons for each machine based on current jobs
        4. Look at jobs that are currently illegal but will become legal in the future
        5. Determine if all currently legal machines could be better utilized by waiting
           for these future jobs
        6. If all legal machines would be needed by future jobs, make waiting legal
        
        This heuristic helps avoid scheduling jobs that might block machines needed for
        more critical operations that will become available soon.
        """
        # Start by assuming the no-op action (waiting) is illegal
        self.legal_actions[self.jobs] = False
        
        # Only consider the no-op action if:
        # 1. There are future events in the schedule
        # 2. There are only a few legal machines (computation optimization)
        # 3. There are only a few legal actions (computation optimization)
        if (
            len(self.next_time_step) > 0
            and self.nb_machine_legal <= 3
            and self.nb_legal_actions <= 4
        ):
            # Track machines that will be needed by jobs that will become legal in the future
            machine_next = set()
            
            # Get the time of the next event (when a machine will become available)
            next_time_step = self.next_time_step[0]
            
            # Initialize time horizons
            max_horizon = self.current_time_step
            
            # For each machine, initialize the max time horizon as current time + max operation time
            # This represents the latest time we would expect each machine to be occupied if scheduled now
            max_horizon_machine = [
                self.current_time_step + self.max_time_op for _ in range(self.machines)
            ]
            
            # First pass: look at currently legal jobs to calculate machine horizons
            for job in range(self.jobs):
                if self.legal_actions[job]:
                    time_step = self.todo_time_step_job[job]
                    machine_needed = self.instance_matrix[job][time_step][0]
                    time_needed = self.instance_matrix[job][time_step][1]
                    end_job = self.current_time_step + time_needed
                    
                    # If any job would finish before the next event, it's better to schedule it
                    # than to wait - so exit the function (keeping no-op illegal)
                    if end_job < next_time_step:
                        return
                    
                    # Update the time horizon for this machine based on this job
                    max_horizon_machine[machine_needed] = min(
                        max_horizon_machine[machine_needed], end_job
                    )
                    max_horizon = max(max_horizon, max_horizon_machine[machine_needed])
            
            # Second pass: analyze jobs that are currently illegal but will become legal soon
            for job in range(self.jobs):
                if not self.legal_actions[job]:
                    # Case 1: Job is currently running on a machine and will have next operations
                    if (
                        self.time_until_finish_current_op_jobs[job] > 0
                        and self.todo_time_step_job[job] + 1 < self.machines
                    ):
                        # Look at the next operation of this job (after current one completes)
                        time_step = self.todo_time_step_job[job] + 1
                        # Calculate when this next operation could start
                        time_needed = (
                            self.current_time_step
                            + self.time_until_finish_current_op_jobs[job]
                        )
                        
                        # Check all subsequent operations of this job
                        while (
                            time_step < self.machines - 1 and max_horizon > time_needed
                        ):
                            machine_needed = self.instance_matrix[job][time_step][0]
                            
                            # Check if this machine would be better utilized by waiting for this job
                            if (
                                max_horizon_machine[machine_needed] > time_needed
                                and self.machine_legal[machine_needed]
                            ):
                                # This is a machine that would be better used if we wait
                                machine_next.add(machine_needed)
                                
                                # KEY OPTIMIZATION: If all currently legal machines would be better 
                                # used by waiting for future jobs, then make waiting legal
                                # This is critical for optimizing the makespan by preventing
                                # suboptimal early scheduling decisions
                                if len(machine_next) == self.nb_machine_legal:
                                    self.legal_actions[self.jobs] = True
                                    return
                                    
                            # Move to the next operation of this job
                            time_needed += self.instance_matrix[job][time_step][1]
                            time_step += 1
                            
                    # Case 2: Job is waiting for a machine to become available
                    elif (
                        not self.action_illegal_no_op[job]
                        and self.todo_time_step_job[job] < self.machines
                    ):
                        time_step = self.todo_time_step_job[job]
                        machine_needed = self.instance_matrix[job][time_step][0]
                        
                        # Calculate when this job's operation could start
                        time_needed = (
                            self.current_time_step
                            + self.time_until_available_machine[machine_needed]
                        )
                        
                        # Check all operations of this job
                        while (
                            time_step < self.machines - 1 and max_horizon > time_needed
                        ):
                            machine_needed = self.instance_matrix[job][time_step][0]
                            
                            # Check if this machine would be better utilized by waiting for this job
                            if (
                                max_horizon_machine[machine_needed] > time_needed
                                and self.machine_legal[machine_needed]
                            ):
                                # This is a machine that would be better used if we wait
                                machine_next.add(machine_needed)
                                
                                # KEY OPTIMIZATION: If all currently legal machines would be better 
                                # used by waiting for future jobs, then make waiting legal
                                if len(machine_next) == self.nb_machine_legal:
                                    self.legal_actions[self.jobs] = True
                                    return
                                    
                            # Move to the next operation of this job
                            time_needed += self.instance_matrix[job][time_step][1]
                            time_step += 1

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Take an action in the environment and observe the next state.
        
        Args:
            action: Index of the action to take (job to allocate or no-op)
            
        Returns:
            Tuple containing:
            - Observation: Dictionary with state representation and action mask
            - Reward: Scalar reward value
            - Done: Boolean indicating if the episode is complete
            - Truncated: Boolean indicating if the episode was truncated (always False in this env)
            - Info: Empty dictionary (reserved for future use)
        """
        reward = 0.0
        if action == self.jobs:  # No-op action
            self.nb_machine_legal = 0
            self.nb_legal_actions = 0
            for job in range(self.jobs):
                if self.legal_actions[job]:
                    self.legal_actions[job] = False
                    needed_machine = self.needed_machine_jobs[job]
                    self.machine_legal[needed_machine] = False
                    self.illegal_actions[needed_machine][job] = True
                    self.action_illegal_no_op[job] = True
            while self.nb_machine_legal == 0:
                reward -= self.increase_time_step()
            scaled_reward = self._reward_scaler(reward)
            self._prioritization_non_final()
            self._check_no_op()
            return (
                self._get_current_state_representation(),
                scaled_reward,
                self._is_done(),
                False,  # truncated flag (always False in this environment)
                {},
            )
        else:  # Job allocation action
            current_time_step_job = self.todo_time_step_job[action]
            machine_needed = self.needed_machine_jobs[action]
            time_needed = self.instance_matrix[action][current_time_step_job][1]
            reward += time_needed
            self.time_until_available_machine[machine_needed] = time_needed
            self.time_until_finish_current_op_jobs[action] = time_needed
            self.state[action][1] = time_needed / self.max_time_op
            to_add_time_step = self.current_time_step + time_needed
            if to_add_time_step not in self.next_time_step:
                index = bisect.bisect_left(self.next_time_step, to_add_time_step)
                self.next_time_step.insert(index, to_add_time_step)
                self.next_jobs.insert(index, action)
            self.solution[action][current_time_step_job] = self.current_time_step
            for job in range(self.jobs):
                if (
                    self.needed_machine_jobs[job] == machine_needed
                    and self.legal_actions[job]
                ):
                    self.legal_actions[job] = False
                    self.nb_legal_actions -= 1
            self.nb_machine_legal -= 1
            self.machine_legal[machine_needed] = False
            for job in range(self.jobs):
                if self.illegal_actions[machine_needed][job]:
                    self.action_illegal_no_op[job] = False
                    self.illegal_actions[machine_needed][job] = False
            # if we can't allocate new job in the current timestep, we pass to the next one
            while self.nb_machine_legal == 0 and len(self.next_time_step) > 0:
                reward -= self.increase_time_step()
            self._prioritization_non_final()
            self._check_no_op()
            # we then need to scale the reward
            scaled_reward = self._reward_scaler(reward)
            return (
                self._get_current_state_representation(),
                scaled_reward,
                self._is_done(),
                False,  # truncated flag (always False in this environment)
                {},
            )

    def _reward_scaler(self, reward: float) -> float:
        """
        Scale the raw reward to a normalized range.
        
        Args:
            reward: Raw reward value
            
        Returns:
            Scaled reward value normalized by maximum operation time
        """
        return reward / self.max_time_op

    def increase_time_step(self) -> int:
        """
        Advance the simulation time to the next event and update all state variables.
        
        This is the core logic of the environment. It updates time counters, job status,
        machine availability, and legal actions when time advances.
        
        The method handles several key aspects:
        1. Time advancement to the next event
        2. Updating job progress and completion status
        3. Calculating idle time for jobs and machines
        4. Determining which jobs become legal for scheduling after the time update
        5. Tracking machine availability
        6. Calculating "holes" in the schedule (machine idle time)
        
        Returns:
            Time elapsed since last time step that represents idle time in the schedule
        """
        # Track idle time in the schedule (when machines are empty but could be working)
        hole_planning = 0
        
        # Get and remove the next time event from the queue
        next_time_step_to_pick = self.next_time_step.pop(0)
        self.next_jobs.pop(0)  # Also remove the corresponding job
        
        # Calculate the time difference since last step
        difference = next_time_step_to_pick - self.current_time_step
        self.current_time_step = next_time_step_to_pick
        
        # Update all jobs
        for job in range(self.jobs):
            was_left_time = self.time_until_finish_current_op_jobs[job]
            
            # Case 1: Job is currently being processed on a machine
            if was_left_time > 0:
                # Calculate how much of the operation was performed during this time step
                performed_op_job = min(difference, was_left_time)
                
                # Update remaining time for the current operation
                self.time_until_finish_current_op_jobs[job] = max(
                    0, self.time_until_finish_current_op_jobs[job] - difference
                )
                
                # Update the state representation for this job
                self.state[job][1] = (
                    self.time_until_finish_current_op_jobs[job] / self.max_time_op
                )
                
                # Track total operation time for this job
                self.total_perform_op_time_jobs[job] += performed_op_job
                self.state[job][3] = (
                    self.total_perform_op_time_jobs[job] / self.max_time_jobs
                )
                
                # If the current operation just finished
                if self.time_until_finish_current_op_jobs[job] == 0:
                    # Calculate and track idle time
                    self.total_idle_time_jobs[job] += difference - was_left_time
                    self.state[job][6] = self.total_idle_time_jobs[job] / self.sum_op
                    self.idle_time_jobs_last_op[job] = difference - was_left_time
                    self.state[job][5] = self.idle_time_jobs_last_op[job] / self.sum_op
                    
                    # Move to the next operation for this job
                    self.todo_time_step_job[job] += 1
                    self.state[job][2] = self.todo_time_step_job[job] / self.machines
                    
                    # If there are more operations for this job
                    if self.todo_time_step_job[job] < self.machines:
                        # Get the next machine needed for this job
                        self.needed_machine_jobs[job] = self.instance_matrix[job][
                            self.todo_time_step_job[job]
                        ][0]
                        
                        # Calculate wait time until the needed machine is available
                        self.state[job][4] = (
                            max(
                                0,
                                self.time_until_available_machine[
                                    self.needed_machine_jobs[job]
                                ]
                                - difference,
                            )
                            / self.max_time_op
                        )
                    else:
                        # Job is completed (all operations done)
                        self.needed_machine_jobs[job] = -1
                        
                        # Set state value to 1.0 to indicate job is complete
                        # This ensures it won't appear as a good candidate for scheduling
                        # (0.0 would incorrectly suggest machine is immediately available)
                        self.state[job][4] = 1.0
                        
                        # If the job was legal, make it illegal since it's now complete
                        if self.legal_actions[job]:
                            self.legal_actions[job] = False
                            self.nb_legal_actions -= 1
            
            # Case 2: Job is waiting (not currently being processed)
            elif self.todo_time_step_job[job] < self.machines:
                # Update idle time for the job
                self.total_idle_time_jobs[job] += difference
                self.idle_time_jobs_last_op[job] += difference
                
                # Update state representation
                self.state[job][5] = self.idle_time_jobs_last_op[job] / self.sum_op
                self.state[job][6] = self.total_idle_time_jobs[job] / self.sum_op
        
        # Update all machines
        for machine in range(self.machines):
            # Calculate idle time for this machine (if it became available during this time step)
            if self.time_until_available_machine[machine] < difference:
                empty = difference - self.time_until_available_machine[machine]
                hole_planning += empty  # This tracks scheduling inefficiency
            
            # Update the time until this machine is available
            self.time_until_available_machine[machine] = max(
                0, self.time_until_available_machine[machine] - difference
            )
            
            # If the machine is now available, update job legality
            if self.time_until_available_machine[machine] == 0:
                for job in range(self.jobs):
                    # OPTIMIZATION: Make jobs legal if:
                    # 1. They need this machine for their next operation
                    # 2. They are not already legal
                    # 3. They aren't specially marked as illegal for this machine
                    if (
                        self.needed_machine_jobs[job] == machine
                        and not self.legal_actions[job]
                        and not self.illegal_actions[machine][job]
                    ):
                        # Make the job legal
                        self.legal_actions[job] = True
                        self.nb_legal_actions += 1
                        
                        # If needed, also mark the machine as legal
                        if not self.machine_legal[machine]:
                            self.machine_legal[machine] = True
                            self.nb_machine_legal += 1
        
        # Return the total idle time in the schedule during this time step
        return hole_planning

    def _is_done(self) -> bool:
        """
        Check if the episode is complete.
        
        The episode is considered done when there are no more legal actions available,
        which means all jobs have been scheduled completely.
        
        Returns:
            Boolean indicating if the episode is complete
        """
        if self.nb_legal_actions == 0:
            self.last_time_step = self.current_time_step
            self.last_solution = self.solution
            return True
        return False

    def render(self, mode: str = "human") -> Optional[Figure]:
        """
        Render the current solution as a Gantt chart visualization.
        
        Args:
            mode: Rendering mode (only 'human' is supported)
            
        Returns:
            Plotly Figure object containing a Gantt chart visualization of the schedule,
            or None if no operations have been scheduled yet
        """
        df = []
        for job in range(self.jobs):
            i = 0
            while i < self.machines and self.solution[job][i] != -1:
                dict_op = dict()
                dict_op["Task"] = f"Job {job}"
                start_sec = self.start_timestamp + self.solution[job][i]
                finish_sec = start_sec + self.instance_matrix[job][i][1]
                dict_op["Start"] = datetime.datetime.fromtimestamp(start_sec)
                dict_op["Finish"] = datetime.datetime.fromtimestamp(finish_sec)
                dict_op["Resource"] = f"Machine {self.instance_matrix[job][i][0]}"
                df.append(dict_op)
                i += 1
        
        fig = None
        if len(df) > 0:
            df = pd.DataFrame(df)
            fig = ff.create_gantt(
                df,
                index_col="Resource",
                colors=self.colors,
                show_colorbar=True,
                group_tasks=True,
            )
            # Otherwise tasks are listed from the bottom up
            fig.update_yaxes(autorange="reversed")
        
        return fig


if __name__ == '__main__':
    # Simple example of environment usage
    env = JssEnv()
    obs = env.reset()
    done = False
    cum_reward = 0
    
    # Run until all jobs are scheduled
    while not done:
        # Get legal actions from observation
        legal_actions = obs["action_mask"]
        
        # Choose a random legal action
        action = np.random.choice(
            len(legal_actions), 1, p=(legal_actions / legal_actions.sum())
        )[0]
        
        # Take action in environment
        obs, reward, done, truncated, _ = env.step(action)
        cum_reward += reward
        
    print(f"Cumulative reward: {cum_reward}")


"""
Dispatching rules for the Job Shop Scheduling environment.

This module implements common dispatching rules used in Job Shop Scheduling:
- Shortest Processing Time (SPT)
- Earliest Due Date (EDD)
- First-In-First-Out (FIFO)
- Most Work Remaining (MWR)
- Least Work Remaining (LWR)
- Most Operations Remaining (MOR)
- Least Operations Remaining (LOR)
- Critical Ratio (CR)
"""

from typing import Callable, Dict, List, Tuple, Union, Any, Optional
import numpy as np

from JSSEnv.envs.jss_env import JssEnv


class DispatchingRule:
    """Base class for all dispatching rules."""
    
    def __init__(self, name: str, description: str):
        """
        Initialize a dispatching rule.
        
        Args:
            name: Short name of the rule (e.g., 'SPT')
            description: Longer description of what the rule does
        """
        self.name = name
        self.description = description
    
    def __call__(self, env: JssEnv) -> int:
        """
        Apply the rule to select an action in the given environment state.
        
        Args:
            env: The JSS environment
            
        Returns:
            The selected action (job index or no-op)
        """
        raise NotImplementedError("Subclasses must implement __call__")
    
    def get_name(self) -> str:
        """Get the name of the rule."""
        return self.name
    
    def get_description(self) -> str:
        """Get the description of the rule."""
        return self.description
    
    def run_episode(self, env: JssEnv) -> Tuple[float, int]:
        """
        Run a complete episode using this dispatching rule.
        
        Args:
            env: The JSS environment
            
        Returns:
            Tuple of (total_reward, makespan)
        """
        obs = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            action = self(env)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
        makespan = env.current_time_step
        return total_reward, makespan


class ShortestProcessingTime(DispatchingRule):
    """
    Shortest Processing Time (SPT) rule.
    
    Selects the job with the shortest processing time for its current operation.
    This is a greedy, myopic rule that minimizes the immediate completion time.
    """
    
    def __init__(self):
        super().__init__(
            name="SPT",
            description="Shortest Processing Time: Schedule the job with the shortest processing time next"
        )
    
    def __call__(self, env: JssEnv) -> int:
        legal_actions = env.get_legal_actions()
        
        # If waiting (no-op) is the only legal action
        if np.sum(legal_actions) == 1 and legal_actions[-1]:
            return env.jobs  # Return the no-op action
        
        min_time = float('inf')
        min_job = -1
        
        # Find the job with the shortest processing time among legal actions
        for job in range(env.jobs):
            if legal_actions[job]:
                current_op = env.todo_time_step_job[job]
                process_time = env.instance_matrix[job][current_op][1]
                
                if process_time < min_time:
                    min_time = process_time
                    min_job = job
        
        # If there's a tie and waiting is legal, prefer waiting
        if legal_actions[env.jobs] and np.random.random() < 0.1:  # Small chance to wait for exploration
            return env.jobs
        
        return min_job


class FirstInFirstOut(DispatchingRule):
    """
    First In First Out (FIFO) rule.
    
    Selects the job that has been waiting the longest.
    This rule is fair in terms of waiting time but may not optimize for makespan.
    """
    
    def __init__(self):
        super().__init__(
            name="FIFO",
            description="First In First Out: Schedule the job that has been waiting the longest"
        )
    
    def __call__(self, env: JssEnv) -> int:
        legal_actions = env.get_legal_actions()
        
        # If waiting (no-op) is the only legal action
        if np.sum(legal_actions) == 1 and legal_actions[-1]:
            return env.jobs  # Return the no-op action
        
        max_idle_time = -1
        max_job = -1
        
        # Find the job with the longest idle time
        for job in range(env.jobs):
            if legal_actions[job]:
                idle_time = env.idle_time_jobs_last_op[job]
                
                if idle_time > max_idle_time:
                    max_idle_time = idle_time
                    max_job = job
        
        # If there's a tie and waiting is legal, prefer waiting
        if legal_actions[env.jobs] and np.random.random() < 0.1:  # Small chance to wait for exploration
            return env.jobs
        
        return max_job


class MostWorkRemaining(DispatchingRule):
    """
    Most Work Remaining (MWR) rule.
    
    Selects the job with the most total processing time remaining.
    This rule prioritizes jobs with the most work to help them finish earlier.
    """
    
    def __init__(self):
        super().__init__(
            name="MWR",
            description="Most Work Remaining: Schedule the job with the most processing time remaining"
        )
    
    def __call__(self, env: JssEnv) -> int:
        legal_actions = env.get_legal_actions()
        
        # If waiting (no-op) is the only legal action
        if np.sum(legal_actions) == 1 and legal_actions[-1]:
            return env.jobs  # Return the no-op action
        
        max_remaining_time = -1
        max_job = -1
        
        # Find the job with the most remaining work
        for job in range(env.jobs):
            if legal_actions[job]:
                # Calculate remaining work for this job
                remaining_time = 0
                for op in range(env.todo_time_step_job[job], env.machines):
                    remaining_time += env.instance_matrix[job][op][1]
                
                if remaining_time > max_remaining_time:
                    max_remaining_time = remaining_time
                    max_job = job
        
        # If there's a tie and waiting is legal, prefer waiting
        if legal_actions[env.jobs] and np.random.random() < 0.1:  # Small chance to wait for exploration
            return env.jobs
        
        return max_job


class LeastWorkRemaining(DispatchingRule):
    """
    Least Work Remaining (LWR) rule.
    
    Selects the job with the least total processing time remaining.
    This rule prioritizes jobs that are close to completion.
    """
    
    def __init__(self):
        super().__init__(
            name="LWR",
            description="Least Work Remaining: Schedule the job with the least processing time remaining"
        )
    
    def __call__(self, env: JssEnv) -> int:
        legal_actions = env.get_legal_actions()
        
        # If waiting (no-op) is the only legal action
        if np.sum(legal_actions) == 1 and legal_actions[-1]:
            return env.jobs  # Return the no-op action
        
        min_remaining_time = float('inf')
        min_job = -1
        
        # Find the job with the least remaining work
        for job in range(env.jobs):
            if legal_actions[job]:
                # Calculate remaining work for this job
                remaining_time = 0
                for op in range(env.todo_time_step_job[job], env.machines):
                    remaining_time += env.instance_matrix[job][op][1]
                
                if remaining_time < min_remaining_time:
                    min_remaining_time = remaining_time
                    min_job = job
        
        # If there's a tie and waiting is legal, prefer waiting
        if legal_actions[env.jobs] and np.random.random() < 0.1:  # Small chance to wait for exploration
            return env.jobs
        
        return min_job


class MostOperationsRemaining(DispatchingRule):
    """
    Most Operations Remaining (MOR) rule.
    
    Selects the job with the most operations remaining.
    This rule aims to start jobs with many operations early.
    """
    
    def __init__(self):
        super().__init__(
            name="MOR",
            description="Most Operations Remaining: Schedule the job with the most operations remaining"
        )
    
    def __call__(self, env: JssEnv) -> int:
        legal_actions = env.get_legal_actions()
        
        # If waiting (no-op) is the only legal action
        if np.sum(legal_actions) == 1 and legal_actions[-1]:
            return env.jobs  # Return the no-op action
        
        max_remaining_ops = -1
        max_job = -1
        
        # Find the job with the most remaining operations
        for job in range(env.jobs):
            if legal_actions[job]:
                # Calculate remaining operations
                remaining_ops = env.machines - env.todo_time_step_job[job]
                
                if remaining_ops > max_remaining_ops:
                    max_remaining_ops = remaining_ops
                    max_job = job
        
        # If there's a tie and waiting is legal, prefer waiting
        if legal_actions[env.jobs] and np.random.random() < 0.1:  # Small chance to wait for exploration
            return env.jobs
        
        return max_job


class LeastOperationsRemaining(DispatchingRule):
    """
    Least Operations Remaining (LOR) rule.
    
    Selects the job with the fewest operations remaining.
    This rule aims to complete jobs that are close to finishing.
    """
    
    def __init__(self):
        super().__init__(
            name="LOR",
            description="Least Operations Remaining: Schedule the job with the fewest operations remaining"
        )
    
    def __call__(self, env: JssEnv) -> int:
        legal_actions = env.get_legal_actions()
        
        # If waiting (no-op) is the only legal action
        if np.sum(legal_actions) == 1 and legal_actions[-1]:
            return env.jobs  # Return the no-op action
        
        min_remaining_ops = float('inf')
        min_job = -1
        
        # Find the job with the fewest remaining operations
        for job in range(env.jobs):
            if legal_actions[job]:
                # Calculate remaining operations
                remaining_ops = env.machines - env.todo_time_step_job[job]
                
                if remaining_ops < min_remaining_ops:
                    min_remaining_ops = remaining_ops
                    min_job = job
        
        # If there's a tie and waiting is legal, prefer waiting
        if legal_actions[env.jobs] and np.random.random() < 0.1:  # Small chance to wait for exploration
            return env.jobs
        
        return min_job


class CriticalRatio(DispatchingRule):
    """
    Critical Ratio (CR) rule.
    
    Selects the job based on the ratio of time remaining to work remaining.
    Jobs with a lower ratio (less slack time) are prioritized.
    
    Note: In this implementation, we use the makespan as a proxy for due dates.
    """
    
    def __init__(self, due_date_factor: float = 1.5):
        """
        Initialize the Critical Ratio rule.
        
        Args:
            due_date_factor: Factor to multiply the maximum possible makespan to get a due date
        """
        super().__init__(
            name="CR",
            description="Critical Ratio: Schedule based on the ratio of time to due date versus remaining work"
        )
        self.due_date_factor = due_date_factor
        self._due_dates = {}  # Cache for due dates by job
    
    def _calculate_due_date(self, env: JssEnv, job: int) -> float:
        """Calculate a due date for the job based on total processing time."""
        if job in self._due_dates:
            return self._due_dates[job]
        
        # Get total processing time for the job
        total_time = sum(env.instance_matrix[job][op][1] for op in range(env.machines))
        
        # Due date is a factor of the total processing time
        due_date = total_time * self.due_date_factor
        self._due_dates[job] = due_date
        
        return due_date
    
    def __call__(self, env: JssEnv) -> int:
        legal_actions = env.get_legal_actions()
        
        # If waiting (no-op) is the only legal action
        if np.sum(legal_actions) == 1 and legal_actions[-1]:
            return env.jobs  # Return the no-op action
        
        # Reset due dates cache at the start of an episode
        if env.current_time_step == 0:
            self._due_dates = {}
        
        min_ratio = float('inf')
        min_job = -1
        
        # Find the job with the lowest critical ratio
        for job in range(env.jobs):
            if legal_actions[job]:
                # Calculate due date for this job
                due_date = self._calculate_due_date(env, job)
                
                # Calculate remaining work
                remaining_time = 0
                for op in range(env.todo_time_step_job[job], env.machines):
                    remaining_time += env.instance_matrix[job][op][1]
                
                # Time until due date
                time_remaining = due_date - env.current_time_step
                
                # Calculate critical ratio (time remaining / work remaining)
                # Lower ratio means less slack time relative to work
                if remaining_time > 0:
                    ratio = time_remaining / remaining_time
                else:
                    ratio = float('inf')  # Job is almost done
                
                if ratio < min_ratio:
                    min_ratio = ratio
                    min_job = job
        
        # If there's a tie and waiting is legal, prefer waiting
        if legal_actions[env.jobs] and np.random.random() < 0.1:  # Small chance to wait for exploration
            return env.jobs
        
        return min_job


# Dictionary of all available dispatching rules
DISPATCHING_RULES = {
    "SPT": ShortestProcessingTime(),
    "FIFO": FirstInFirstOut(),
    "MWR": MostWorkRemaining(),
    "LWR": LeastWorkRemaining(),
    "MOR": MostOperationsRemaining(),
    "LOR": LeastOperationsRemaining(),
    "CR": CriticalRatio(),
}


def get_rule(rule_name: str) -> DispatchingRule:
    """
    Get a dispatching rule by name.
    
    Args:
        rule_name: Name of the rule to get
        
    Returns:
        The dispatching rule
        
    Raises:
        ValueError: If the rule doesn't exist
    """
    if rule_name not in DISPATCHING_RULES:
        raise ValueError(f"Rule '{rule_name}' not found. Available rules: {list(DISPATCHING_RULES.keys())}")
    
    return DISPATCHING_RULES[rule_name]


def compare_rules(env: JssEnv, rules: Optional[List[str]] = None, 
                 num_episodes: int = 10) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple dispatching rules on the same environment.
    
    Args:
        env: The environment to test on
        rules: List of rule names to compare (if None, use all rules)
        num_episodes: Number of episodes to run for each rule
        
    Returns:
        Dictionary with results for each rule (average reward and makespan)
    """
    if rules is None:
        rules = list(DISPATCHING_RULES.keys())
    
    results = {}
    
    for rule_name in rules:
        rule = get_rule(rule_name)
        total_reward = 0.0
        total_makespan = 0.0
        
        for _ in range(num_episodes):
            reward, makespan = rule.run_episode(env)
            total_reward += reward
            total_makespan += makespan
        
        results[rule_name] = {
            "avg_reward": total_reward / num_episodes,
            "avg_makespan": total_makespan / num_episodes
        }
    
    return results
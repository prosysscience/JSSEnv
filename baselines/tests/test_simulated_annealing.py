import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from baselines.simulated_annealing import SimulatedAnnealingPolicy
from baselines.tests.mock_env import MockJSSEnv

class TestSimulatedAnnealingPolicy(unittest.TestCase):
    def setUp(self):
        self.num_jobs = 4
        self.mock_env = MockJSSEnv(num_jobs=self.num_jobs, num_features=1)
        self.sa_params = {
            'initial_temp': 10.0, # Lower temp for faster testing
            'cooling_rate': 0.8, # Faster cooling
            'max_iter_per_restart': 5, # Fewer iterations
            'num_restarts': 1, # Fewer restarts
            'seed': 42
        }
        self.policy = SimulatedAnnealingPolicy(self.mock_env, **self.sa_params)

    def test_initialization(self):
        """Test SA policy initializes with parameters."""
        self.assertEqual(self.policy.initial_temp, self.sa_params['initial_temp'])
        self.assertEqual(self.policy.cooling_rate, self.sa_params['cooling_rate'])
        self.assertEqual(self.policy.max_iter_per_restart, self.sa_params['max_iter_per_restart'])
        self.assertEqual(self.policy.num_restarts, self.sa_params['num_restarts'])
        self.assertEqual(self.policy.num_jobs, self.num_jobs)
        self.assertIsNotNone(self.policy.np_random)
        self.assertIsNone(self.policy.best_job_permutation)
        self.assertFalse(self.policy.is_optimized)

    def test_generate_initial_solution(self):
        """Test generation of an initial random job permutation."""
        solution = self.policy._generate_initial_solution()
        self.assertIsInstance(solution, np.ndarray)
        self.assertEqual(len(solution), self.num_jobs)
        self.assertEqual(len(set(solution)), self.num_jobs, "Solution should have unique job indices.")
        for job_idx in solution:
            self.assertIn(job_idx, range(self.num_jobs))

    def test_get_neighbor(self):
        """Test generation of a neighbor solution."""
        initial_solution = np.arange(self.num_jobs)
        neighbor = self.policy._get_neighbor(initial_solution.copy())
        
        self.assertIsInstance(neighbor, np.ndarray)
        self.assertEqual(len(neighbor), self.num_jobs)
        self.assertEqual(len(set(neighbor)), self.num_jobs)
        # Check that it's a permutation of the original jobs
        self.assertTrue(np.array_equal(np.sort(initial_solution), np.sort(neighbor)))
        # Check that it's different (highly probable for num_jobs > 1)
        if self.num_jobs > 1:
            self.assertFalse(np.array_equal(initial_solution, neighbor), 
                             "Neighbor should typically be different from the initial solution.")

    @patch.object(SimulatedAnnealingPolicy, '_evaluate_schedule_makespan')
    def test_run_sa_optimization(self, mock_evaluate_schedule):
        """Test the main SA optimization loop runs and sets a permutation."""
        # Mock the evaluation to return predictable makespans
        # Let initial solution be better initially, then a neighbor.
        mock_evaluate_schedule.side_effect = [100, 80, 90, 70] # Example makespans

        self.policy.run_sa_optimization()
        
        self.assertTrue(self.policy.is_optimized)
        self.assertIsNotNone(self.policy.best_job_permutation)
        self.assertEqual(len(self.policy.best_job_permutation), self.num_jobs)
        # Check that _evaluate_schedule_makespan was called multiple times
        self.assertTrue(mock_evaluate_schedule.call_count >= self.sa_params['num_restarts'] * (1 + self.sa_params['max_iter_per_restart']))


    def test_select_action_triggers_optimization_first_time(self):
        """Test select_action calls run_sa_optimization if not optimized."""
        # Mock the actual optimization process to avoid long runtimes in this unit test
        # We only want to check if it's called.
        self.policy.run_sa_optimization = MagicMock() 
        
        action_mask = np.array([1, 1, 0, 0])
        real_obs = np.zeros((self.num_jobs, 1))
        self.mock_env.set_observation_parts(action_mask, real_obs)
        observation = self.mock_env.get_current_observation()

        self.assertFalse(self.policy.is_optimized)
        self.policy.select_action(observation)
        
        self.policy.run_sa_optimization.assert_called_once()
        self.assertTrue(self.policy.is_optimized) # run_sa_optimization should set this

    def test_select_action_uses_permutation(self):
        """Test select_action follows the optimized permutation."""
        # Pre-set an optimized permutation
        self.policy.best_job_permutation = np.array([2, 1, 0, 3]) # Job 2 is highest priority
        self.policy.is_optimized = True
        
        # Case 1: Highest priority job (2) is legal
        action_mask = np.array([1, 1, 1, 1]) # All legal
        real_obs = np.zeros((self.num_jobs, 1))
        self.mock_env.set_observation_parts(action_mask, real_obs)
        observation = self.mock_env.get_current_observation()
        action = self.policy.select_action(observation)
        self.assertEqual(action, 2)

        # Case 2: Job 2 is not legal, Job 1 (next in perm) is legal
        action_mask = np.array([1, 1, 0, 1]) # Job 2 illegal, Job 1 legal
        self.mock_env.set_observation_parts(action_mask, real_obs)
        observation = self.mock_env.get_current_observation()
        action = self.policy.select_action(observation)
        self.assertEqual(action, 1)

        # Case 3: Job 2, 1, 0 are not legal, Job 3 (last in perm) is legal
        action_mask = np.array([0, 0, 0, 1]) # Only Job 3 legal
        self.mock_env.set_observation_parts(action_mask, real_obs)
        observation = self.mock_env.get_current_observation()
        action = self.policy.select_action(observation)
        self.assertEqual(action, 3)


    def test_select_action_fallback_if_permutation_all_illegal(self):
        """Test fallback if permutation jobs are all illegal (should pick any legal)."""
        self.policy.best_job_permutation = np.array([0, 1, 2, 3]) # J0, J1, J2, J3 preference
        self.policy.is_optimized = True

        # Permutation jobs [0,1] are illegal, but job 3 (not high in perm) is legal
        # This tests the fallback: "if job_idx in self.best_job_permutation ... else fallback"
        # The current SA logic iterates through its permutation. If none in the permutation
        # are in the action_mask, it falls back to choosing randomly from legal actions.
        action_mask = np.array([0, 0, 1, 1]) # Jobs 2 and 3 are legal. Permutation prefers 0 then 1.
        real_obs = np.zeros((self.num_jobs, 1))
        self.mock_env.set_observation_parts(action_mask, real_obs)
        observation = self.mock_env.get_current_observation()
        
        action = self.policy.select_action(observation)
        # Permutation is [0,1,2,3]. Mask is [0,0,1,1].
        # Policy will check 0 (illegal), 1 (illegal), 2 (legal!) -> returns 2
        # This test case name is slightly misleading for current implementation.
        # It will pick 2. The fallback is if *NO* job in permutation is legal.
        self.assertEqual(action, 2) 

        # Actual Fallback Test: No job in permutation is legal
        self.policy.best_job_permutation = np.array([0, 1]) # Only jobs 0, 1 in permutation for some reason (e.g. error)
                                                           # Or, num_jobs was smaller when SA ran.
                                                           # To make it robust, let's assume full permutation.
        self.policy.best_job_permutation = np.array([0, 1, 2, 3])
        action_mask = np.array([0,0,0,0]) # No legal actions at all
        if self.num_jobs == 2: # Adjust if num_jobs changed for a specific test
             action_mask = np.array([0,0])
        else:
            action_mask = np.array([0] * self.num_jobs)

        self.mock_env.set_observation_parts(action_mask, real_obs)
        observation = self.mock_env.get_current_observation()
        with self.assertRaisesRegex(ValueError, "SA Policy: No legal actions available"):
            self.policy.select_action(observation)


    def test_reset_clears_optimization_flag_and_permutation(self):
        """Test reset method clears optimization state."""
        self.policy.best_job_permutation = np.array([0, 1, 2, 3])
        self.policy.is_optimized = True
        
        self.policy.reset()
        
        self.assertIsNone(self.policy.best_job_permutation)
        self.assertFalse(self.policy.is_optimized)

    def test_evaluate_schedule_makespan_integration_with_mock_env(self):
        """
        Test _evaluate_schedule_makespan with the mock environment.
        This is a limited integration test for that specific (critical) method.
        """
        test_permutation = np.array([0, 1, 2, 3])
        expected_makespan = 123
        
        # Configure mock_env to return specific makespan after fixed steps
        self.mock_env.set_evaluation_behavior(
            makespan_to_return=expected_makespan, 
            max_steps_in_eval=self.num_jobs # SA eval loop will run num_jobs times
        )
        
        # Reset the env state for evaluation (action mask, etc.)
        self.mock_env.set_observation_parts(
            action_mask=np.ones(self.num_jobs), 
            real_obs=np.zeros((self.num_jobs,1))
        )
        
        # Need to use a fresh env copy like the method does, or ensure mock_env resets properly for eval
        # The _evaluate_schedule_makespan uses copy.deepcopy(self.env)
        # So we need to patch that or ensure our self.mock_env behaves as a deepcopy would.
        
        # For this test, we can trust that self.mock_env is what deepcopy would return,
        # as we are controlling it directly.
        
        # The method itself uses a deepcopy. To test it directly, we'd need to
        # ensure the deepcopy call returns our configured mock_env, or
        # we make self.policy.env the already configured mock_env.
        # The latter is simpler for this test.
        
        # Ensure the environment used by the policy is our configured mock
        self.policy.env = self.mock_env 
                                     
        returned_makespan = self.policy._evaluate_schedule_makespan(test_permutation)
        
        self.assertEqual(returned_makespan, expected_makespan)
        # Check that the mock env's step was called roughly num_jobs times
        self.assertEqual(self.mock_env.steps_taken_in_eval, self.num_jobs)


if __name__ == '__main__':
    unittest.main()

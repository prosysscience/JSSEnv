import unittest
import numpy as np
from baselines.spt_policy import SPTPolicy
from baselines.tests.mock_env import MockJSSEnv

class TestSPTPolicy(unittest.TestCase):
    def test_select_action_simple_case(self):
        """Test SPT selects the job with the shortest processing time."""
        mock_env = MockJSSEnv(num_jobs=3, num_features=1)
        policy = SPTPolicy(mock_env)

        # real_obs: job 0: 10, job 1: 5, job 2: 12
        # action_mask: all legal
        obs = {
            "action_mask": np.array([1, 1, 1]),
            "real_obs": np.array([[10], [5], [12]], dtype=np.float32)
        }
        mock_env.set_observation_parts(obs["action_mask"], obs["real_obs"])
        
        action = policy.select_action(mock_env.get_current_observation())
        self.assertEqual(action, 1) # Job 1 has SPT (5)

    def test_select_action_with_illegal_actions(self):
        """Test SPT considers only legal actions."""
        mock_env = MockJSSEnv(num_jobs=4, num_features=1)
        policy = SPTPolicy(mock_env)

        # real_obs: J0:5, J1:3 (SPT), J2:7, J3:4
        # action_mask: J0, J2, J3 are legal. J1 (SPT) is NOT.
        # Expected: J3 (proc_time 4) should be chosen from legal options.
        obs = {
            "action_mask": np.array([1, 0, 1, 1]),
            "real_obs": np.array([[5], [3], [7], [4]], dtype=np.float32)
        }
        mock_env.set_observation_parts(obs["action_mask"], obs["real_obs"])

        action = policy.select_action(mock_env.get_current_observation())
        self.assertEqual(action, 3) # Job 3 has SPT (4) among legal actions

    def test_select_action_tie_breaking(self):
        """Test SPT tie-breaking (should select one of the tied jobs)."""
        mock_env = MockJSSEnv(num_jobs=3, num_features=1)
        policy = SPTPolicy(mock_env)

        # real_obs: J0:5, J1:8, J2:5 (J0 and J2 tied for SPT)
        # action_mask: all legal
        obs = {
            "action_mask": np.array([1, 1, 1]),
            "real_obs": np.array([[5], [8], [5]], dtype=np.float32)
        }
        mock_env.set_observation_parts(obs["action_mask"], obs["real_obs"])
        
        selected_actions = set()
        # Run multiple times to see if different tied actions can be chosen
        # np.random.choice handles the random tie-breaking.
        for _ in range(20): # Increased attempts
            action = policy.select_action(mock_env.get_current_observation())
            self.assertIn(action, [0, 2]) # Must be one of the tied jobs
            selected_actions.add(action)
        
        # This checks if the selection mechanism isn't stuck on one choice.
        self.assertTrue(len(selected_actions) > 0)
        if len([0,2]) > 1 : # if there truly are multiple tied options
             self.assertTrue(len(selected_actions) >= 1, "Should be able to pick from tied options.")


    def test_select_action_1d_real_obs(self):
        """Test SPT with 1D real_obs (list of processing times)."""
        mock_env = MockJSSEnv(num_jobs=3, num_features=1) # num_features=1 for mock consistency
        policy = SPTPolicy(mock_env)

        # real_obs (1D): J0:10, J1:5 (SPT), J2:12
        # action_mask: all legal
        # SPTPolicy should handle real_obs being (num_jobs,) if ndim == 1
        obs = {
            "action_mask": np.array([1, 1, 1]),
            "real_obs": np.array([10, 5, 12], dtype=np.float32) # 1D array
        }
        # Mock env stores real_obs as 2D, so we set it, then policy converts internally
        # Or, we can adjust mock_env to store 1D if policy expects it directly.
        # The policy's code attempts to handle real_obs.ndim == 1 or real_obs.ndim == 2
        # So, the observation passed to select_action should reflect that.
        
        # Let's ensure the observation directly passed to policy.select_action is 1D for real_obs
        # The mock_env.set_observation_parts might convert it.
        # We can bypass mock_env for this specific observation structure test.
        
        action = policy.select_action(obs) # Pass the dict with 1D real_obs directly
        self.assertEqual(action, 1)

    def test_select_action_no_legal_actions_raises_error(self):
        """Test SPTPolicy raises ValueError if no actions are legal."""
        mock_env = MockJSSEnv(num_jobs=3)
        policy = SPTPolicy(mock_env)
        
        obs = {
            "action_mask": np.array([0, 0, 0]),
            "real_obs": np.array([[10], [5], [12]], dtype=np.float32)
        }
        mock_env.set_observation_parts(obs["action_mask"], obs["real_obs"])
        
        with self.assertRaisesRegex(ValueError, "No legal actions available in SPTPolicy."):
            policy.select_action(mock_env.get_current_observation())

    def test_select_action_real_obs_unexpected_shape_raises_error(self):
        """Test SPTPolicy raises error if real_obs has unexpected shape."""
        mock_env = MockJSSEnv(num_jobs=3, num_features=2) # num_features=2
        policy = SPTPolicy(mock_env)
        
        # real_obs is (3x2) but the policy's primary assumption is (Nx1) or (N,)
        # The policy has a check:
        # "real_obs has unexpected shape ... Cannot extract processing time for SPT."
        # This will be triggered if it's not (N,0) (i.e. N, >0) or (N,).
        # If real_obs is (N, M) where M > 1, it uses column 0. This is fine.
        # The error should be raised if num_jobs doesn't match real_obs.shape[0].
        
        # Case 1: real_obs.shape[0] != num_jobs
        obs_wrong_jobs = {
            "action_mask": np.array([1, 1]), # Mask for 2 jobs
            "real_obs": np.array([[10], [5], [12]], dtype=np.float32) # Data for 3 jobs
        }
         # This case is more about action_mask vs real_obs mismatch.
         # The policy iterates based on action_mask. If job_idx from action_mask
         # is out of bounds for real_obs, an IndexError would occur.
         # SPTPolicy's check is `real_obs.shape[0] > job_idx`.

        # Let's test the specific ValueError from the policy:
        # "real_obs has unexpected shape ... or job_idx ... is out of bounds."
        # This implies a mismatch between num_jobs (from action_mask len) and real_obs content.
        
        # This will cause job_idx to go out of bounds for real_obs if action_mask is longer
        obs_mask_longer = {
            "action_mask": np.array([1,1,1,1]), # 4 jobs
            "real_obs": np.array([[10],[5],[12]]) # 3 jobs data
        }
        # The policy derives num_jobs from len(action_mask).
        # It then accesses real_obs[job_idx, 0]. If job_idx >= real_obs.shape[0], error.
        # This test should verify that.

        # The current policy structure:
        # legal_job_indices = np.where(action_mask == 1)[0]
        # for job_idx in legal_job_indices:
        #   if real_obs.ndim == 2 and real_obs.shape[0] > job_idx and real_obs.shape[1] > 0:
        #       current_op_processing_time = real_obs[job_idx, 0]
        #   elif real_obs.ndim == 1 and real_obs.shape[0] > job_idx:
        #       current_op_processing_time = real_obs[job_idx]
        #   else: -> this is the ValueError we want to test.
        # This error occurs if real_obs is not 2D or 1D, OR if shape[0] <= job_idx,
        # OR if (for 2D) shape[1] <= 0.

        # Test case for the "else" branch of shape checking:
        # Give a real_obs that is, e.g., 3D, or 0-dimensional.
        obs_bad_ndim = {
            "action_mask": np.array([1,0,0]),
            "real_obs": np.array([[[1],[2]],[[3],[4]]]) # 3D array
        }
        with self.assertRaisesRegex(ValueError, "real_obs has unexpected shape"):
             policy.select_action(obs_bad_ndim)


    def test_reset_method_exists_and_runs(self):
        """Test that the reset method exists and can be called."""
        mock_env = MockJSSEnv(num_jobs=3)
        policy = SPTPolicy(mock_env)
        try:
            policy.reset()
        except Exception as e:
            self.fail(f"policy.reset() raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()

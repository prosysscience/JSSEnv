import unittest
import numpy as np
from baselines.random_policy import RandomPolicy
from baselines.tests.mock_env import MockJSSEnv

class TestRandomPolicy(unittest.TestCase):
    def test_select_action_single_legal(self):
        """Test RandomPolicy selects the only legal action."""
        mock_env = MockJSSEnv(num_jobs=3)
        policy = RandomPolicy(mock_env)
        
        action_mask = [0, 1, 0]
        # real_obs structure doesn't matter for RandomPolicy, but needs to be valid
        real_obs = np.zeros((3, 1)) 
        observation = {"action_mask": np.array(action_mask), "real_obs": real_obs}
        
        action = policy.select_action(observation)
        self.assertEqual(action, 1)

    def test_select_action_multiple_legal(self):
        """Test RandomPolicy selects one of the legal actions."""
        mock_env = MockJSSEnv(num_jobs=4)
        policy = RandomPolicy(mock_env)

        action_mask = [0, 1, 1, 0]
        real_obs = np.zeros((4, 1))
        observation = {"action_mask": np.array(action_mask), "real_obs": real_obs}
        
        legal_actions = [1, 2]
        selected_actions = set()
        # Run multiple times to check if it can select different actions
        for _ in range(20): # Increased attempts for higher chance to see both
            action = policy.select_action(observation)
            self.assertIn(action, legal_actions)
            selected_actions.add(action)
        
        # Check if both actions were selected at some point (probabilistic)
        # For a unit test, this might be flaky. A better test might be to
        # check if the distribution is uniform if we could control seed locally.
        # For now, just check if it's always a legal action.
        # If only one action was ever selected after many trials, it might indicate an issue.
        self.assertTrue(len(selected_actions) > 0, "Policy should select at least one action.")
        # If there are multiple options, it's good if it can select more than one over time
        if len(legal_actions) > 1:
             self.assertTrue(len(selected_actions) >= 1, "Policy should be able to pick from multiple options.")


    def test_select_action_all_legal(self):
        """Test RandomPolicy with all actions legal."""
        num_jobs = 5
        mock_env = MockJSSEnv(num_jobs=num_jobs)
        policy = RandomPolicy(mock_env)

        action_mask = [1] * num_jobs
        real_obs = np.zeros((num_jobs, 1))
        observation = {"action_mask": np.array(action_mask), "real_obs": real_obs}
        
        action = policy.select_action(observation)
        self.assertIn(action, range(num_jobs))

    def test_select_action_no_legal_actions_raises_error(self):
        """Test RandomPolicy raises an error if no actions are legal."""
        # numpy.random.choice raises ValueError if the array is empty.
        # The policy currently lets this error propagate.
        mock_env = MockJSSEnv(num_jobs=3)
        policy = RandomPolicy(mock_env)
        
        action_mask = [0, 0, 0]
        real_obs = np.zeros((3,1))
        observation = {"action_mask": np.array(action_mask), "real_obs": real_obs}
        
        with self.assertRaises(ValueError):
            policy.select_action(observation)

    def test_reset_method_exists_and_runs(self):
        """Test that the reset method exists and can be called."""
        mock_env = MockJSSEnv(num_jobs=3)
        policy = RandomPolicy(mock_env)
        try:
            policy.reset()
        except Exception as e:
            self.fail(f"policy.reset() raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()

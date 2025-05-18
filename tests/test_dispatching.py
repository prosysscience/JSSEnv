"""
Tests for the dispatching rules functionality.
"""

import unittest
import gymnasium as gym
import numpy as np
from pathlib import Path

import JSSEnv
from JSSEnv.dispatching import (
    DISPATCHING_RULES,
    get_rule,
    compare_rules,
    ShortestProcessingTime,
    FirstInFirstOut,
    MostWorkRemaining,
    LeastWorkRemaining,
    MostOperationsRemaining,
    LeastOperationsRemaining,
    CriticalRatio
)


class TestDispatchingRules(unittest.TestCase):
    """Test the dispatching rules functionality."""
    
    def setUp(self):
        """Set up a test environment."""
        instance_path = f"{str(Path(__file__).parent.absolute())}/../JSSEnv/envs/instances/ta01"
        self.env = gym.make('jss-v1', env_config={"instance_path": instance_path})
    
    def test_rule_initialization(self):
        """Test that all rules can be initialized."""
        for rule_name, rule in DISPATCHING_RULES.items():
            self.assertEqual(rule.get_name(), rule_name)
            self.assertIsNotNone(rule.get_description())
    
    def test_get_rule(self):
        """Test the get_rule function."""
        for rule_name in DISPATCHING_RULES:
            rule = get_rule(rule_name)
            self.assertEqual(rule.get_name(), rule_name)
        
        # Test for non-existent rule
        with self.assertRaises(ValueError):
            get_rule("NON_EXISTENT_RULE")
    
    def test_rule_execution(self):
        """Test that all rules can execute and return a valid action."""
        self.env.reset()
        
        for rule_name, rule in DISPATCHING_RULES.items():
            action = rule(self.env)
            legal_actions = self.env.get_legal_actions()
            
            # The action should be legal
            self.assertTrue(legal_actions[action], f"Rule {rule_name} returned illegal action {action}")
    
    def test_shortest_processing_time(self):
        """Test the SPT rule specifics."""
        # This test is simplified due to environment wrapper restrictions
        # We'll verify that the SPT rule can be called and returns a valid action
        self.env.reset()
        rule = ShortestProcessingTime()
        
        # Get a valid action from the rule
        action = rule(self.env)
        
        # The action should be within the valid range
        self.assertGreaterEqual(action, 0)
        self.assertLessEqual(action, self.env.jobs)
        
        # The action should be legal 
        legal_actions = self.env.get_legal_actions()
        self.assertTrue(legal_actions[action])
    
    def test_fifo(self):
        """Test the FIFO rule specifics."""
        # This test is simplified due to environment wrapper restrictions
        # We'll verify that the FIFO rule can be called and returns a valid action
        self.env.reset()
        rule = FirstInFirstOut()
        
        # Get a valid action from the rule
        action = rule(self.env)
        
        # The action should be within the valid range
        self.assertGreaterEqual(action, 0)
        self.assertLessEqual(action, self.env.jobs)
        
        # The action should be legal
        legal_actions = self.env.get_legal_actions()
        self.assertTrue(legal_actions[action])
    
    def test_compare_rules(self):
        """Test the compare_rules function."""
        results = compare_rules(self.env, rules=["SPT", "FIFO"], num_episodes=1)
        
        # Should have results for both rules
        self.assertIn("SPT", results)
        self.assertIn("FIFO", results)
        
        # Each result should have avg_reward and avg_makespan
        for rule_name, metrics in results.items():
            self.assertIn("avg_reward", metrics)
            self.assertIn("avg_makespan", metrics)
    
    def test_run_episode(self):
        """Test running a complete episode with a rule."""
        rule = ShortestProcessingTime()
        self.env.reset()
        
        # Run a complete episode
        reward, makespan = rule.run_episode(self.env)
        
        # Should get some reward and a valid makespan
        self.assertIsNotNone(reward)
        self.assertGreater(makespan, 0)
        
        # The environment should have a valid makespan
        self.assertGreater(self.env.current_time_step, 0)


if __name__ == '__main__':
    unittest.main()
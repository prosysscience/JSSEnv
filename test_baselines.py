#!/usr/bin/env python3
"""
Quick Test Script for Baseline Policies

This script performs basic functionality tests on all baseline policies
to ensure they can be instantiated and run without errors.
"""

import sys
import os
import traceback

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from baselines.tests.mock_env import MockJSSEnv
    from baselines import RandomPolicy, SPTPolicy, SimulatedAnnealingPolicy
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

import numpy as np

def test_policy(policy_class, policy_name, env, **kwargs):
    """Test a single policy."""
    print(f"\n🧪 Testing {policy_name}")
    print("-" * 30)
    
    try:
        # Create policy
        print("  📝 Creating policy...", end=" ")
        policy = policy_class(env, **kwargs)
        print("✅")
        
        # Test reset method
        print("  🔄 Testing reset method...", end=" ")
        policy.reset()
        print("✅")
        
        # Test select_action with different scenarios
        test_cases = [
            {
                'name': 'Single legal action',
                'action_mask': [0, 1, 0],
                'real_obs': [[0.0], [5.0], [0.0]],
                'expected_action': 1
            },
            {
                'name': 'Multiple legal actions',
                'action_mask': [1, 1, 0],
                'real_obs': [[3.0], [5.0], [0.0]],
                'expected_actions': [0, 1]
            },
            {
                'name': 'All actions legal',
                'action_mask': [1, 1, 1],
                'real_obs': [[3.0], [5.0], [2.0]],
                'expected_actions': [0, 1, 2]
            }
        ]
        
        for test_case in test_cases:
            print(f"  🎯 Testing: {test_case['name']}...", end=" ")
            
            observation = {
                'action_mask': np.array(test_case['action_mask']),
                'real_obs': np.array(test_case['real_obs'])
            }
            
            action = policy.select_action(observation)
            
            if 'expected_action' in test_case:
                if action == test_case['expected_action']:
                    print("✅")
                else:
                    print(f"❌ Expected {test_case['expected_action']}, got {action}")
                    return False
            elif 'expected_actions' in test_case:
                if action in test_case['expected_actions']:
                    print("✅")
                else:
                    print(f"❌ Expected one of {test_case['expected_actions']}, got {action}")
                    return False
        
        # Test error case (no legal actions)
        print("  ⚠️  Testing no legal actions...", end=" ")
        try:
            observation = {
                'action_mask': np.array([0, 0, 0]),
                'real_obs': np.array([[0.0], [0.0], [0.0]])
            }
            action = policy.select_action(observation)
            print("❌ Should have raised an error")
            return False
        except (ValueError, IndexError):
            print("✅ (correctly raised error)")
        except Exception as e:
            print(f"❌ Unexpected error type: {e}")
            return False
        
        print(f"  🎉 {policy_name} passed all tests!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing {policy_name}: {e}")
        traceback.print_exc()
        return False

def test_simulated_annealing_specific():
    """Test SA-specific functionality."""
    print(f"\n🧪 Testing SimulatedAnnealing Specific Features")
    print("-" * 50)
    
    try:
        env = MockJSSEnv(num_jobs=3)
        env.set_evaluation_behavior(makespan_to_return=100, max_steps_in_eval=3)
        
        # Test with different configurations
        configs = [
            {'initial_temp': 50.0, 'cooling_rate': 0.9, 'max_iter_per_restart': 5, 'num_restarts': 2, 'seed': 42},
            {'initial_temp': 100.0, 'cooling_rate': 0.95, 'max_iter_per_restart': 10, 'num_restarts': 3, 'seed': 123}
        ]
        
        for i, config in enumerate(configs):
            print(f"  🔧 Testing SA config {i+1}...", end=" ")
            
            policy = SimulatedAnnealingPolicy(env, **config)
            
            # Test that optimization runs
            observation = {
                'action_mask': np.array([1, 1, 1]),
                'real_obs': np.array([[3.0], [5.0], [2.0]])
            }
            
            action = policy.select_action(observation)
            
            if action in [0, 1, 2]:
                print("✅")
            else:
                print(f"❌ Invalid action: {action}")
                return False
        
        print("  🎉 SimulatedAnnealing specific tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in SA specific tests: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Baseline Policy Tests")
    print("=" * 50)
    
    # Create mock environment
    env = MockJSSEnv(num_jobs=3)
    
    # Test each policy
    test_results = []
    
    # Test RandomPolicy
    result = test_policy(RandomPolicy, "RandomPolicy", env)
    test_results.append(("RandomPolicy", result))
    
    # Test SPTPolicy
    result = test_policy(SPTPolicy, "SPTPolicy", env)
    test_results.append(("SPTPolicy", result))
    
    # Test SimulatedAnnealingPolicy
    sa_config = {
        'initial_temp': 50.0,
        'cooling_rate': 0.9,
        'max_iter_per_restart': 5,
        'num_restarts': 2,
        'seed': 42
    }
    result = test_policy(SimulatedAnnealingPolicy, "SimulatedAnnealingPolicy", env, **sa_config)
    test_results.append(("SimulatedAnnealingPolicy", result))
    
    # Test SA-specific features
    sa_specific_result = test_simulated_annealing_specific()
    test_results.append(("SA Specific Tests", sa_specific_result))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for policy_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {policy_name:<25} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Your baselines are working correctly.")
        print("You can now run the full evaluation with: python run_baseline_evaluation.py")
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
        return 1
    
    print("=" * 50)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
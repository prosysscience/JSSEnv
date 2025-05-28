# Baseline Evaluation Scripts

This directory contains comprehensive evaluation scripts for testing and comparing the performance of all implemented baseline policies for Job Shop Scheduling.

## Overview

The evaluation suite includes:

1. **`test_baselines.py`** - Quick functionality tests to verify baselines work correctly
2. **`run_baseline_evaluation.py`** - Comprehensive performance evaluation and comparison
3. **`baselines/evaluate.py`** - Original evaluation script (existing)

## Quick Start

### 1. Test Basic Functionality

First, verify that all your baseline policies work correctly:

```bash
python test_baselines.py
```

This will run basic functionality tests on all policies and report any issues.

### 2. Run Quick Evaluation

For a quick performance evaluation with reduced SA iterations:

```bash
python run_baseline_evaluation.py --quick --runs 3
```

### 3. Run Full Evaluation

For a comprehensive evaluation with default settings:

```bash
python run_baseline_evaluation.py
```

## Baseline Policies Tested

The evaluation scripts test the following baseline policies:

1. **RandomPolicy** - Selects actions randomly from legal actions
2. **SPTPolicy** - Shortest Processing Time dispatching rule
3. **SimulatedAnnealingPolicy** - Multiple SA configurations:
   - SA_Quick: Fast evaluation (50 iterations, 3 restarts)
   - SA_Standard: Standard evaluation (100 iterations, 5 restarts)
   - SA_Intensive: Thorough evaluation (200 iterations, 10 restarts)

## Command Line Options

### `run_baseline_evaluation.py`

```bash
python run_baseline_evaluation.py [OPTIONS]
```

**Options:**
- `--instances INSTANCE1 INSTANCE2 ...` - Specify which instances to test (default: ta01, ta02, ta03, ta04, ta05, ft06, ft10, ft20)
- `--runs N` - Number of runs per instance per policy (default: 5)
- `--output-dir DIR` - Output directory for results (default: evaluation_results)
- `--quick` - Run quick evaluation with fewer SA iterations
- `--help` - Show help message

**Examples:**

```bash
# Test only specific instances
python run_baseline_evaluation.py --instances ta01 ta02 ft06

# Run more thorough evaluation
python run_baseline_evaluation.py --runs 10

# Quick test with custom output directory
python run_baseline_evaluation.py --quick --output-dir quick_results
```

## Output Files

The evaluation script generates several output files in the specified output directory:

### CSV Results
- `baseline_results_TIMESTAMP.csv` - Detailed results for every run
  - Columns: policy, instance, run_id, makespan, runtime, steps, success

### JSON Summary
- `baseline_summary_TIMESTAMP.json` - Complete evaluation summary including:
  - Evaluation metadata (timestamp, instances, parameters)
  - Aggregated results by policy
  - Error log

### Error Log
- `baseline_errors_TIMESTAMP.txt` - Detailed error messages if any issues occurred

## Understanding the Results

### Performance Metrics

For each policy and instance combination, the following metrics are reported:

- **Makespan** - Total completion time (lower is better)
- **Runtime** - Time taken to compute the solution (lower is better)
- **Success Rate** - Percentage of runs that completed successfully
- **Steps** - Number of environment steps taken

### Analysis Sections

The evaluation provides several analysis views:

1. **Overall Performance Summary** - Average performance across all instances
2. **Per-Instance Performance** - Detailed breakdown by instance
3. **Best Policy Per Instance** - Which policy performed best on each instance

### Sample Output

```
🎯 Overall Performance Summary:
----------------------------------------
Policy                    Avg_Makespan  Std_Makespan  Min_Makespan  Max_Makespan  Avg_Runtime  Success_Rate  Total_Runs
SPTPolicy                      245.600         12.450       230.000       265.000        0.045         1.000          40
SA_Standard                    251.200         15.230       235.000       275.000        2.340         1.000          40
RandomPolicy                   289.450         28.670       250.000       340.000        0.032         1.000          40
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   Error importing baselines: No module named 'baselines'
   ```
   **Solution:** Make sure you're running the script from the project root directory.

2. **Environment Creation Errors**
   ```
   Failed to create environment for instance 'ta01': ...
   ```
   **Solution:** Check that JSSEnv is properly installed and the instance identifiers are correct for your setup.

3. **Policy Instantiation Errors**
   ```
   Error creating RandomPolicy: Cannot determine num_jobs from environment
   ```
   **Solution:** Ensure your environment has the required `action_space.n` attribute or modify the policy code.

### Debugging Steps

1. **Run Basic Tests First**
   ```bash
   python test_baselines.py
   ```

2. **Test with Mock Environment**
   ```bash
   python -c "from baselines.tests.mock_env import MockJSSEnv; print('Mock env works!')"
   ```

3. **Test Individual Policies**
   ```bash
   python -c "from baselines import RandomPolicy; print('RandomPolicy import works!')"
   ```

4. **Check JSSEnv Installation**
   ```bash
   python -c "import gymnasium as gym; env = gym.make('jss-v1'); print('JSSEnv works!')"
   ```

## Customization

### Adding New Instances

Edit the `DEFAULT_INSTANCES` list in `run_baseline_evaluation.py`:

```python
DEFAULT_INSTANCES = [
    "ta01", "ta02", "ta03",  # Taillard instances
    "ft06", "ft10", "ft20",  # Fisher-Thompson instances
    "your_custom_instance"   # Your instances
]
```

### Adding New SA Configurations

Add new configurations to the `SA_CONFIGS` dictionary:

```python
SA_CONFIGS = {
    "SA_Custom": {
        'initial_temp': 150.0,
        'cooling_rate': 0.97,
        'max_iter_per_restart': 150,
        'num_restarts': 7,
        'seed': 42
    }
}
```

### Modifying Environment Creation

If your JSSEnv setup requires different environment creation parameters, modify the `create_environment` method in the `BaselineEvaluator` class.

## Performance Tips

1. **Use `--quick` for Development** - Reduces SA computation time significantly
2. **Start with Fewer Instances** - Test with 1-2 instances first
3. **Monitor Memory Usage** - SA with deep copying can be memory intensive
4. **Use Multiple Runs** - At least 5 runs per instance for statistical significance

## Integration with Existing Code

The evaluation scripts are designed to work alongside your existing baseline implementations without modification. They use the same interfaces defined in `base_policy.py`.

## Contributing

When adding new baseline policies:

1. Ensure they inherit from `BaselinePolicy`
2. Implement the required `select_action` method
3. Add them to the imports in `baselines/__init__.py`
4. Update the evaluation scripts to include the new policy
5. Add appropriate tests

## Support

If you encounter issues:

1. Check the error logs generated by the evaluation
2. Run the basic functionality tests first
3. Verify your JSSEnv installation and instance files
4. Check that all baseline policies can be imported correctly 
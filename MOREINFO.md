Cloud RL Scheduling Project
Goal: Develop a custom Deep Reinforcement Learning (DRL) scheduling policy for the CloudSim environment to optimize cloud resource allocation, minimize resource waste, and reduce job queue/waiting times.

Team: Ali, Matthew, Chris, Abhinav, Gautham (CSE 190, UCSD)

1. Project Overview
This project aims to replace the default scheduling mechanisms in CloudSim with one or more DRL agents. We will explore various RL architectures, starting with Deep Q-Networks (DQN) and potentially extending to Multi-Agent RL (MARL), Hierarchical RL (HRL), and Meta-RL to handle dynamic cloud workloads effectively.

Key Objectives:

Integrate a DRL agent with the CloudSim simulation environment.

Define appropriate state spaces, action spaces, and reward functions for cloud scheduling.

Train DRL agents to learn optimal or near-optimal scheduling policies.

Benchmark DRL schedulers against CloudSim's default schedulers and other baselines.

Evaluate performance based on metrics like job waiting time, resource utilization, and energy consumption.

2. Prerequisites
Java Development Kit (JDK): Version 8 or higher (CloudSim is Java-based). Ensure JAVA_HOME is set and java is in your PATH.

Python: Version 3.8 or higher.

CloudSim: Version 4.0 or a specific version/fork you decide to use (e.g., CloudSimPlus). See setup for details.

Git: For version control.

(Recommended) Conda or venv: For Python environment management.

(Recommended) IDE: VS Code (Cursor), PyCharm, or IntelliJ IDEA (especially for Java/CloudSim parts).

3. Environment Setup
3.1. Clone the Repository
git clone <your-repository-url>
cd <your-project-directory-name>

3.2. Set up Python Virtual Environment
Using a virtual environment is crucial for managing project dependencies and ensuring reproducibility.

Using venv (standard Python):

# Navigate to your project directory
python -m venv .venv

# Activate the environment:
# On Windows:
# .\.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate

Using conda:

conda create -n cloudrl_env python=3.9  # Or your preferred Python version
conda activate cloudrl_env

3.3. Install Python Dependencies
A requirements.txt file should be maintained for Python packages.

pip install -r requirements.txt

Create/Update requirements.txt with essential packages:

# Basic ML/RL
numpy
gymnasium  # Or gym for older versions
torch      # Or tensorflow
cloudpickle

# Data Handling & Plotting
pandas
matplotlib
seaborn

# Utilities
jupyterlab # For notebooks
tqdm       # For progress bars
pyyaml     # For config files

# Potentially for Java-Python communication (research best option for your needs)
# pyjnius
# grpcio
# flask (if building a simple API wrapper around CloudSim)

3.4. CloudSim Setup & Integration
This is a critical step, as CloudSim is Java-based and your RL agent will be in Python.

Obtain CloudSim:

Recommendation: Consider using CloudSimPlus. It's often easier to work with and has more modern features. You can clone its repository or download a release.

Alternatively, use a standard CloudSim distribution (e.g., from the original CloudSim repository or its forks).

Place the CloudSim source or JARs in a designated directory within your project, e.g., cloudsim_framework/.

Python-Java Interaction Strategy:

Option 1: External Script Execution (Simple Start):

Python writes a configuration file (e.g., JSON, XML) for a CloudSim scenario.

Python executes a compiled Java program (that runs CloudSim) as a subprocess.

The Java program writes results (state, reward) to a file or standard output.

Python parses these results.

Pros: Simple to implement initially. Cons: Slow for many iterations.

Option 2: Local Server/API (More Robust):

Write a small Java application using CloudSim that runs as a server (e.g., using Spring Boot, SparkJava, or even a simple java.net.ServerSocket).

This server exposes endpoints for: initializing simulation, stepping simulation with an action, getting state, getting reward.

Python client (e.g., using requests or grpc) communicates with this Java server.

Pros: Better for iterative development. Cons: More setup.

Option 3: pyjnius or JPype (Direct Call):

These libraries allow Python to instantiate Java objects and call Java methods directly.

Pros: Potentially fastest communication. Cons: Can be complex to set up, manage dependencies, and debug.

Example CloudSim Directory Structure (if included in your repo):

cloudsim_framework/
├── cloudsimplus/  # If using CloudSimPlus source
├── lib/           # Or pre-compiled CloudSim JARs
└── src_java/      # Your custom Java code for the CloudSim simulation runner or API server

If you're using CloudSim as a Maven/Gradle project for your Java part, manage it there.

4. Project Structure (Suggested)
├── .git/
├── .venv/                     # Python virtual environment (if using venv)
├── cloudsim_framework/        # CloudSim source/JARs, custom Java wrappers/server
│   ├── lib/                   # Compiled JARs (CloudSim, your Java code)
│   └── src_java/              # Your Java source for interacting with CloudSim
├── data/
│   ├── workloads/             # Definitions of job workloads (e.g., JSON, SWF traces)
│   └── results/               # Experiment outputs, logs, trained models, plots
├── notebooks/                 # Jupyter notebooks for exploration, analysis, visualization
├── src_python/                # Main Python source code for the RL project
│   ├── agents/                # DRL agent implementations (DQN, MARL, etc.)
│   ├── environment/           # CloudSim-Python interface, MDP definitions (state, action, reward)
│   ├── common/                # Utility functions, replay buffers, network architectures
│   ├── configs/               # Configuration files for experiments (e.g., YAML)
│   ├── main_train.py          # Main script for training agents
│   ├── main_evaluate.py       # Main script for evaluating agents
│   └── baselines/             # Implementations of baseline schedulers (FCFS, Random etc.)
├── tests/                     # Unit and integration tests
│   ├── python/
│   └── java/                  # If you have custom Java code
├── .gitignore
├── LICENSE
├── README.md                  # This file
└── requirements.txt           # Python package dependencies

5. How to Run
(This section will evolve. Start with simple steps and expand.)

5.1. Running a Baseline CloudSim Simulation (Illustrative)
(This depends heavily on your chosen Python-Java interaction method and CloudSim setup)

If you have a Java main class com.example.MyCloudSimRunner in cloudsim_framework/src_java/ compiled to cloudsim_framework/lib/my_runner.jar and CloudSim JARs in cloudsim_framework/lib/:

# Compile your Java code first (e.g., using javac or a build tool like Maven/Gradle)
# Example execution:
java -cp "cloudsim_framework/lib/*" com.example.MyCloudSimRunner --workload data/workloads/sample_workload.json

5.2. Training an RL Agent (Example)
# Activate your Python environment
# source .venv/bin/activate  OR  conda activate cloudrl_env

# Run the training script
python src_python/main_train.py --config src_python/configs/dqn_experiment_01.yaml

5.3. Evaluating a Trained Agent
python src_python/main_evaluate.py --model_path data/results/dqn_model_final.pt --config src_python/configs/dqn_experiment_01.yaml --output_dir data/results/evaluation_run_X/

6. Key Files for Quick Iteration
RL Agent Logic: src_python/agents/<your_agent_class>.py

Modifying neural network architectures, exploration strategies, learning updates.

Environment Wrapper & MDP Definition: src_python/environment/cloudsim_rl_env.py

Defining state, action, reward. This is where Python talks to CloudSim.

Reward Function: Likely a core part of cloudsim_rl_env.py. Iterating on this is key.

Training Script: src_python/main_train.py

Adjusting training loops, logging, and experiment orchestration.

Experiment Configuration Files: src_python/configs/*.yaml

Changing hyperparameters, workload files, agent types without code modification.

CloudSim Java Runner/Server: (e.g., cloudsim_framework/src_java/com/example/MyCloudSimRunner.java)

If you need to modify how CloudSim itself is set up or what data it exposes.

7. Development Workflow & Tips
IDE Integration (VS Code/Cursor):

Install extensions: Python (Microsoft), Pylance, Ruff, (optionally) Java Extension Pack.

Configure the Python interpreter to your project's virtual environment (.venv).

Use the built-in debugger for both Python and Java (if your IDE supports it for Java).

Start Simple, Iterate:

Get a basic CloudSim scenario running purely in Java.

Establish the simplest possible Python-to-Java communication (e.g., Python writes a file, Java reads it, simulates, writes results, Python reads results).

Implement a Python script with a random agent making decisions through this bridge.

Replace the random agent with your DRL agent (e.g., DQN).

Gradually improve the communication bridge for speed if needed.

Version Control (Git):

Commit frequently with clear, descriptive messages.

Use feature branches (git checkout -b feature/new-reward-logic).

Regularly git pull and git push when collaborating.

Consider a branching strategy (e.g., Gitflow or GitHub Flow).

Comprehensive Logging:

Log key information: selected actions, states, rewards, episode lengths, loss values, custom metrics (e.g., average wait time from CloudSim).

Use Python's logging module.

Configuration Management:

Use YAML or JSON files for experiment parameters (learning rates, network sizes, workload paths, etc.) to avoid hardcoding.

Modular Code: Design your Python agent, environment wrapper, and utility functions to be as modular and reusable as possible.

Unit & Integration Tests:

Write tests for critical components: reward calculations, state transformations, parts of your Python-Java bridge.

8. Contribution Guidelines (For Team Collaboration)
Branching: Main development on develop branch. Create feature branches from develop. Merge back via Pull Requests. main branch for stable, tagged releases/milestones.

Code Style:

Python: Follow PEP 8. Use a formatter like Black and a linter like Ruff.

Java: Follow standard Java coding conventions.

Pull Requests (PRs):

All new code should be merged via PRs.

Require at least one reviewer from the team.

Ensure PRs include a clear description of changes.

Issue Tracking: Use your Git provider's issue tracker (e.g., GitHub Issues) for tasks, bugs, and discussions.

Regular Sync-ups: Hold brief, regular meetings (e.g., daily or bi-weekly stand-ups) to discuss progress, blockers, and plan.

This README is a living document. Update it as your project evolves, tools change, or setup instructions become more refined. Good luck with your Cloud RL project!
import gym
import unittest
import numpy as np
from pathlib import Path


class TestState(unittest.TestCase):
    def test_random(self):
        env = gym.make(
            "jss-v1",
            env_config={
                "instance_path": f"{str(Path(__file__).parent.absolute())}/../JSSEnv/envs/instances/ta01"
            },
        )
        average = 0
        for _ in range(100):
            state = env.reset()
            self.assertEqual(env.current_time_step, 0)
            legal_actions = env.get_legal_actions()
            done = False
            total_reward = 0
            self.assertTrue(
                max(state["real_obs"].flatten()) <= 1.0, "Out of max bound state"
            )
            self.assertTrue(
                min(state["real_obs"].flatten()) >= 0.0, "Out of min bound state"
            )
            self.assertTrue(
                not np.isnan(state["real_obs"]).any(), "NaN inside state rep!"
            )
            self.assertTrue(
                not np.isinf(state["real_obs"]).any(), "Inf inside state rep!"
            )
            machines_available = set()
            for job in range(len(env.legal_actions[:-1])):
                if env.legal_actions[job]:
                    machine_needed = env.needed_machine_jobs[job]
                    machines_available.add(machine_needed)
            self.assertEqual(
                len(machines_available),
                env.nb_machine_legal,
                "machine available and nb machine available are not coherant",
            )
            while not done:
                actions = np.random.choice(
                    len(legal_actions), 1, p=(legal_actions / legal_actions.sum())
                )[0]
                assert legal_actions[:-1].sum() == env.nb_legal_actions
                state, rewards, done, _ = env.step(actions)
                legal_actions = env.get_legal_actions()
                total_reward += rewards
                self.assertTrue(
                    max(state["real_obs"].flatten()) <= 1.0, "Out of max bound state"
                )
                self.assertTrue(
                    min(state["real_obs"].flatten()) >= 0.0, "Out of min bound state"
                )
                self.assertTrue(
                    not np.isnan(state["real_obs"]).any(), "NaN inside state rep!"
                )
                self.assertTrue(
                    not np.isinf(state["real_obs"]).any(), "Inf inside state rep!"
                )
                machines_available = set()
                for job in range(len(env.legal_actions[:-1])):
                    if env.legal_actions[job]:
                        machine_needed = env.needed_machine_jobs[job]
                        machines_available.add(machine_needed)
                assert len(machines_available) == env.nb_machine_legal
            average += env.last_time_step
            self.assertEqual(len(env.next_time_step), 0)
            self.assertNotEqual(
                min(env.solution.flatten()), -1, np.array2string(env.solution.flatten())
            )
            for job in range(env.jobs):
                self.assertEqual(env.todo_time_step_job[job], env.machines)

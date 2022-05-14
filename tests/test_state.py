import gym
import unittest
import numpy as np


class TestState(unittest.TestCase):
    def test_last_action(self):
        """Threw IndexError: index 101 is out of bounds for axis 0 with size 100 until it was fixed"""

        seed = 34
        env = gym.make(
            "JSSEnv:JSSEnv-v1",
            env_config={"instance_path": "../JSSEnv/envs/instances/ta80"},
        )
        env.seed(seed)
        _ = env.reset()

        action = env.action_space.n
        obs, reward, done, _ = env.step(action)
        env.render()
        if done:
            print("Episode ended")

        env.close()

    def test_seed(self):
        env = gym.make(
            "JSSEnv:JSSEnv-v1",
            env_config={"instance_path": "../JSSEnv/envs/instances/ta80"},
        )
        env.seed(42)

        action_list1 = [env.action_space.sample() for _ in range(5)]

        env.seed(3141592)
        action_list2 = [env.action_space.sample() for _ in range(5)]

        expected_actions1 = [75, 25, 61, 29, 59]
        expected_actions2 = [53, 18, 59, 95, 18]

        self.assertEqual(action_list1, expected_actions1)
        self.assertEqual(action_list2, expected_actions2)

    def test_random_episode1(self):
        """Threw IndexError: pop from empty list until it was fixed"""
        seed_list = [42, 3, 314, 315]
        env = gym.make(
            "JSSEnv:JSSEnv-v1",
            env_config={"instance_path": "../JSSEnv/envs/instances/ta80"},
        )
        for seed in seed_list:
            env.seed(seed)
            _ = env.reset()

            while True:
                action = env.action_space.sample()
                obs, reward, done, _ = env.step(action)
                env.render()
                if done:
                    print("Episode ended")
                    break

        env.close()

    def test_random(self):
        env = gym.make(
            "JSSEnv:JSSEnv-v1",
            env_config={"instance_path": "../JSSEnv/envs/instances/ta80"},
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


if __name__ == "__main__":
    unittest.main()

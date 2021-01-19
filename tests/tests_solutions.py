from JSSEnv.envs import JssEnv

import unittest


class TestSolution(unittest.TestCase):

    def test_optimum_ta01(self):
        env = JssEnv({'instance_path': '../JSSEnv/envs/instances/ta01'})
        env.reset()
        self.assertEqual(env.current_time_step, 0)
        # for every machine give the jobs to process in order for every machine
        solution_sequence = [
            [7, 11, 9, 10, 8, 3, 12, 2, 14, 5, 1, 6, 4, 0, 13],
            [2, 8, 7, 14, 6, 13, 9, 11, 4, 5, 12, 3, 10, 1, 0],
            [11, 9, 3, 0, 4, 12, 8, 7, 5, 2, 6, 14, 13, 10, 1],
            [6, 5, 0, 9, 12, 7, 11, 10, 14, 1, 13, 2, 3, 4, 8],
            [10, 13, 0, 4, 1, 5, 14, 3, 7, 6, 12, 8, 2, 9, 11],
            [5, 7, 3, 12, 13, 10, 1, 11, 8, 4, 2, 6, 0, 9, 14],
            [9, 0, 4, 8, 3, 11, 13, 14, 6, 12, 10, 2, 1, 7, 5],
            [4, 6, 7, 10, 0, 11, 1, 9, 3, 5, 13, 14, 8, 2, 12],
            [13, 4, 6, 2, 9, 14, 12, 11, 7, 10, 0, 1, 3, 8, 5],
            [9, 3, 2, 4, 13, 11, 12, 1, 0, 7, 8, 5, 14, 10, 6],
            [8, 14, 4, 3, 11, 12, 9, 0, 10, 13, 5, 1, 6, 2, 7],
            [7, 9, 8, 5, 6, 0, 2, 3, 1, 13, 14, 12, 4, 10, 11],
            [6, 0, 7, 11, 5, 14, 10, 2, 4, 13, 8, 9, 3, 12, 1],
            [13, 10, 7, 9, 5, 3, 11, 1, 12, 14, 2, 4, 0, 6, 8],
            [13, 11, 6, 8, 7, 4, 1, 5, 3, 10, 0, 14, 9, 2, 12]
        ]
        done = False
        job_nb = len(solution_sequence[0])
        machine_nb = len(solution_sequence)
        index_machine = [0 for x in range(machine_nb)]
        step_nb = 0
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(solution_sequence)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = solution_sequence[machine][index_machine[machine]]
                    if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                        no_op = False
                        self.assertTrue(env.legal_actions[action_to_do], "We don't perform illegal actions")
                        self.assertEqual(sum(env.legal_actions[:-1]), env.nb_legal_actions)
                        state, reward, done, _ = env.step(action_to_do)
                        index_machine[machine] += 1
                        step_nb += 1
            if no_op and not done:
                self.assertTrue(len(env.next_time_step) > 0, "step {}".format(step_nb))
                previous_time_step = env.current_time_step
                env._increase_time_step()
                self.assertTrue(env.current_time_step > previous_time_step, "we increase the time step")
        self.assertEqual(sum(index_machine), len(solution_sequence) * len(solution_sequence[0]))
        self.assertEqual(env.current_time_step, 1231)
        env.reset()
        self.assertEqual(env.current_time_step, 0)

    def test_optimum_ta51(self):
        env = JssEnv({'instance_path': '../JSSEnv/envs/instances/ta51'})
        env.reset()
        self.assertEqual(env.current_time_step, 0)
        # for every machine give the jobs to process in order for every machine
        solution_sequence = [
            [37, 5, 24, 19, 41, 31, 17, 45, 4, 26, 34, 22, 0, 14, 8, 38, 44, 6, 32, 39, 15, 13, 35, 12, 33, 29, 9, 20,
             7, 10, 3, 16, 23, 28, 25, 48, 18, 43, 27, 21, 11, 30, 1, 47, 42, 36, 40, 46, 49, 2],
            [14, 20, 1, 33, 45, 28, 9, 6, 5, 17, 18, 8, 42, 35, 3, 23, 0, 30, 44, 31, 16, 21, 38, 32, 41, 10, 34, 24,
             43, 22, 40, 29, 4, 26, 27, 48, 15, 25, 2, 47, 36, 12, 39, 46, 7, 11, 13, 49, 19, 37],
            [28, 2, 34, 44, 5, 20, 8, 37, 18, 42, 14, 22, 41, 24, 12, 32, 49, 15, 40, 33, 10, 13, 26, 11, 19, 3, 9, 45,
             21, 1, 4, 16, 23, 35, 39, 47, 29, 6, 0, 36, 7, 38, 48, 31, 27, 46, 43, 17, 25, 30],
            [5, 4, 28, 40, 41, 9, 0, 13, 14, 20, 34, 11, 46, 42, 12, 8, 37, 49, 23, 44, 24, 22, 32, 29, 48, 38, 2, 33,
             3, 18, 45, 36, 35, 15, 30, 19, 1, 31, 21, 6, 10, 27, 25, 26, 17, 7, 47, 43, 39, 16],
            [15, 9, 47, 29, 20, 14, 34, 12, 5, 40, 49, 2, 23, 0, 37, 8, 10, 46, 18, 41, 11, 38, 3, 16, 7, 39, 33, 30, 4,
             35, 26, 44, 45, 24, 48, 1, 19, 21, 17, 31, 27, 25, 43, 42, 22, 28, 6, 32, 13, 36],
            [4, 9, 44, 35, 37, 34, 0, 5, 14, 10, 42, 13, 40, 8, 39, 45, 21, 3, 11, 18, 46, 33, 32, 47, 27, 49, 41, 26,
             20, 22, 38, 2, 28, 7, 24, 31, 15, 29, 23, 30, 19, 36, 6, 25, 48, 1, 16, 12, 17, 43],
            [5, 34, 49, 14, 6, 42, 39, 7, 26, 4, 43, 40, 10, 37, 38, 8, 9, 41, 12, 27, 45, 25, 15, 48, 30, 44, 0, 29, 2,
             11, 19, 23, 21, 13, 16, 33, 24, 17, 35, 3, 22, 46, 20, 18, 36, 31, 28, 32, 47, 1],
            [46, 31, 9, 8, 35, 45, 34, 26, 4, 20, 0, 44, 21, 3, 13, 38, 18, 41, 14, 11, 22, 36, 27, 15, 25, 33, 1, 29,
             43, 40, 16, 32, 30, 10, 28, 7, 24, 47, 48, 6, 2, 23, 39, 49, 17, 5, 19, 42, 37, 12],
            [29, 12, 35, 28, 20, 6, 0, 13, 10, 8, 34, 31, 24, 15, 32, 40, 45, 30, 14, 9, 2, 49, 41, 37, 38, 48, 11, 26,
             16, 1, 44, 36, 18, 3, 39, 46, 23, 42, 4, 43, 21, 22, 25, 19, 27, 17, 5, 7, 47, 33],
            [10, 0, 45, 6, 17, 49, 46, 34, 23, 38, 44, 4, 11, 27, 37, 21, 31, 14, 18, 8, 12, 20, 40, 3, 33, 41, 9, 32,
             35, 48, 16, 1, 43, 39, 24, 15, 47, 36, 29, 5, 2, 28, 26, 42, 25, 22, 19, 30, 13, 7],
            [16, 0, 6, 7, 46, 3, 14, 18, 41, 5, 8, 35, 32, 39, 43, 34, 37, 22, 4, 24, 13, 48, 12, 11, 45, 49, 2, 44, 40,
             9, 10, 29, 38, 31, 19, 27, 21, 17, 23, 33, 30, 25, 15, 28, 42, 47, 20, 36, 1, 26],
            [34, 10, 7, 17, 45, 40, 2, 47, 26, 13, 9, 14, 15, 44, 4, 48, 19, 37, 42, 29, 49, 43, 41, 36, 22, 11, 32, 6,
             25, 8, 5, 28, 0, 20, 3, 38, 27, 1, 18, 33, 23, 31, 16, 30, 39, 24, 21, 35, 46, 12],
            [9, 16, 29, 44, 17, 34, 28, 7, 12, 3, 20, 21, 41, 13, 0, 8, 46, 32, 4, 6, 37, 14, 36, 10, 15, 24, 38, 11,
             33, 26, 25, 48, 39, 1, 31, 42, 47, 27, 22, 30, 19, 45, 43, 23, 35, 2, 40, 5, 49, 18],
            [6, 17, 2, 31, 25, 7, 36, 28, 47, 35, 5, 46, 20, 34, 18, 0, 14, 24, 15, 8, 42, 37, 13, 49, 41, 43, 12, 45,
             32, 9, 40, 33, 26, 38, 44, 16, 11, 3, 1, 4, 29, 21, 30, 27, 10, 48, 23, 22, 19, 39],
            [7, 35, 37, 13, 34, 14, 42, 39, 5, 18, 40, 45, 8, 6, 23, 32, 2, 49, 9, 3, 43, 47, 25, 41, 22, 10, 11, 20,
             26, 16, 33, 4, 30, 44, 21, 1, 29, 27, 38, 12, 17, 46, 0, 24, 36, 19, 28, 15, 31, 48]]
        done = False
        machine_nb = len(solution_sequence)
        job_nb = len(solution_sequence[0])
        index_machine = [0 for x in range(machine_nb)]
        step_nb = 0
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(solution_sequence)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = solution_sequence[machine][index_machine[machine]]
                    if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                        no_op = False
                        self.assertTrue(env.legal_actions[action_to_do], "We don't perform illegal actions")
                        self.assertEqual(sum(env.legal_actions[:-1]), env.nb_legal_actions)
                        state, reward, done, _ = env.step(action_to_do)
                        index_machine[machine] += 1
                        step_nb += 1
            if no_op and not done:
                self.assertTrue(len(env.next_time_step) > 0, "step {}".format(step_nb))
                previous_time_step = env.current_time_step
                state, reward, done, _ = env.step(env.jobs)
                self.assertTrue(env.current_time_step > previous_time_step, "we increase the time step")
        self.assertEqual(sum(index_machine), len(solution_sequence) * len(solution_sequence[0]))
        self.assertEqual(env.current_time_step, 2760)
        env.reset()
        self.assertEqual(env.current_time_step, 0)


if __name__ == '__main__':
    unittest.main()

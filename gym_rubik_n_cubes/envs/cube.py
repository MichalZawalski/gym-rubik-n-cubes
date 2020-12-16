import numpy as np
import math
import gym
from gym import spaces

from gym_rubik_n_cubes.envs import helpers


class RubiksCubeEnv(gym.Env):
    def __init__(self, n, n_shuffles):
        self.n = n
        self.state = None
        self.moves = []
        self.shape = (6, n, n, 6)
        self.n_shuffles = n_shuffles
        self.smart_shuffles = False
        self.compute_reward = None

        self.action_space = spaces.Discrete(6 * (n // 2))
        self.observation_space = spaces.Box(low=0, high=1, shape=self.shape, dtype=np.float32)

        self._setup_state()
        self._setup_moves()
        self._setup_rewards()

        self.goal_state = self.state

    def _setup_state(self):
        # TODO implement custom patterns
        self.state = np.arange(6 * self.n * self.n) // (self.n * self.n)

    def _setup_moves(self):
        faces = [0, 1, 2, 3, 4, 5]
        centers = [(0, 1, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1), (1, 0, 0), (-1, 0, 0)]
        directions = [[(-1, 0, 0), (0, 0, 1)], [(-1, 0, 0), (0, 0, -1)], [(-1, 0, 0), (0, 1, 0)],
                      [(1, 0, 0), (0, 1, 0)], [(0, 0, -1), (0, 1, 0)], [(0, 0, 1), (0, 1, 0)]]

        stickers = []

        for f in faces:
            for i in range(self.n):
                for j in range(self.n):
                    move_y = helpers.multiply(directions[f][0], 1 - (2 * i + 1) / self.n)
                    move_x = helpers.multiply(directions[f][1], (2 * j + 1) / self.n - 1)

                    stickers.append([helpers.add(centers[f], helpers.add(move_x, move_y)), f])

        for f in faces:
            for level in range(self.n // 2):
                for direction in [-1, 1]:
                    dist = 1 - (2 * level + 1) / self.n
                    moved_stickers = []

                    for s in stickers:
                        if math.isclose(dist, helpers.dot(centers[f], s[0])) or (
                                level == 0 and math.isclose(1, helpers.dot(centers[f], s[0]))):
                            moved_stickers.append([helpers.rotate(s[0], centers[f], direction), s[1]])
                        else:
                            moved_stickers.append(s)

                    # moved_stickers stand for permutation
                    move_permutation = [0] * (6 * self.n * self.n)

                    for i in range(len(moved_stickers)):
                        found = False
                        for j in range(len(stickers)):
                            if np.allclose(moved_stickers[i][0], stickers[j][0]):
                                move_permutation[i] = j
                                found = True
                                break

                        assert found

                    self.moves.append(np.array(move_permutation))

    def _setup_rewards(self):
        def sparse_penalize_step(state, goal):
            return 0 if np.array_equal(state, goal) else -1

        def sparse_reached(state, goal):
            return 1 if np.array_equal(state, goal) else 0

        def shaped_matching_stickers(state, goal):
            return np.average(state == goal)

        # TODO implement reward choice
        self.compute_reward = sparse_penalize_step

    def reset(self):
        self._setup_state()
        self._shuffle()

        return self._get_state()

    def step(self, action):
        self._move_by_action(action)
        reward = self.compute_reward(self.state, self.goal_state)
        done = np.array_equal(self.state, self.goal_state)

        return self._get_state(), reward, done, dict()

    def render(self, mode='human'):
        figure = np.full((3 * self.n, 4 * self.n), -1)
        state = np.reshape(self.state, self.shape[:-1])
        order = [0, 5, 2, 4, 3, 1]

        figure[0:self.n, self.n:2 * self.n] = np.rot90(state[order[0]])

        for i in range(4):
            face = np.rot90(state[order[i + 1]], 1)
            figure[self.n:2 * self.n, i * self.n:(i + 1) * self.n] = face

        face = np.rot90(state[order[5]])
        figure[2 * self.n:3 * self.n, self.n:2 * self.n] = face

        for i in range(figure.shape[0]):
            for j in range(figure.shape[1]):
                helpers.print_coloured_square(figure[i, j])
            print()

    def _move_by_action(self, action):
        self.state = self.state[self.moves[action]]

    def _shuffle(self):
        for _ in range(self.n_shuffles):
            # TODO implement smart shuffles
            action = self.action_space.sample()
            self._move_by_action(action)

    def _get_state(self):
        return np.reshape(np.eye(6)[self.state], self.shape)


class GoalRubiksCubeEnv(RubiksCubeEnv):
    def __init__(self, n, n_shuffles):
        super().__init__(n, n_shuffles)

    def reset(self):
        observation = super().reset()
        return self._make_goal_observation(observation)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return self._make_goal_observation(observation), reward, done, info

    def _make_goal_observation(self, observation):
        return {'observation': observation, 'achieved_goal': observation, 'desired_goal': self.goal_state}

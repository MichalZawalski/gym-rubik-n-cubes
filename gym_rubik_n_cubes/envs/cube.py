import numpy as np
from scipy.spatial.transform import Rotation
import math
import gym
from gym import spaces


def add(u, v):
    return tuple(sum(x) for x in zip(u, v))


def multiply(v, m):
    return tuple(x * m for x in v)


def rotate(v, along, direction=1):
    return tuple(Rotation.from_rotvec(multiply(along, direction * np.pi / 2)).apply(v))


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


class RubiksCubeEnv(gym.Env):
    def __init__(self, n):
        self.n = n
        self.state = np.arange(6 * n * n) // (n * n)
        self.moves = []
        self.shape = (6, n, n)
        self.action_space = spaces.Discrete(6 * (n // 2))
        self.observation_space = spaces.Box(low=0, high=1, shape=(6, n, n, 6), dtype=np.float32)

        self.setup_moves()

    def setup_moves(self):
        faces = [0, 1, 2, 3, 4, 5]
        centers = [(0, 1, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1), (1, 0, 0), (-1, 0, 0)]
        directions = [[(-1, 0, 0), (0, 0, 1)], [(-1, 0, 0), (0, 0, -1)], [(-1, 0, 0), (0, 1, 0)],
                      [(1, 0, 0), (0, 1, 0)], [(0, 0, -1), (0, 1, 0)], [(0, 0, 1), (0, 1, 0)]]

        stickers = []

        for f in faces:
            for i in range(self.n):
                for j in range(self.n):
                    move_y = multiply(directions[f][0], 1 - (2 * i + 1) / self.n)
                    move_x = multiply(directions[f][1], (2 * j + 1) / self.n - 1)

                    stickers.append([add(centers[f], add(move_x, move_y)), f])

        for f in faces:
            for level in range(self.n // 2):
                for direction in [-1, 1]:
                    dist = 1 - (2 * level + 1) / self.n
                    moved_stickers = []

                    for s in stickers:
                        if math.isclose(dist, dot(centers[f], s[0])) or (
                                level == 0 and math.isclose(1, dot(centers[f], s[0]))):
                            moved_stickers.append([rotate(s[0], centers[f], direction), s[1]])
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

    def reset(self):
        self.state = np.arange(6 * self.n * self.n) // (self.n * self.n)

        return self._get_state()

    def step(self, action):
        self.state = self.state[self.moves[action]]

        return self._get_state()

    def render(self, mode='human'):
        print(self._get_state())

    def _get_state(self):
        return np.reshape(self.state, self.shape)


class GoalRubiksCubeEnv(RubiksCubeEnv):
    def __init__(self, n):
        super().__init__(n)

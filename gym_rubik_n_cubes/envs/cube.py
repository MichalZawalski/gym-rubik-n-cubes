import sys

import numpy as np
import math
import gym
from gym import spaces

from gym_rubik_n_cubes.envs import helpers


class RubiksCubeEnv(gym.Env):
    def __init__(self, n=3, n_shuffles=100, step_limit=100, reward_type='penalize_step', colour_pattern=None,
                 smart_shuffles=True):
        self.n = n

        self.n_shuffles = n_shuffles
        self.step_limit = step_limit
        self.reward_type = reward_type
        self.smart_shuffles = smart_shuffles

        self.colour_pattern = np.arange(6) if colour_pattern is None else np.array(colour_pattern)

        self.state = None
        self.moves = []
        self.observation_shape = (6, n, n, 6)
        self.compute_reward = None
        self.steps = 0

        self.action_space = spaces.Discrete(6 * (n // 2) * 2)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float32)

        self._setup_state()
        self._setup_moves()
        self._setup_rewards()

        self.goal_state = self.state

    def _setup_state(self):
        state = np.arange(6 * self.n * self.n) // (self.n * self.n)
        self.state = self.colour_pattern[state]

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
        def penalize_step(state, goal):
            return 0 if np.array_equal(state, goal) else -1

        def on_reached(state, goal):
            return 1 if np.array_equal(state, goal) else 0

        def matching_stickers(state, goal):
            return np.average(state == goal)

        if self.reward_type == 'penalize_step':
            self.compute_reward = penalize_step
        elif self.reward_type == 'on_reached':
            self.compute_reward = on_reached
        elif self.reward_type == 'matching_stickers':
            self.compute_reward = matching_stickers
        else:
            # No matching reward scheme found.
            if isinstance(self.reward_type, str):
                raise NameError('Incorrect reward type: "{0}"'.format(self.reward_type))
            else:
                print(
                    'Reward type "{0}" not on the list. It will be assumed to be a function.'.format(self.reward_type),
                    file=sys.stderr)
                self.compute_reward = self.reward_type

    def reset(self):
        self._setup_state()
        self._shuffle()

        self.steps = 0

        return self._get_state()

    def step(self, action):
        self._move_by_action(action)
        reward = self.compute_reward(self.state, self.goal_state)

        self.steps += 1
        done = np.array_equal(self.state, self.goal_state) or self.steps >= self.step_limit

        return self._get_state(), reward, done, dict()

    def render(self, mode='human'):
        figure = np.full((3 * self.n, 4 * self.n), -1)
        state = np.reshape(self.state, self.observation_shape[:-1])
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

    def _is_reversed(self, action1, action2):
        if action1 is None or action2 is None:
            return False
        return action1 // 2 == action2 // 2

    def _shuffle(self):
        last_action = None
        last_action_counter = 0

        for _ in range(self.n_shuffles):
            while True:
                action = self.action_space.sample()
                if not self.smart_shuffles:
                    break
                if self._is_reversed(action, last_action):
                    continue
                if action == last_action and last_action_counter >= 3:
                    continue
                break

            if last_action == action:
                last_action_counter += 1
            else:
                last_action = action
                last_action_counter = 1
            self._move_by_action(action)

    def _get_state(self):
        return np.reshape(np.eye(6)[self.state], self.observation_shape)


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

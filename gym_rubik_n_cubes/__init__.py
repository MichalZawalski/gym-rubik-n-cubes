from gym.envs.registration import register

register(
    id='RubiksCubeEnv-v0',
    entry_point='gym_rubik_n_cubes.envs:RubiksCubeEnv'
)

register(
    id='GoalRubiksCubeEnv-v0',
    entry_point='gym_rubik_n_cubes.envs:GoalRubiksCubeEnv'
)

"""
Contains utility functions for interacting with an environment.
"""

N_EPISODES = 10_000
TS_PER_EPISODE = 500


def print_env_info(env):
    print(f'Action space: {env.action_space}')
    print(f'Observation space: {env.observation_space}')
    print(f'Size of Observation space: {env.observation_space.n}')


def create_policy(env):
    """
    Returns a policy where in every state, there is a 25% chance for each action to occur (left, down, right, up).
    """
    policy = {}
    for state in range(0, env.observation_space.n):
        current_end = 0
        p = {}
        for action in range(0, env.action_space.n):
            p[action] = 1 / env.action_space.n
        policy[state] = p
    return policy

"""
Contains utility functions for interacting with an environment.
"""

import numpy as np

N_EPISODES = 10_000
MAX_TS_PER_EPISODE = 100

# FrozenLake-v0 actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


def print_env_info(env):
    print(f'Action space: {env.action_space}')
    print(f'Size of Action space: {env.action_space.n}')

    print(f'Observation space: {env.observation_space}')
    print(f'Size of Observation space: {env.observation_space.n}')


def create_policy(env):
    """
    Returns a stochastic policy where in every state, there is an equal chance for each action to occur.
    """
    policy = {}
    for state in range(0, env.observation_space.n):
        current_end = 0
        p = {}
        for action in range(0, env.action_space.n):
            p[action] = 1 / env.action_space.n
        policy[state] = p
    return policy


def greedy(Q, s):
    return np.argmax(Q[s])


def epsilon_greedy(Q, s, env, epsilon):
    # Select action based on e-greedy policy
    if np.random.random() <= epsilon:  # choose random exploratory action
        return np.random.randint(0, env.action_space.n)
    else:  # choose greedy action
        return np.argmax(Q[s])


def follow_greedy_policy(env, Q):
    """
    Given an environment and a Q-value table, greedily follow a policy derived from Q.
    For best results, Q should be approximately Q*.
    """

    s = env.reset()
    done = False

    while not done:
        env.render()

        a = greedy(Q, s)

        s, reward, done, _ = env.step(a)

    env.render()
    env.close()

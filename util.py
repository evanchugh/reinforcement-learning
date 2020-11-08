"""
Contains utility functions for interacting with an environment.
"""

import numpy as np

N_EPISODES = 10
MAX_TS_PER_EPISODE = 10

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


def epsilon_greedy_discrete(Q, s, env, epsilon):
    '''
    Chooses an epsilon-greedy action for discrete environments.
    '''
    # Select action based on e-greedy policy
    if np.random.random() <= epsilon:  # choose random exploratory action
        return np.random.randint(0, env.action_space.n)
    else:  # choose greedy action
        return np.argmax(Q[s])


def epsilon_greedy(model, s, env, epsilon):
    '''
    Chooses an epsilon-greedy action for environments with continuous state spaces but discrete action spaces.
    '''
    if np.random.random() <= epsilon:  # choose random exploratory action
        return np.random.randint(0, env.action_space.n)
    else:  # choose greedy action
        pred = model.predict(np.array([s]))
        return np.argmax(pred)


def follow_greedy_policy_discrete(env, Q):
    """
    Given an environment and a Q-value table, greedily follow a policy derived from Q.
    For best results, Q should be approximately Q*. Works with discrete environments.
    """

    s = env.reset()
    done = False

    while not done:
        env.render()

        a = greedy(Q, s)

        s, reward, done, _ = env.step(a)

    env.render()
    env.close()


def follow_greedy_policy(env, policy_network):
    """
    Given an environment and a Q-value table, greedily follow a policy derived from the policy network.
    Use in environments with continuous state spaces but discrete action spaces.
    """

    obs_space_size = env.observation_space.shape[0]

    s = env.reset()
    done = False

    ts_alive = 0

    while not done:
        env.render()
        
        a = np.argmax(policy_network.predict(s.reshape(-1, obs_space_size))[0])

        s, r, done, _ = env.step(a)

        if done:
            print(f'Survived for {ts_alive} timesteps.')

        ts_alive += 1

    env.close()

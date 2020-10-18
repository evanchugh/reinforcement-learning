"""
SARSA Policy Iteration. Implemented for environments with discrete state spaces.

On-policy.
"""

import numpy as np
import gym
from util import N_EPISODES


def greedy(Q, s):
    return np.argmax(Q[s])

def epsilon_greedy(Q, s, env, epsilon):
    # Select action based on e-greedy policy
    if np.random.random() <= epsilon:  # choose random exploratory action
        return np.random.randint(0, env.action_space.n)
    else:  # choose greedy action
        return np.argmax(Q[s])


def sarsa_policy_iteration(env, alpha=0.1, gamma=0.9, epsilon=0.5):
    """
    Returns approximation of Q*(s,a). Uses an epsilon-greedy policy derived from Q.
    alpha: learning rate
    gamma: discount rate
    epsilon: probability of taking an exploratory action
    """

    # arbitrary starting Q(s,a)
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for e in range(N_EPISODES):

        s = env.reset()  # current state

        a = epsilon_greedy(Q, s, env, epsilon)

        done = False

        while not done:
            # env.render()

            s_, reward, done, _ = env.step(a)
            a_ = epsilon_greedy(Q, s_, env, epsilon)

            Q[s, a] += alpha * (reward + gamma * Q[s_, a_] - Q[s, a])

            s, a = s_, a_

    # env.render()
    print(Q)

    env.close()
    return Q


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


if __name__ == '__main__':

    env = gym.make('FrozenLake-v0', is_slippery=False)

    q_star = sarsa_policy_iteration(env)  # approximation

    follow_greedy_policy(env, q_star)

    # print(f'Approximation of Q*: {q_star}')

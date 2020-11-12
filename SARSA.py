"""
Tabular SARSA Policy Iteration.

On-policy.
"""

import numpy as np
import gym
from util import epsilon_greedy_discrete, follow_greedy_policy_discrete


N_EPISODES = 1000
MAX_TS_PER_EPISODE = 100


def sarsa_policy_iteration(env, alpha=0.1, gamma=0.9, epsilon=0.5):
    """
    Returns approximation of Q*(s,a). Uses an epsilon-greedy policy derived from Q.
    alpha: learning rate
    gamma: discount rate
    epsilon: probability of taking an exploratory action
    """

    # arbitrary initialization
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for e in range(N_EPISODES):

        s = env.reset()  # current state
        a = epsilon_greedy_discrete(Q, s, env, epsilon)

        for i in range(MAX_TS_PER_EPISODE):
            # env.render()

            s_, reward, done, _ = env.step(a)
            a_ = epsilon_greedy_discrete(Q, s_, env, epsilon)

            Q[s, a] += alpha * (reward + gamma * Q[s_, a_] - Q[s, a])

            s, a = s_, a_

            if done:
                break

    # env.render()
    env.close()
    return Q


if __name__ == '__main__':

    env = gym.make('FrozenLake-v0', is_slippery=False)

    # q_star = sarsa_policy_iteration(env)  # approximation
    # follow_greedy_policy_discrete(env, q_star)
    # print(f'Approximation of Q*: {q_star}')

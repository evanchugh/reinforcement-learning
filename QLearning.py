"""
Tabular Q-Learning methods.

Off-policy.
"""

import numpy as np
import gym

from util import N_EPISODES, MAX_TS_PER_EPISODE, epsilon_greedy_discrete, follow_greedy_policy_discrete


def q_learning(env, alpha=0.1, gamma=0.9, epsilon=0.5):
    """
    Returns approximation of Q*(s,a). Uses an epsilon-greedy policy derived from Q.
    alpha: learning rate
    gamma: discount rate
    epsilon: probability of taking an exploratory action
    """

    # arbitrary initialization
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for e in range(N_EPISODES):

        s = env.reset()
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


def double_q_learning(env, alpha=0.1, gamma=0.9, epsilon=0.5):

    Q1 = np.zeros((env.observation_space.n, env.action_space.n))
    Q2 = np.zeros((env.observation_space.n, env.action_space.n))

    for e in range(N_EPISODES):

        s = env.reset()

        for i in range(MAX_TS_PER_EPISODE):
            # env.render()

            sumQ = Q1 + Q2
            a = epsilon_greedy_discrete(sumQ, s, env, epsilon)

            s_, reward, done, _ = env.step(a)

            if np.random.random() <= 0.5:
                # update Q1
                Q1[s, a] += alpha * (reward + gamma * Q2[s_, np.argmax(Q1[s_])] - Q1[s, a])
                pass
            else:
                # update Q2
                Q2[s, a] += alpha * (reward + gamma * Q1[s_, np.argmax(Q2[s_])] - Q2[s, a])

            s = s_

            if done:
                break

    # env.render()
    env.close()

    return Q1 + Q2


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0', is_slippery=False)

    # q_star1 = q_learning(env)  # approximation
    # follow_greedy_policy(env, q_star1)
    # print(f'Approximation of Q*: {q_star1}')

    q_star2 = double_q_learning(env)
    follow_greedy_policy_discrete(env, q_star2)
    # print(f'Approximation of Q*: {q_star2}')

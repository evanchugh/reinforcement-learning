import numpy as np
import gym

from util import create_policy, N_EPISODES, TS_PER_EPISODE


def TD0(env, policy=None, alpha=0.01, gamma=1.0):
    """
    Given a fixed, stochastic policy, estimate the Value function associated with that policy.
    Returns estimate of V_pi(s), for all s in S.
    """

    if policy is None:
        policy = create_policy(env)

    v_pi = [0] * env.observation_space.n

    for e in range(N_EPISODES):

        obs = env.reset()

        for t in range(TS_PER_EPISODE):

            # env.render()

            prev_state = obs

            action = np.random.choice(list(policy[obs].keys()), p=list(policy[obs].values()))

            obs, reward, done, info = env.step(action)

            v_pi[prev_state] += alpha * (reward + gamma * v_pi[obs] - v_pi[prev_state])

            if done:
                # print(f'Episode {e} finished after {t + 1} timesteps')
                # print(f'Visited: {visited}')
                # print(f'Memory: {memory}')
                break

    # env.render()
    env.close()
    return v_pi


if __name__ == '__main__':

    env = gym.make('FrozenLake-v0', is_slippery=False)

    est_v = TD0(env)
    print(f'Estimated value-function for policy: {est_v}')
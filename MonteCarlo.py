import numpy as np
import gym

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

N_EPISODES = 1000
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


def first_visit_monte_carlo_policy_evaluation(env, policy=None):

    """
    Given a fixed, stochastic policy, estimate the Value function associated with that policy.
    Returns estimate of V_pi(s), for all s in S
    """

    if policy is None:
        policy = create_policy(env)

    n_visits = [0] * env.observation_space.n
    returns = [0] * env.observation_space.n
    v_pi = [0] * env.observation_space.n
    visited = []

    for e in range(N_EPISODES):

        memory = []  # each element is a list of the form [s, a, r]

        obs = env.reset()

        for t in range(TS_PER_EPISODE):

            if obs not in visited:
                visited.append(obs)

            sar = [obs]  # state, action, reward

            # env.render()

            action = np.random.choice(list(policy[obs].keys()), p=list(policy[obs].values()))
            sar.append(action)

            obs, reward, done, info = env.step(action)
            sar.append(reward)

            memory.append(sar)

            if done:
                # print(f'Episode {e} finished after {t + 1} timesteps')
                # print(f'Visited: {visited}')
                # print(f'Memory: {memory}')
                break

        returns_through_end_of_episode = 0

        for t, (s, a, r) in enumerate(memory[::-1]):  # goes from T-1 to 0
            returns_through_end_of_episode += r

            earlier_states = [sar[0] for sar in memory[0:t]]

            if s not in earlier_states:
                returns[s] += returns_through_end_of_episode
                n_visits[s] += 1

    for i, state_value in enumerate(v_pi):
        if n_visits[i] != 0:
            v_pi[i] = returns[i] / n_visits[i]

    # env.render()
    env.close()
    return v_pi


def every_visit_monte_carlo_policy_evaluation(env, policy=None):

    """
    Given a fixed, stochastic policy, estimate the Value function associated with that policy.
    Returns estimate of V_pi(s), for all s in S
    """

    if policy is None:
        policy = create_policy(env)

    n_visits = [0] * env.observation_space.n
    returns = [0] * env.observation_space.n
    v_pi = [0] * env.observation_space.n

    for e in range(N_EPISODES):

        memory = []  # each element is a list of the form [s, a, r]

        obs = env.reset()

        for t in range(TS_PER_EPISODE):

            sar = [obs]  # state, action, reward

            # env.render()

            action = np.random.choice(list(policy[obs].keys()), p=list(policy[obs].values()))
            sar.append(action)

            obs, reward, done, info = env.step(action)
            sar.append(reward)

            memory.append(sar)

            if done:
                # print(f'Episode {e} finished after {t + 1} timesteps')
                # print(f'Memory: {memory}')
                break

        returns_through_end_of_episode = 0

        for t, (s, a, r) in enumerate(memory[::-1]):  # goes from T-1 to 0

            returns_through_end_of_episode += r

            returns[s] += returns_through_end_of_episode
            n_visits[s] += 1

    for i, state_value in enumerate(v_pi):
        if n_visits[i] != 0:
            v_pi[i] = returns[i] / n_visits[i]

    # env.render()
    env.close()
    return v_pi


if __name__ == '__main__':

    env = gym.make('FrozenLake-v0', is_slippery=False)

    # print_env_info(env)
    est_v = every_visit_monte_carlo_policy_evaluation(env)
    print(f'Estimated value-function for policy: {est_v}')


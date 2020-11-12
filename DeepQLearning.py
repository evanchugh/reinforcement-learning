import random
import numpy as np
from collections import deque
import datetime
# import tensorflow as tf
# from tensorflow.keras import layers
import keras
from keras import layers
from tensorflow.keras.callbacks import TensorBoard
import gym

from util import epsilon_greedy, follow_greedy_policy

# HYPERPARAMETERS
REPLAY_MEMORY_SIZE = 1_000_000

EPSILON_START = 0.95  # probability of taking a random action
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.95  # every episode

GAMMA = 0.95
ALPHA = 0.01
MINIBATCH_SIZE = 20
TARGET_NETWORK_UPDATE_INTERVAL = 5  # episodes


def get_cartpole_model(env):
    model = keras.models.Sequential([
        layers.Dense(24, activation='relu', input_shape=(4,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(env.action_space.n, activation='linear')
    ])

    model.compile(optimizer=keras.optimizers.Adam(lr=ALPHA), loss='mse')
    return model


def train(replay_memory, policy_network, target_network):
    if len(replay_memory) >= MINIBATCH_SIZE:
        minibatch = random.sample(replay_memory, MINIBATCH_SIZE)

        # state_mb = []
        # q_val_mb = []

        for (s, a, r, s_, terminal) in minibatch:

            # state_mb.append(s)

            # s = s.reshape(-1, 4)
            # s_ = s_.reshape(-1, 4)

            q_values = policy_network.predict(s)[0]
            max_q_value = np.amax(target_network.predict(s_)[0])  # target q-value for next state

            if terminal:
                q_values[a] = r

            else:
                q_values[a] = r + GAMMA * max_q_value

            q_values = q_values.reshape(-1, 2)

            policy_network.fit(s, q_values, verbose=0)

            # q_val_mb.append(q_values)

        # state_mb = np.array(state_mb)
        # q_val_mb = np.array(q_val_mb)

        # policy_network.fit(state_mb, q_val_mb, epochs=1, verbose=0)

def act(state, epsilon, policy_network):
    if np.random.rand() < epsilon:
        return random.randrange(0, 1)
    q_values = policy_network.predict(state)
    return np.argmax(q_values[0])


def deep_q_learning_cartpole(env, n_episodes=100, max_timesteps=500):
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    policy_network = get_cartpole_model(env)
    target_network = get_cartpole_model(env)  # updated occasionally

    target_network.set_weights(policy_network.get_weights())

    epsilon = EPSILON_START

    for e in range(n_episodes):

        s = env.reset()
        s = s.reshape(-1, 4)

        for t in range(max_timesteps):

            a = act(s, epsilon, policy_network)

            s_, r, done, _ = env.step(a)
            s_ = s_.reshape(-1, 4)

            if done:
                r = -r

            replay_memory.append((s, a, r, s_, done))

            s = s_

            if done:
                print(f'Run {e + 1}, Exploration {epsilon}, score {t}')
                break

            train(replay_memory, policy_network, target_network)
            # experience_replay(replay_memory, policy_network)

            # epsilon decay
        if epsilon >= EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        # update target network
        if e % TARGET_NETWORK_UPDATE_INTERVAL == 0:
            target_network.set_weights(policy_network.get_weights())

    env.close()

    return policy_network


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    # policy = deep_q_learning_cartpole(env)
    # policy.save('cartpole_model')

    # follow_greedy_policy(env, get_cartpole_model(env))
    follow_greedy_policy(env, keras.models.load_model('models/cartpole_model'))

import random
import numpy as np
from queue import Queue
import datetime
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import gym

from util import epsilon_greedy, follow_greedy_policy

# HYPERPARAMETERS
REPLAY_MEMORY_SIZE = 1024

EPSILON_START = 0.9  # probability of taking a random action
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9  # every episode

GAMMA = 0.9
MINIBATCH_SIZE = 16
TARGET_NETWORK_UPDATE_INTERVAL = 2  # episodes


def get_cartpole_model(env):
    model = tf.keras.models.Sequential([
        layers.Dense(24, activation='relu', input_shape=(4,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(env.action_space.n, activation='linear')
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10_000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    return model


def train(replay_memory, policy_network, target_network):

    if len(replay_memory.queue) >= MINIBATCH_SIZE:
        minibatch = random.sample(replay_memory.queue, MINIBATCH_SIZE)

        state_mb = []
        q_val_mb = []

        for (s, a, r, s_, terminal) in minibatch:

            state_mb.append(s)

            s = s.reshape(-1, 4)
            s_ = s_.reshape(-1, 4)

            q_values = policy_network.predict(s)[0]
            max_q_value = np.amax(target_network.predict(s_)[0])  # target q-value for next state

            if terminal:
                q_values[a] = r

            else:
                q_values[a] = r + GAMMA * max_q_value

            q_val_mb.append(q_values)

        state_mb = np.array(state_mb)
        q_val_mb = np.array(q_val_mb)

        policy_network.fit(state_mb, q_val_mb, epochs=1, verbose=0)


def deep_q_learning_cartpole(env, n_episodes=100, max_timesteps=200):
    replay_memory = Queue(maxsize=REPLAY_MEMORY_SIZE)

    policy_network = get_cartpole_model(env)
    target_network = get_cartpole_model(env)    # updated occasionally

    target_network.set_weights(policy_network.get_weights())

    epsilon = EPSILON_START

    for e in range(n_episodes):

        s = env.reset()

        print(f'Training on episode {e}...')

        for t in range(max_timesteps):

            a = epsilon_greedy(policy_network, s, env, epsilon)

            s_, r, done, _ = env.step(a)

            replay_memory.put(np.array([s, a, r, s_, done]))

            train(replay_memory, policy_network, target_network)

            s = s_

            if done:
                break

        # epsilon decay
        if epsilon >= EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        # update target network
        if e % TARGET_NETWORK_UPDATE_INTERVAL == 0:
            target_network.set_weights(policy_network.get_weights())

    env.close()

    return policy_network


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    policy = deep_q_learning_cartpole(env)
    follow_greedy_policy(env, policy)

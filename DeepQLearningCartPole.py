"""
Deep Q-Learning implementation for the CartPole environment. This concept allows the agent to
make decisions in states it has never exactly experienced through the use of function approximation
with neural networks. This approach is useful for environments where the state space is extremely
large or continuous.
"""

import numpy as np
import random
from datetime import datetime
from collections import deque
import gym
from gym import wrappers
import keras
from keras import layers

from util import epsilon_greedy, follow_greedy_policy

MODEL_SAVEPATH = f'./models/CartPoleDeepQLearning-{datetime.now().strftime("%m.%d.%Y")}'

model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=MODEL_SAVEPATH,
    monitor='acc',
    mode='auto',
    save_best_only=True,
    verbose=0)

# HYPERPARAMETERS
REPLAY_MEMORY_SIZE = 1_000_000

EPSILON_START = 0.95  # probability of taking a random action
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.95  # every episode

GAMMA = 0.95
ALPHA = 0.01
MINIBATCH_SIZE = 16
TARGET_NETWORK_UPDATE_INTERVAL = 5  # episodes


def get_cartpole_model(env):
    model = keras.models.Sequential([
        layers.Dense(24, activation='relu', input_shape=(4,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(env.action_space.n, activation='linear')
    ])

    model.compile(optimizer=keras.optimizers.Adam(lr=ALPHA), loss='mse', metrics=['acc'])
    return model


def train(replay_memory, policy_network, target_network):
    if len(replay_memory) >= MINIBATCH_SIZE:
        minibatch = random.sample(replay_memory, MINIBATCH_SIZE)

        state_mb = []
        q_val_mb = []

        for (s, a, r, s_, terminal) in minibatch:

            state_mb.append(s)

            q_values = policy_network.predict(s)[0]
            max_q_value = np.amax(target_network.predict(s_)[0])  # target q-value for next state

            if terminal:
                q_values[a] = r

            else:
                q_values[a] = r + GAMMA * max_q_value

            q_val_mb.append(q_values)

        state_mb = np.array(state_mb).reshape(MINIBATCH_SIZE, -1)
        q_val_mb = np.array(q_val_mb).reshape(MINIBATCH_SIZE, -1)

        policy_network.fit(state_mb, q_val_mb, callbacks=[model_checkpoint], verbose=0)


def deep_q_learning_cartpole(env, pretrained_model=None, n_episodes=150, max_timesteps=500):
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    policy_network = get_cartpole_model(env)
    target_network = get_cartpole_model(env)  # updated occasionally

    if pretrained_model:
        policy_network.set_weights(pretrained_model.get_weights())

    target_network.set_weights(policy_network.get_weights())

    epsilon = EPSILON_START

    for e in range(n_episodes):

        s = env.reset()
        s = s.reshape(-1, 4)

        for t in range(max_timesteps):

            a = epsilon_greedy(policy_network, s, env, epsilon)

            s_, r, done, _ = env.step(a)
            s_ = s_.reshape(-1, 4)

            r = t

            if done:
                r = -r

            replay_memory.append((s, a, r, s_, done))

            s = s_

            train(replay_memory, policy_network, target_network)

            if done:
                print(f'Episode: {e + 1:<4} \t Score: {t:<3}')
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
    env = gym.make('CartPole-v1')
    # env = wrappers.Monitor(env, './demos/CartPole/')

    policy = deep_q_learning_cartpole(env)

    # policy = keras.models.load_model('./models/CartPoleDeepQLearning-11.12.2020')

    # follow_greedy_policy(env, policy)

"""
Deep Q-Learning implementation for the Atari game Breakout.
"""

import numpy as np
import random
from datetime import datetime
from collections import deque
import gym
from gym import wrappers
import keras
from keras import layers

from util import epsilon_greedy, follow_greedy_policy, print_env_info

MODEL_SAVEPATH = f'models/BreakoutDeepQLearning-{datetime.now().strftime("%m.%d.%Y")}'

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

def get_breakout_model(env):
    pass

def deep_q_learning_breakout(env, pretrained_model=None, n_episodes=5, max_timesteps=500):

    for e in range(n_episodes):

        s = env.reset()

        for t in range(max_timesteps):
            env.render()

            a = 2

            s_, r, done, _ = env.step(a)

            if done:
                print('done')

            s = s_


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    # env = wrappers.Monitor(env, './demos/Breakout/')

    # policy = deep_q_learning_breakout(env)
    print_env_info(env)


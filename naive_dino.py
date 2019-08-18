import gym
import gym_chrome_dino

import numpy as np
from PIL import Image
from time import sleep

env = gym.make('ChromeDino-v0')

while True:
    observation = env.reset()
    done = False

    while not done:
        # Choose an action according to the model

        if np.random.random() > 0.1:
            action = 0
        else:
            action = 1

        sleep(5)

        observation, reward, done, info = env.step(action)

        img = observation.reshape((150, 600, 3))
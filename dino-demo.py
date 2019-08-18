"""

Used to visualize the behavior of a trained dino in Chrome browser.


author: Maxime Darrin
"""
from keras.models import load_model
from keras import backend as K

import numpy as np



if __name__ == "__main__":
    # Load the model
    model = load_model("best_dino.chpkt")

    import gym
    import gym_chrome_dino

    env = gym.make('ChromeDino-v0')

    while True:
        observation = env.reset()
        done = False

        while not done:
            # Choose an action according to the model
            action = np.argmax(model.predict(np.array([observation / 255]))[0])
            observation, reward, done, info = env.step(action)

        print(env.unwrapped.game.get_score())

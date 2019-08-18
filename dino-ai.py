"""
author: Maxime Darrin
"""

import os

# Used to enforce the use of the CPU (since we have different processes, which use a keras model, we cannot share the
# cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Our Evoluation Strategy library
import ImprovedES as IES

import cv2
from PIL import Image

import numpy as np
from dinomodels import dinoModelStructureDense, dinoModelStructure, smallDino

class evolvingDino(IES.IESLearner):
    """
    Implement the evolution learning environement for the Chrome Dino. Sequential.
    """

    @staticmethod
    def evaluation_function(specimen, model_structure):
        import gym
        import gym_chrome_dino
        env = gym.make('ChromeDinoNoBrowser-v0')

        modelStruct = model_structure()
        model = modelStruct.model_from_parameters(specimen)

        score = 0
        for i in range(10):
            observation = env.reset()
            done = False

            while not done:
                action = np.argmax(model.predict(np.array([observation / 255]))[0])
                observation, reward, done, info = env.step(action)
                score += env.unwrapped.game.get_score()

        return float(score) / 10

def preprocess(observation):
    observation = observation.reshape((150, 600, 3))

    observation = cv2.resize(observation, (200, 50))

    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)

    return observation.reshape((50, 200, 1))

class parallelDino(IES.parallelIES):
    """
    Implement the evolution learning environement for the Chrome Dino. Parallel.
    """

    @staticmethod
    def parallel_evaluation(q, ids, specimens, model_structure):
        """
        Implementation of the parallel evaluation function for the dino. It fills the queue with (specimen_id, score)
        :param q: A queue in which we store the results
        :param ids: bulk of specimen ids to evaluate
        :param specimens: the specimen to evaluate
        :param model_structure: A class which provide hint about the model we are using
        :return: None
        """
        # This is the dino environement
        import gym
        import gym_chrome_dino
        env = gym.make('ChromeDino-v0')

        modelStruct = model_structure()

        for specimen, id in zip(specimens, ids):
            # For each specimen we build the model from its weights
            model = modelStruct.model_from_parameters(specimen)
            score = 0
            # We compute its scores on ten games and use the mean as the score of this specimen
            for i in range(10):
                observation = env.reset()

                done = False

                while True:
                    # Choose an action: the action with highest probability
                    action = np.argmax(model.predict(np.array([preprocess(observation) / 255]))[0])
                    observation, reward, done, info = env.step(action)

                    if done:
                        reward *= 10

                    if done:
                        score += env.unwrapped.game.get_score()
                        break
            # Storing the result
            q.put((id, float(score) / 10))

        env.close()
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Used the Improved evolving strategy to train a dino.")
    parser.add_argument("-p", "--population", default=128, type=int, help="Size of the population")
    parser.add_argument("-c", "--core", type=int, default=32, help="Number of process to use in case of parallelization")
    parser.add_argument("-e", "--epoch", type=int, default=100, help="Number of evolution epoch")
    parser.add_argument("savefile", type=str, default="model.chkpt", help="Where to save the model")

    args = parser.parse_args()
    # Create a pack of dino to train
    dinoPack = parallelDino(smallDino, population_size=args.population, core_number=args.core)

    # Train for N epoch
    dinoPack.train(args.epoch)

    # Get the best dino
    best = np.argmax(dinoPack.last_score)
    # Save it
    bestModel = smallDino.model_from_parameters(dinoPack.population[best])
    bestModel.save(args.savefile)

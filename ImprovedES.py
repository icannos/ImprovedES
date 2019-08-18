"""
author: Maxime Darrin
"""

import numpy as np

from multiprocessing import Pool, Queue
import multiprocessing as mp
from threading import Thread


def split_list(l, n):
    result = [[] for i in range(n)]
    for i in range(len(l)):
        result[i % n].append(l[i])

    return result


class ModelStructure:
    """
    This is an ABSTRACT class, you need to override it to define your own model structure.

    Methods
    ------------
    init_parameters unit --> array like (which supports to be used as mean in numpy.random.normal)
    model_from_parameters theta --> a compiled, usable model. (Basically, it uses the parameters
    to initialize the model)
    distance theta1 theta2 --> real number given two model parameters, returns their distances
    """

    def __init__(self):
        pass

    @staticmethod
    def init_parameters():
        raise NotImplemented

    @staticmethod
    def model_from_parameters(theta):
        raise NotImplemented

    @staticmethod
    def distance(theta1, theta2):
        pass

class IESLearner:
    """
    This is an ABSTRACT class, it needs to be subclassed.
        Implement an improved evolution strategy learner. Using species partitioning and potential mesure to allocate
    ressources.

    Methods to override:
    --------------------
    evaluation_function specimen, model_structure --> score

    In order to use this class you also need to write your own evaluation function and provide a model_structure
    object which supports:
    * init_parameters unit --> parameters for the model (array like, should support to be used in np.random.normal as
    mean)
    * model_from_parameters parameters --> model
    * distance (weights a, weights b) --> distance > 0 (Used to compare 2 specimen)
    """

    def __init__(self, model_structure, population_size=100, species_threshold=5000, stagnation_max=5):
        """

        :param model_structure: Object which specified the structure of the model (for example the architecture of the
        neural network. It should provide a random initialization method (init_parameters) which returns the weights
        and a method to get an initialized model from weights.
        :param population_size: Maximal size of the populatio used during the training
        :param species_threshold: The distance max between specimen to be considered as in the same species.
        :param stagnation_max: The number of epoch that we accept to run for a species which does not improve anymore
        """

        self.population_size = population_size
        self.species_threshold = species_threshold
        self.stagnation_max = stagnation_max

        self.model_structure = model_structure

        self.population = [model_structure.init_parameters() for i in range(population_size)]

        self.speciesId = [i for i in range(self.population_size)]

        self.last_score = np.zeros(population_size, dtype=np.int)

        self.species_best_score = [0 for i in range(self.population_size)]
        self.species_stagnation = [0 for i in range(self.population_size)]
        self.species_representant = []

    def potential(self, species):
        """
        Computes the potential of a species. It is basically proportional to the score of the better specimen of the
        species times a function which go from 1 to 0 as the number of epoch without improvement reach the
        stagnation_max parameter. See the internship report for more precision.

        :param species: id of the species
        :return: The potential score of this species
        """
        p = float(self.species_best_score[species]) * (1.0 - float(self.species_stagnation[species]) / self.stagnation_max)

        if p <= 0:
            return 0.
        else:
            return p

    def run_evaluation(self):
        """
        Evaluates each specimen using the evaluation_function method.
        You should override run_evaluation to parallelize the process.
        :return: None
        """
        for specimen in range(self.population_size):
            score = self.evaluation_function(self.population[specimen], self.model_structure)
            self.last_score[specimen] = score

    def sample_population(self, sigma):
        """
        Use the potential score of the population to generate a new one from the representants of each species. It uses
        a normal law, centered on the representant and of standard deviation sigma
        :param sigma: standard deviation
        :return: the potential of each sub population
        """
        potential = []
        tot_potential = 0
        population_size = 0

        # We copy the species representant before we generate a new generation
        weight_representant = [self.population[i] for i in self.species_representant]

        # For each representant we compute the potential of its species
        for i in range(len(self.species_representant)):
            p = self.potential(i)
            potential.append(p)
            tot_potential += p

        # For each species from the previous generation we will reproduce it into the new one
        for i in range(len(self.species_representant)):
            p = potential[i]

            # We compute the number of specimen allowed to this species
            species_size = int(np.trunc(self.population_size * (float(p) / tot_potential)))

            for c in range(species_size - 1):
                # We create the specimen of this species for the next generation
                self.population[population_size + c] = self.mutate(weight_representant[i], sigma)
                self.speciesId[population_size + c] = i
            # The former best is added to ensure that there is no regression
            self.population[population_size + species_size - 1] = weight_representant[i]

            # The population has grown
            population_size += species_size

        # If we did not populate the whole population we add extra specimen in the best species
        if population_size < self.population_size:
            for i in range(len(self.species_representant)):
                if not (population_size < self.population_size):
                    break
                self.population[population_size] = self.mutate(weight_representant[i], sigma)
                self.speciesId[population_size] = i
                population_size += 1

        # Return the potential of each species of this population
        return potential

    def mutate(self, specimen, sigma):
        """
        Make a small mutation on a specimen
        :param specimen: reference specimen (its weights list)
        :param sigma: standard deviation
        :return: new specimen randomly choosen around specimen
        """
        lw = []

        for w in specimen:
            lw.append(np.random.normal(w.astype(np.float64), sigma))

        return lw

    def species_discovery(self):
        """
        This updates the species that we consider during our training. We iterate on the specimen from the best to the
        worst. It becomes the representant of a new species if it does not in a species. If it becomes a representant,
        then every specimen genetically close to it (which means distance is below the species_threshold) is put in the
        same species. This allow to create more diversity and better ressources allocation.
        :return: None
        """

        # We get the ids of the best specimen
        sortedIdx = np.argsort(self.last_score)

        # We initialize our new species affectation and we keep the visited specimen
        new_species_id = np.array([i for i in range(self.population_size)], dtype=np.int8)
        visited = np.zeros(self.population_size, dtype=np.int8)

        # The id of our species
        species_count = 0
        species_representants = []

        # We iterate over the specimen of the last generation in descendant score order
        for i in range(1, self.population_size + 1):
            # If it doest not have bee taken
            if not visited[sortedIdx[-i]]:
                # It becomes a representant and a member of this new species
                new_species_id[i - 1] = species_count
                species_representants.append(sortedIdx[-i])

                # This one has been visited
                visited[sortedIdx[-i]] = 1

                # We check the specimen which could be in the same species
                for j in range(i + 1, self.population_size + 1):
                    if not visited[sortedIdx[-j]]:
                        # If they are close, they are in the same species
                        if self.model_structure.distance(self.population[sortedIdx[-i]],
                                                         self.population[sortedIdx[-j]]) < self.species_threshold:
                            new_species_id[sortedIdx[-j]] = species_count
                            visited[sortedIdx[-j]] = 1

                # We update our species counte
                species_count += 1

        new_species_best_score = []
        new_species_stagnation = []

        # We then compute the stagnation and the best score of each species
        for r in species_representants:
            former_species_id = self.speciesId[r]

            if self.last_score[r] > self.species_best_score[former_species_id]:
                new_species_best_score.append(self.last_score[r])
                new_species_stagnation.append(0)
            else:
                new_species_best_score.append(self.species_best_score[former_species_id])
                new_species_stagnation.append(1 + self.species_stagnation[former_species_id])

        # Updating our data
        self.species_best_score = new_species_best_score
        self.species_stagnation = new_species_stagnation
        self.species_representant = species_representants

    def train(self, epoch=50):
        """
        Train our population for a given number of epoch
        """

        for i in range(epoch):
            print("Generation " + str(i))

            self.run_evaluation()
            self.species_discovery()
            potential = self.sample_population(1)

            print("Best Score: " + str(np.max(self.last_score)))
            print("Potentials: ", potential)

    @staticmethod
    def evaluation_function(specimen, model_structure):
        """
        Should be override with an env-wise evaluation function
        :param specimen:
        :param model_structure:
        :return:
        """
        score = 2
        return score


class parallelIES(IESLearner):
    """
    A fully parallel implementation of the Improved Evolution Strategy

    This is an ABSTRACT class, it needs to be subclassed.
        Implement an improved evolution strategy learner. Using species partitioning and potential mesure to allocate
    ressources.

    Methods to override:
    --------------------
    parallel_evaluation(q, ids, specimens, model_structure)

    In order to use this class you also need to write your own evaluation function and provide a model_structure
    object which supports:
    * init_parameters unit --> parameters for the model (array like, should support to be used in np.random.normal as
    mean)
    * model_from_parameters parameters --> model
    * distance (weights a, weights b) --> distance > 0 (Used to compare 2 specimen)

    """

    def __init__(self, model_structure, population_size=100, species_threshold=5000, stagnation_max=5, core_number=32):
        IESLearner.__init__(self, model_structure, population_size, species_threshold, stagnation_max)
        """

        :param model_structure: Object which specified the structure of the model (for example the architecture of the
        neural network. It should provide a random initialization method (init_parameters) which returns the weights
        and a method to get an initialized model from weights.
        :param population_size: Maximal size of the populatio used during the training
        :param species_threshold: The distance max between specimen to be considered as in the same species.
        :param stagnation_max: The number of epoch that we accept to run for a species which does not improve anymore
        """

        self.core_number = core_number

    def parallel_mutation(self, ids, specimen, sigma, species):
        """
        This function should be used in a subthread
        :param ids: The set of specimen that this process should work with
        :param specimen: The representant specimen
        :param sigma: Standard deviation of the mutation
        :return: None
        """
        for i in ids:
            self.population[i] = self.mutate(specimen, sigma)
            self.speciesId[i] = species

        return

    def parallel_affectation(self, ids, i, new_species_id, species_count, visited, sortedIdx):
        """
        Affect specimen in their species in parallel. Should be used in a subthread
        :param ids: The set of specimen that this process should work with
        :param i: Id of the representant of the species
        :param new_species_id: Array containing the affectations specimen_id --> species_id
        :param species_count: The current species id
        :param visited: specimen --> bool true if already affected, false otherwise
        :param sortedIdx: The ids of the specimen, sorted in ascending order (on their score)
        :return:
        """
        for j in ids:
            if not visited[sortedIdx[-j]]:
                # If this specimen is close enough to the representant of the species, it is put in the same species
                if self.model_structure.distance(self.population[sortedIdx[-i]],
                                                 self.population[sortedIdx[-j]]) < self.species_threshold:
                    new_species_id[j-1] = species_count
                    # And it has been affected
                    visited[sortedIdx[-j]] = 1

        return

    def species_discovery(self):
        """
        Parallel version, It uses parallel_affectation

        This updates the species that we consider during our training. We iterate on the specimen from the best to the
        worst. It becomes the representant of a new species if it does not in a species. If it becomes a representant,
        then every specimen genetically close to it (which means distance is below the species_threshold) is put in the
        same species. This allow to create more diversity and better ressources allocation.
        :return: None
        """

        sortedIdx = np.argsort(self.last_score)

        new_species_id = np.array([i for i in range(self.population_size)], dtype=np.int8)
        visited = np.zeros(self.population_size, dtype=np.int8)

        species_count = 0
        species_representants = []

        for i in range(1, self.population_size + 1):
            if not visited[sortedIdx[-i]]:
                new_species_id[i - 1] = species_count
                species_representants.append(sortedIdx[-i])

                # This one has been visited
                visited[sortedIdx[-i]] = 1

                ids = split_list([j for j in range(i + 1, self.population_size + 1) if not visited[sortedIdx[-j]]],
                                 self.core_number)

                pool = []

                for id in ids:
                    p = Thread(target=self.parallel_affectation,
                               args=(id, i, new_species_id, species_count, visited, sortedIdx))
                    p.start()
                    pool.append(p)

                for p in pool:
                    p.join()

                species_count += 1

        new_species_best_score = []
        new_species_stagnation = []

        for r in species_representants:
            former_species_id = self.speciesId[r]

            a = self.last_score[r]
            b = self.species_best_score[former_species_id]

            if a > b:
                new_species_best_score.append(self.last_score[r])
                new_species_stagnation.append(0)
            else:
                new_species_best_score.append(self.species_best_score[former_species_id])
                new_species_stagnation.append(1 + self.species_stagnation[former_species_id])

        self.species_best_score = new_species_best_score
        self.species_stagnation = new_species_stagnation
        self.species_representant = species_representants

        print(len(species_representants))

        print("species discovery end")

        return

    def sample_population(self, sigma):
        """
        Parallel Version

        Use the potential score of the population to generate a new one from the representants of each species. It uses
        a normal law, centered on the representant and of standard deviation sigma
        :param sigma: standard deviation
        :return: the potential of each sub population
        """

        potential = []
        tot_potential = 0
        population_size = 0

        weight_representant = [self.population[i] for i in self.species_representant]

        # Computes the potential of each species
        for i in range(len(self.species_representant)):
            p = self.potential(i)
            potential.append(p)
            tot_potential += p

        for i in range(len(self.species_representant)):
            p = potential[i]
            # Computes the number of the specimen allocated to this species
            species_size = int(np.trunc(self.population_size * (float(p) / tot_potential)))

            m = int(float(species_size - 1) / self.core_number)
            r = m * self.core_number

            # Build index list for parallelism
            ids = [list(range(c * m, (c + 1) * m)) for c in range(self.core_number - 1)]
            ids.append(list(range((self.core_number - 1) * m, self.core_number * m + r)))

            pool = []
            # Exexutes on all core
            for c in range(self.core_number):
                pool.append(Thread(target=self.parallel_mutation, args=(ids[c], weight_representant[i], sigma, i)))

            for p in pool:
                p.start()
            for p in pool:
                p.join()

            # We keep the representant in the new population
            self.population[population_size + species_size - 1] = weight_representant[i]
            self.speciesId[population_size + species_size - 1] = i

            population_size += species_size

        # If we still have available specimen, we add them to some species
        if population_size < self.population_size:
            for i in range(len(self.species_representant)):
                if not (population_size < self.population_size):
                    break
                self.population[population_size] = self.mutate(weight_representant[i], sigma)
                self.speciesId[population_size] = i
                population_size += 1

        # Return the potential of each species of this new population
        return potential

    @staticmethod
    def parallel_evaluation(q, ids, specimens, model_structure):
        raise NotImplemented

    def train(self, epoch=50):

        for i in range(epoch):
            print("Generation " + str(i))

            self.parallel_run_evaluation()
            self.species_discovery()
            potential = self.sample_population(0.2)

            print("Best Score: " + str(np.max(self.last_score)))
            print("Potentials: ", potential)

    def parallel_run_evaluation(self):
        """
        Execute the actions of the specimen in parallel. It uses self.number_core process and split
        evenly the population
        between the processes.
        :return:
        """
        q = Queue()
        m = int(float(self.population_size) / self.core_number)
        r = self.population_size - m * self.core_number
        ids = list(range(self.population_size))

        pool = []
        for i in range(self.core_number - 1):
            pool.append(mp.Process(target=self.parallel_evaluation,
                                   args=(q, ids[i * m:(i + 1) * m],
                                         self.population[i * m:(i + 1) * m],
                                         self.model_structure
                                         )))

            pool[i].start()

        pool.append(mp.Process(target=self.parallel_evaluation,
                               args=(q, ids[(self.core_number - 1) * m: self.core_number * m + r],
                                     self.population[(self.core_number - 1) * m: self.core_number * m + r],
                                     self.model_structure
                                     )))

        pool[self.core_number - 1].start()

        for i in range(self.core_number):
            pool[i].join()

        for i in range(self.population_size):
            id, score = q.get()
            self.last_score[id] = score

import random

from tqdm import tqdm

from experiments.soup_exp import SoupExperiment
from functionalities_test import test_for_fixpoints


class MeltingSoupExperiment(SoupExperiment):

    def __init__(self, melt_chance, *args, keep_population_size=True, **kwargs):
        super(MeltingSoupExperiment, self).__init__(*args, **kwargs)
        self.keep_population_size = keep_population_size
        self.melt_chance = melt_chance

    def population_melt(self):
        # A network melting with another network by a given percentage
        if random.randint(1, 100) <= self.melt_chance:
            random_net1_idx, random_net2_idx, destroy_idx = random.sample(range(self.population_size), 3)
            random_net1 = self.population[random_net1_idx]
            random_net2 = self.population[random_net2_idx]
            print(f"\n Melt: {random_net1.name} -> {random_net2.name}")
            melted_network = random_net1.melt(random_net2)
            if self.keep_population_size:
                del self.population[destroy_idx]
            self.population.append(melted_network)

    def evolve(self):
        """ Evolving consists of attacking, melting & self-training. """

        loop_epochs = tqdm(range(self.epochs))
        for i in loop_epochs:
            loop_epochs.set_description("Evolving soup %s" % i)

            self.population_attack()

            self.population_melt()

            self.population_self_train()

            # Testing for fixpoints after each batch of ST steps to see relevant data
            if i % self.ST_steps == 0:
                test_for_fixpoints(self.fixpoint_counters, self.population)
                fixpoints_percentage = round(self.fixpoint_counters["identity_func"] / self.population_size, 1)
                self.fixpoint_counters_history.append(fixpoints_percentage)

            # Resetting the fixpoint counter. Last iteration not to be reset -
            #  it is important for the bar_chart_fixpoints().
            if i < self.epochs:
                self.reset_fixpoint_counters()

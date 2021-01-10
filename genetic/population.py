
class Population:
    
    def __init__(self, species, size=None, indiv_list=None, crossover_rate=0.3, mutation_rate=0.015, maximize=True):
        self.species = species
        self.maximize = maximize
        self.population_size = size if size != None else 0
        self.individuals = [self.species.random_init(crossover_rate, mutation_rate) for _ in range(size)] if indiv_list == None else indiv_list
        print("Initializing a random population. Size: {}".format(size))

    def add_individual(self, individual):
        assert type(individual) is self.species
        self.individuals.append(individual)
        self.population_size += 1

    def get_fittest(self, elite_size=1):
        for gen in self.individuals:
            if gen.fitness == None:
                gen.set_fitness()
        if self.maximize:
            return sorted(self.individuals, key=lambda x: x.fitness, reverse=True)[:elite_size]
        return sorted(self.individuals, key=lambda x: x.fitness, reverse=True)[:elite_size]

    def __getitem__(self, item):
        return self.individuals[item]

if __name__ == '__main__':
    pass
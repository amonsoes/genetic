import random

from a2c_cartpole import main as a2c_main
from a2c_cartpole import gym_room as gr
from a2c_cartpole import agent as a2c

# from ppo_cartpole import main as ppo_main
# from ppo_cartpole import agent as ppo

class Genome:

    def __init__(self, genes, crossover_rate, mutation_rate):
        self.genes = genes
        self.validate_genes()
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.fitness = None  # Until evaluated an individual fitness is unknown

    def validate_genes(self):
        """Check that genes are compatible with genome."""
        if set(self.genes.keys()) != set(self.__class__.genome):
            raise ValueError("Genes passed don't correspond to individual's genome.")
        return True
    
    def reproduce(self, partner):
        assert self.__class__ == partner.__class__  # Can only reproduce if they're the same species
        child_genes = {}
        for name, value in self.genes.items():
            if random.random() < self.crossover_rate:
                child_genes[name] = partner.genes[name]
            else:
                child_genes[name] = value
        return self.__class__(
            child_genes, self.crossover_rate, self.mutation_rate,
        )

    def crossover(self, partner):
        assert self.__class__ == partner.__class__  # Can only cross if they're the same species
        for name in self.genes.keys():
            if random.random() < self.crossover_rate:
                self.genes[name], partner.genes[name] = partner.genes[name], self.genes[name]
        self.fitness = None
        partner.fitness = None

    def mutate(self):
        """Mutate instance's genes with a certain probability."""
        for name in self.genes.keys():
            if random.random() < self.mutation_rate:
                possible_values = self.__class__.genome[name]
                self.genes[name] = random.choice(possible_values)   
        self.fitness = None

    def copy(self):
        """Copy instance."""
        individual_copy = self.__class__(
            self.genes.copy(), self.crossover_rate,
            self.mutation_rate
        )
        individual_copy.fitness = self.fitness
        return individual_copy

class A2CGenome(Genome):
    
    genome = {
        'gamma': [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90], 
        'alpha': [0.000001, 0.000005, 0.000007, 0.00001, 0.00005, 0.00007, 0.0001, 0.0005, 0.0007, 0.001, 0.005, 0.007, 0.01, 0.05, 0.07], 
        'beta': [0.000001, 0.000005, 0.000007, 0.00001, 0.00005, 0.00007, 0.0001, 0.0005, 0.0007, 0.001, 0.005, 0.007, 0.01, 0.05, 0.07], 
        'hidden_dim':[32, 64, 128, 200, 300, 400, 512, 600, 700, 800, 900, 1024]
        }
    
    room = gr.GymRoom('CartPole-v1') # before you call get_fitness for genome, set env with A2CGenome.room = GymRoom(env)
    
    def __init__(self,  genes, crossover_rate, mutation_rate):
        super().__init__(genes, crossover_rate, mutation_rate)
    
    @classmethod
    def random_init(cls, crossover_rate, mutation_rate):
        genes = {k:random.choice(cls.genome[k]) for k in cls.genome.keys()}
        return cls(genes, crossover_rate, mutation_rate)
    
    def set_fitness(self):
        agent = a2c.TDA2CLearner(gamma=self.genes['gamma'],
                               nr_actions=A2CGenome.room.num_actions_available(),
                               nr_outputs=2,
                               alpha=self.genes['alpha'],
                               beta=self.genes['beta'],
                               observation_dim=A2CGenome.room.env.observation_space.shape[0],
                               hidden_dim=self.genes['hidden_dim'])
        _, evaluation = a2c_main.a2c_main(agent, A2CGenome.room, 1000, 50)
        self.fitness = evaluation
        return evaluation
    
class PPOGenome:
    
    
    genome = {
        'gamma': [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90], 
        'alpha': [0.000001, 0.000005, 0.000007, 0.00001, 0.00005, 0.00007, 0.0001, 0.0005, 0.0007, 0.001, 0.005, 0.007, 0.01, 0.05, 0.07], 
        'beta': [0.000001, 0.000005, 0.000007, 0.00001, 0.00005, 0.00007, 0.0001, 0.0005, 0.0007, 0.001, 0.005, 0.007, 0.01, 0.05, 0.07], 
        'hidden_dim':[32, 64, 128, 200, 300, 400, 512, 600, 700, 800, 900, 1024], 
        'gae_lambda' : [0.90, 0.92, 0.94, 0.96, 0.98],
        'memory_batchsize': [16, 32, 64, 128],
        'policy_clip' : [0.1, 0.2, 0.3, 0.4]
        }
    
    room = gr.GymRoom('CartPole-v1')
    
    def __init__(self,  genes, crossover_rate, mutation_rate):
        super().__init__(genes, crossover_rate, mutation_rate)
    
    @classmethod
    def random_init(cls, crossover_rate, mutation_rate):
        genes = {k:random.choice(cls.genome[k]) for k in cls.genome.keys()}
        return cls(genes, crossover_rate, mutation_rate)
    
    def set_fitness(self):
        agent = ppo.PPOLearner(gamma=self.genes['gamma'],
                               nr_actions=A2CGenome.room.num_actions_available(),
                               nr_outputs=2,
                               alpha=self.genes['alpha'],
                               beta=self.genes['beta'],
                               observation_dim=A2CGenome.room.env.observation_space.shape[0],
                               hidden_dim=self.genes['hidden_dim'])
        _, evaluation = ppo_main.ppo_main(agent, A2CGenome.room, 1000, 50)
        self.fitness = evaluation
        return evaluation
        
if __name__ == '__main__':
    
    
    genes = {'gamma' : 0.99, 'alpha' : 0.0005, 'beta': 0.001, 'hidden_dim': 64}
    genes2 = {'gamma' : 0.98, 'alpha' : 0.0001, 'beta': 0.003, 'hidden_dim': 128}
    individual = A2CGenome(genes, 0.5, 0.5)
    individual2 = A2CGenome(genes2, 0.5, 0.5) 
    
    # test reproduce
    individual_repr = individual.reproduce(individual2)
    print('reproduction sucessfully tested...')
    
    # test cross
    indiv_copy = individual.copy()
    individual.crossover(individual2)
    print('crossover sucessfully tested...')
    
    # test mutate
    indiv_copy = individual.copy()
    individual.mutate()
    print('mutate sucessfully tested...')
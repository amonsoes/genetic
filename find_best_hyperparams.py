import argparse

from genetic.tournament import GeneticAlgorithm
from genetic.genome import LSTMGenome
from genetic.population import Population


def find_best_hyperparams(species, 
                          population_size, 
                          crossover_rate, 
                          mutation_rate, 
                          elitism, 
                          elite_size, 
                          maximize, 
                          generation_epochs):

    if species in ('lstm', 'LSTM'):
        population = Population(LSTMGenome, 
                size=population_size, 
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                maximize=maximize)
        
    # Add your species name here and call a population
    # if species in ('your species',...):
        
    else:
        assert argparse.ArgumentError('WRONG INPUT FOR SPECIES. SET TO LSTM')
        population = None
        
        
    selection = GeneticAlgorithm(population, elitism=elitism, elite_size=elite_size)
    selection.run(generation_epochs)
    with open('./best_hyperparams.txt', 'w') as w:
        w.write(str(selection.population.get_fittest()[0].genes))
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('species', type=str, help='set species', default='LSTM')
    parser.add_argument('--pop_size', type=int, default=50, help='set population rate')
    parser.add_argument('--cross_rate', type=float, default=0.3, help='set crossover rate')
    parser.add_argument('--mut_rate', type=float, default=0.015, help='set mutation rate')
    parser.add_argument('--elitism', type=lambda x : x=='True', default=True, help='set if best indivs get to live on')
    parser.add_argument('--elite_size', type=int, default=3, help='set how many best indivs get to live on')
    parser.add_argument('--maximize', type=lambda x : x=='True', default=True, help='should indivs be maximized or minimized')
    parser.add_argument('--gen_epochs', type=int, default=3, help='number of generations')
    
    args = parser.parse_args()
    
    find_best_hyperparams(species=args.species, 
                          population_size=args.pop_size, 
                          crossover_rate=args.cross_rate, 
                          mutation_rate=args.mut_rate, 
                          elitism=args.elitism, 
                          elite_size=args.elite_size, 
                          maximize=args.maximize, 
                          generation_epochs=args.gen_epochs)
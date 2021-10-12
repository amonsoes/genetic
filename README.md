

# Genetic Hyperparameter Optimization for Deep Learning Algorithms

This is a generic package to find the best training parameter for any given Deep Learning Algorithm.
The best HP's will be saved at: **./best_hyperparams.txt**

```
python3 find_best_hyperparams.py species --pop_size 50 --cross_rate 0.3 --mut_rate 0.015 --elitism True --elite_size 3 --maximize True --gen_epochs 7
```

#### customize your own Genome for your DL Algorithm

Steps:

1.) subclass Genome
2.) write **Class.genome** :set a genome dictionary with senisble hyperparameter values
3.) write **random_init()** : make a classmethod to randomly initialize the species
4.) wirte **set_fitness()** : make a method in which you run the DL training Algorithm and get an evaluation
        -> (add your script in this folder so you can actually use your classes in set_fitness)
5.) change find_best_hyperparameters.py such that a population can be initialized with your subclass

#### args:

- species : subclass your own Genome for your Deep Learning Algorithm
- cross_rate : set with what prob individuals cross their genes
- mut_rate : set with what prob individuals mutate
- elitism : set if good indivs progress generations without tournament
- elite_size : how many of those progress w/o tournament
- maximize : maximize fitness=True or minimize=False
- gen_epochs: set the generation number

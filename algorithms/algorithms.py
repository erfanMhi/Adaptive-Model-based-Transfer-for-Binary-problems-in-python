
import numpy as np

from utils.tools import Tools
from utils.fitness_util import fitness_eval
from probability_models.probability_model import ProbabilityModel

def BGA(problem, dims, th_best):
    """[bestSol, fitness_hist] = BGA(problem,dims,th_best): simple binary GA with
        uniform crossover and bit-flip mutation. 
       INPUT:
         problem: problem type, 'onemax', 'onemin', or 'trap5'
         dims: problem dimensionality
         th_best: global best fitness value (used for early stop to build
         probabilistic models)
        
       OUTPUT:
         bestSol: best solution
         fitness: history of best fitness for each generation"""
    bestSol = None
    all_models = Tools.load_from_file('all_models')
    
    pop = 200
    gen = 1000
    
    fitness_hist = np.zeros(gen)
    population = np.round(np.random.rand(pop,dims))    
    fitness = fitness_eval(population,problem,dims);
    buildmodel = True;
    fitness_hist[0] = max(fitness);
    print('Generation 0 best fitness = ',str(fitness_hist[0]));
    
    for i in range(1,gen):
        parent1 = population[np.random.permutation(pop),:];
        parent2 = population[np.random.permutation(pop),:];
        tmp = np.random.rand(pop,dims);        
        offspring = np.zeros((pop,dims));
        index = tmp>=0.5;
        offspring[index] = parent1[index];
        index = tmp<0.5;
        offspring[index] = parent2[index];
        tmp = np.random.rand(pop,dims);
        index = tmp<(1/dims);
        offspring[index] = np.abs(1-offspring[index]);
        cfitness = fitness_eval(offspring,problem,dims);
        interpop = np.append(population,offspring,0)
        interfitness = np.append(fitness,cfitness)
        index = np.argsort((-interfitness));
        interfitness = interfitness[index]
        fitness = interfitness[0:pop]
        interpop = interpop[index,:]        
        population = interpop[0:pop,:]
        fitness_hist[i] = fitness[0];
        print('Generation ', str(i), ' best fitness = ',str(fitness_hist[i])) 
 
        if (fitness[1] >= th_best or i == gen) and buildmodel:
            print('Building probablistic model...')
            model = ProbabilityModel('umd')
            model.buildmodel(population)
            all_models.append(model)
            Tools.save_to_file('all_models',all_models)
            print('Complete!')
            fitness_hist[i+1:] = fitness[0];
            bestSol = population[0, :];
            break
    return bestSol, fitness_hist
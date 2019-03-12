
import numpy as np

from utils.tools import Tools
from utils.fitness_util import fitness_eval
from probability_models.probability_model import ProbabilityModel
from probability_models.mixture_model import MixtureModel

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
        fitness = interfitness[:pop]
        interpop = interpop[index,:]        
        population = interpop[:pop,:]
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

def AMT_BGA(problem, dims, reps, trans):
    """[bestSol, fitness_hist, alpha] = TSBGA(problem, dims, reps, trans): Adaptive
    %Model-based Transfer Binary GA. The crossover and mutation for this simple
    %binary GA are uniform crossover and bit-flip mutation.
    %INPUT:
    % problem: problem type, 'onemax', 'onemin', or 'trap5'
    % dims: problem dimensionality
    % reps: number of repeated trial runs
    % trans:    trans.transfer: binary variable 
    %           trans.TrInt: transfer interval for AMT
    %
    %OUTPUT:
    % bestSol: best solution for each repetiion
    % fitness: history of best fitness for each generation
    % alpha: transfer coefficient
    """
    pop = 50
    gen = 100
    transfer = trans['transfer']
    if transfer:
        TrInt = trans['TrInt']
        all_models = Tools.load_from_file('all_models')
        
    fitness_hist = np.zeros((reps, gen))
    bestSol = np.zeros((reps, dims))
    alpha = [None] * (reps)
    
    for rep in range(reps):
        alpha_rep = []
        population = np.round(np.random.rand(pop, dims))
        fitness = fitness_eval(population,problem,dims)
        ind = np.argmax(fitness)
        best_fit = fitness[ind]
        print('Generation 0 best fitness = ',str(best_fit))
        fitness_hist[rep, 0] = best_fit
        
        for i in range(1,gen):
            if transfer and i % TrInt == 0:
                mmodel = MixtureModel(all_models)
                mmodel.createtable(population, True, 'umd')
                mmodel.EMstacking(); # Recombination of probability models
                mmodel.mutate(); # Mutation of stacked probability model
                offspring = mmodel.sample(pop)
                alpha_rep.append(mmodel.alpha)
                print('Transfer coefficient at generation ', str(i), ': ',str(mmodel.alpha))
                
            else:
                parent1 = population[np.random.permutation(pop),:]
                parent2 = population[np.random.permutation(pop),:]
                tmp = np.random.rand(pop,dims)       
                offspring = np.zeros((pop,dims))
                index = tmp>=0.5
                offspring[index] = parent1[index]
                index = tmp<0.5
                offspring[index] = parent2[index]
                tmp = np.random.rand(pop,dims)
                index = tmp<(1/dims)
                offspring[index] = np.abs(1-offspring[index])
                    
            cfitness = fitness_eval(offspring,problem,dims)
            interpop = np.append(population,offspring,0)
            interfitness = np.append(fitness,cfitness)
            index = np.argsort((-interfitness))
            interfitness = interfitness[index]
            fitness = interfitness[:pop]
            interpop = interpop[index,:]       
            population = interpop[:pop,:]
            print('Generation ', str(i), ' best fitness = ',str(np.max(fitness_hist)))
            fitness_hist[rep, i] = fitness[0]
         
        alpha[rep] = alpha_rep
        bestSol[rep, :] = population[ind, :]
    return bestSol, fitness_hist, alpha
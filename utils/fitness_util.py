import numpy as np

def fitness_eval(population,problem,dims):
    fitness = None
    if problem == 'onemax':
        fitness = np.sum(population,1);
    elif problem == 'onemin':
        fitness = dims - np.sum(population,1);
    elif problem == 'trap5':
        pop = population.shape[0];
        fitness = np.zeros(pop);
        index = np.arange(dims).reshape(-1,5)
        rows = index.shape[0]
        for i in range(pop):
            fitsum = 0;
            for j in range(rows):
                contri = sum(population[i,index[j,:]]);
                if contri == 5:
                    fitsum = fitsum+5;
                else:
                    fitsum = fitsum+(4-contri);
                
            fitness[i] = fitsum;
    else:
        raise Exception('Function not implemented.')
    return fitness
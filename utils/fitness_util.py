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

def knapsack_fitness_eval(population, problem, dims,pop): # Question about BV[l] changing in matlab in comparison with python
    fitness = np.zeros(pop);
    Weights = problem['w'];
    Profits = problem['p'];
    Ratios = Profits/Weights;
    for i in range(pop):
        BV = population[i, :] == 1;
        TotalWeight = np.sum(Weights[BV]);
        TotalProfit = np.sum(Profits[BV]);
        
        if TotalWeight > problem['cap']: # Repair solution
            selections = np.sum(BV)
            List = np.zeros((selections,2))
            counter = 0
            for j in range(dims):
                if BV[j] == 1:
                    List[counter,0] = Ratios[j]
                    List[counter,1] = int(j)
                    counter = counter + 1
                
                if counter >= selections:
                    break
            List = List[List[:,0].argsort()[::-1]]
            counter = selections-1
            while TotalWeight > problem['cap']:
                l = int(List[counter,1])
                BV[l] = 0 
                TotalWeight = TotalWeight - Weights[l]
                TotalProfit = TotalProfit - Profits[l]
                counter = counter - 1

        fitness[i] = TotalProfit
    return fitness
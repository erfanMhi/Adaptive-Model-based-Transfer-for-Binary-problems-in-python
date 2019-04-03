import numpy as np
from scipy.stats import multivariate_normal
from probability_models.probability_model import ProbabilityModel
from copy import deepcopy

class MixtureModel: # Works reliably for 2(+) Dimensional distributions
    """properties
    model_list; % cell array of ProbabilityModels
    alpha; % weights of the models in stacking of mixture models
    noms; % number of models    
    probtable; % Probability table required for stacking EM algorithm
    nsols; % number of solutions in probability table
    end"""
    def __init__(self, allmodels):
        self.model_list = deepcopy(allmodels)
        self.noms = len(allmodels)
        self.alpha = (1/self.noms)*np.ones(self.noms)
        
    def EMstacking(self):
        """[Determining the mixture weights of each model]
        """

        iterations = 100
        for i in range(iterations):
            talpha = deepcopy(self.alpha) #Probability of every model (mixture weights) $\alpha$ in the main paper
            probvector = np.dot(self.probtable, talpha.T)
            for j in range(self.noms):
                # according to this, if model j give a greater probability to each solution of target problem, it is more likely to have a greater alpha
                talpha[j] = sum((1/self.nsols)*talpha[j]*self.probtable[:, j]/probvector); 
            self.alpha = talpha
    
    def mutate(self):
        """[Adding noise to mixture weights ($\alpha$) and doing some 
            mathematical stuff to make the alpha suitable for probability theory axioms]
        """

        #print('modifalpha', self.alpha)
        modifalpha = self.alpha+np.random.normal(0,0.01,self.noms); # Determining std dev for mutation can be a parameteric study
        #print('modifalpha', modifalpha)
        pusum = np.sum(modifalpha);
        if pusum == 0: # Then complete weightage assigned to target model alone
            self.alpha = np.zeros(self.noms)
            self.alpha[-1] = 1
        else:
            self.alpha = modifalpha/pusum

    def sample(self, nos):
        #print(self.alpha)
        indsamples = np.ceil(nos*self.alpha);
        #print('indsamples', indsamples)
        totalsamples = int(np.sum(indsamples));
        if indsamples[0] <= 0:
            solutions = self.model_list[0].sample(0); # added for np.append to work
        else:
            solutions = self.model_list[0].sample(indsamples[0]);
        #print('solutions_0 ', solutions.shape)
        for i in range(1, self.noms):
            if indsamples[i] <= 0:
                continue;
            else:
                sols = self.model_list[i].sample(indsamples[i]);
                solutions = np.append(solutions, sols, 0)
        #print('solutions_n ', solutions.shape)
        #print('totalsamples ', totalsamples)
        #print('np.random.permutation(totalsamples)', np.random.permutation(totalsamples).shape)
        solutions = solutions[np.random.permutation(totalsamples),:];
        solutions = solutions[:nos,:];
        return solutions
    
    def createtable(self,solutions, CV, c_type):
        if CV:       
            self.noms = self.noms+1 # NOTE: Last model in the list is the target model
            #print('self.noms', self.noms)
            self.model_list.append(ProbabilityModel(c_type))
            #print('len(self.model_list)', len(self.model_list))
#             self.model_list[self.noms] = ProbabilityModel(c_type)       
            self.model_list[self.noms-1].buildmodel(solutions)
            self.alpha = (1/self.noms)*np.ones(self.noms);
            nos = solutions.shape[0];
            self.probtable = np.ones((nos, self.noms));
            for j in range(self.noms-1): # Calculating of the assigned probability of each source model to target solutions
                self.probtable[:, j] = self.model_list[j].pdfeval(solutions);
            for i in range(nos): # Leave-one-out cross validation scheme
                x = np.append(solutions[:i,:], solutions[i+1:,:], 0)
                tmodel = ProbabilityModel(c_type);
                tmodel.buildmodel(x);
                self.probtable[i,self.noms-1] = tmodel.pdfeval(solutions[i,:].reshape(1,-1));
                    
        else:
            nos = solutions.shape[0];
            self.probtable = np.ones((nos, self.noms));
            for j in range(self.noms):
                self.probtable[:, j] = self.model_list[j].pdfeval(solutions);
        self.nsols = nos
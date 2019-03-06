import numpy as np
from scipy.stats import multivariate_normal

class ProbabilityModel: # Works reliably for 2(+) Dimensional distributions
    """ properties
        modeltype; % multivariate normal ('mvarnorm' - for real coded) or univariate marginal distribution ('umd' - for binary coded)    
        mean_noisy;
        mean_true;
        covarmat_noisy;
        covarmat_true;
        probofone_noisy;
        probofone_true;
        probofzero_noisy;
        probofzero_true;    
        vars;
      end""" 
#  methods (Static)
    def __init__(self, modeltype):
        self.modeltype = modeltype
        
    def sample(self, nos):
        if self.modeltype == 'mvarnorm':
            solutions = np.random.multivariate_normal(self.mean_true,self.covarmat_true,size=nos)
        elif self.modeltype == 'umd':
            solutions = rand(nos,self.vars);
            for i in range(nos):
                index1 = solutions[i,:] <= self.probofone_true;
                index0 = solutions[i,:] > self.probofone_true;
                solutions[i,index1] = 1;
                solutions[i,index0] = 0;
        return solutions
    def pdfeval(self, solutions):
        if self.modeltype == 'mvarnorm':
            mvn = multivariate_normal(self.mean_noisy,self.covarmat_noisy) #create a multivariate Gaussian object with specified mean and covariance matrix
            probofsols = mvn.pdf(solutions)
        elif self.modeltype == 'umd':
            nos = solutions.shape[0]
            probofsols = np.zeros(nos);
            probvector = np.zeros(self.vars);
            for i in range(nos):
                index = solutions[i, :] == 1;
                probvector[index] = self.probofone_noisy[index];
                index = solutions[i, :] == 0;
                probvector[index] = self.probofzero_noisy[index];
                probofsols[i] = np.prod(probvector);
        return probofsols
    def buildmodel(self,solutions):
        pop,self.vars = solutions.shape
        if self.modeltype == 'mvarnorm':
            self.mean_true = np.mean(solutions,0);
            covariance = np.cov(solutions);
            self.covarmat_true = np.diag(np.diag(covariance)); # Simplifying to univariate distribution by ignoring off diagonal terms of covariance matrix
            solutions_noisy = np.append(solutions, np.random.rand(round(0.1*pop),self.vars), 0)
            self.mean_noisy = np.mean(solutions_noisy, 0);
            covariance = np.cov(solutions_noisy);
            self.covarmat_noisy = np.diag(np.diag(covariance));# Simplifying to univariate distribution by ignoring off diagonal terms of covariance matrix
            self.covarmat_noisy = np.cov(solutions_noisy);
        elif self.modeltype == 'umd':
            self.probofone_true = np.mean(solutions,0);
            print(self.probofone_true)
            self.probofzero_true = 1 - self.probofone_true;
            print('probofone_true')
            print(self.probofzero_true.shape)
            solutions_noisy = np.append(solutions, np.round(np.random.rand(round(0.1*pop),self.vars)),axis=0)
            print(solutions_noisy.shape)
            self.probofone_noisy = np.mean(solutions_noisy, 0);
            print(self.probofone_noisy)
            self.probofzero_noisy = 1 - self.probofone_noisy;
            print(self.probofzero_noisy)
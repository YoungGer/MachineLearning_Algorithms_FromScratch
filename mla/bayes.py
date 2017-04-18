import numpy as np
from scipy.stats import norm

class NaiveBayesClassifier():
    def __init__(self):
        self.mean = None
        self.std = None
        self.prior = None
        
    def train(self, X, Y):
        '''
        #input
        X,Y: nunpy array
        '''
        # seperate data by Y
        seperated = {}
        for i in range(len(Y)):
            x, y = X[i], Y[i]
            if y not in seperated:
                seperated[y] = [x]
            else:
                seperated[y].append(x)
        # get mean and var group by y
        mean = {}
        std = {}
        prior = {}
        for y in seperated.keys():
            seperated[y] = np.array(seperated[y])
            mean[y] = np.mean(seperated[y], axis=0)
            std[y] = np.std(seperated[y], axis=0)
            prior[y] = len(seperated[y])/len(Y)
        self.mean = mean
        self.std = std
        self.prior = prior
    
    def predict(self, X):
        mtx = np.ones([len(X), len(self.prior)])
        for i in range(len(X)):
            for y in range(len(self.prior)):
                row = X[i]
                mean, std, prior = self.mean[y], self.std[y], self.prior[y]
                # calculate probability
                mtx[i,y] *= prior
                for vi in range(X.shape[1]):
                    gauss = norm(mean[vi], std[vi])
                    mtx[i,y] *= gauss.pdf(row[vi])
        rst = np.argmax(mtx, axis=1)
        return rst  
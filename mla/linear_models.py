import numpy as np

class LinearRegression():
    
    def __init__(self):
        self.W = None
        
    def train(self, X, Y):
        '''
        #input
        X: numpy array of size (N,C)
        Y: numpy array of size (N,)
        '''
        self.W = np.linalg.pinv(X).dot(Y)
        
    def predict(self, X):
        '''
        #input
        X: numpy array of size (N,C)
        #output
        Y: numpy array of size (N,)
        '''
        return X.dot(self.W)

class LinearRegression_GD():
    def __init__(self):
        self.W = None
        self.loss_list = []
    
    def forward(self, X, Y, W):
        loss = np.sum((X.dot(W)-Y)**2)/X.shape[0]
        cache = [W,X,Y]
        return loss, cache

    def backward(self, loss, cache):
        W, X, Y= cache
        Y_hat = X.dot(W)
        dW = 2*X.T.dot(Y_hat-Y)
        return dW

    def train(self, X, Y, iteration, lr):
        self.loss_list = []
        # init
        N, C = X.shape
        W = np.random.randn(C,1)
        # train process
        for i in range(iteration):
            loss, cache = self.forward(X, Y, W)
            self.loss_list.append(loss)
            dW = self.backward(loss, cache)
            W -= dW*lr
        self.W = W
    
    def predict(self, X):
        return X.dot(self.W)
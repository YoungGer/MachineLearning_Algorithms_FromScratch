from scipy.linalg import svd
import numpy as np

class PCA():
    def __init__(self, solver='svd'):
        self.solver = solver
        self.means = None
        self.components = None
        self.S = None # squared singular value sorted in non-decreasing order
        
    def train(self, X):
        # mean centering
        X = X.copy()
        self.means = np.mean(X, axis=0)
        X -= self.means
        # solver
        if self.solver=="svd":
            _,S,V = svd(X) # X=U*S*V, V: Unitary matrix having right singular vectors as rows.
        elif self.solver=="eig":
            S,V = np.linalg.eig(np.cov(X.T)) # V: the column of V is the eigenvector
            V = V.T
            sort_idx = np.argsort(S)[::-1]
            V = V[sort_idx]
            S = S[sort_idx]
        else:
            raise Exception("Solver Not Exist") 
        # save 
        self.S = S**2
        self.components = V

    def predict(self, X, n_components=2):
        # mean centering
        X = X.copy()
        X -= self.means
        # predict
        V = self.components[:n_components].T
        assert X.shape[1]==V.shape[0]
        print ("Explained variance ratio: %f" % (np.sum(self.S[:n_components])/np.sum(self.S)) )
        return X.dot(V)
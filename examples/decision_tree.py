import sys
sys.path.append("../")
import numpy as np
from mla.trees import DecisionTree
from sklearn import datasets
from sklearn.cross_validation import train_test_split

# load data
iris = datasets.load_iris()
X, Y = iris['data'], iris['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# train model
model = DecisionTree()
model.train(X_train, Y_train)

# test model
print ("Train Accuracy: ", np.mean(model.predict(X_train)==Y_train))
print ("Test Accuracy: ", np.mean(model.predict(X_test)==Y_test))
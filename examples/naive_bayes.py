import sys
sys.path.append("../")
import numpy as np
from mla.bayes import NaiveBayesClassifier

# construct data
X = np.vstack([np.random.normal(0,1,[100,2]), np.random.normal(2,1,[100,2])])
Y = np.hstack([np.zeros(100), np.ones(100)])

# build model and train
model = NaiveBayesClassifier()
model.train(X, Y)

# model evaluation
print ("Traing Accuracy:", np.mean(model.predict(X)==Y))
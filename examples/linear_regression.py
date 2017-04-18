import sys
sys.path.append("../")
import numpy as np
from mla.linear_models import LinearRegression, LinearRegression_GD
from mla.metrics import mean_squared_error

# construct data
X = np.random.random(1000)[:,None]
Y = np.sqrt(X)
X = np.hstack([np.ones(X.shape),X,X**2,X**3, X**4])

# linear regression with close form
model1 = LinearRegression()
model1.train(X, Y)
print ("Closed Form LR MSE:", mean_squared_error(Y,model1.predict(X)))

# linear regression with gradient descent
model2 = LinearRegression_GD()
model2.train(X, Y, 10000, 0.0001)
print ("Gradient Descent LR MSE:", mean_squared_error(Y,model2.predict(X)))
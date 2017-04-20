import sys
sys.path.append("../")
from mla.pca import PCA
from sklearn import datasets

# load data
iris = datasets.load_iris()
# construce model
model = PCA("eig")
# model train
model.train(iris.data)
# model prediction
new_data = model.predict(iris.data, 2)
# model evaluation
print ()
print ("Transformed Data Top 5 Rows")
print (new_data[:5])
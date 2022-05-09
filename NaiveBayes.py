import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

data = pd.read_csv("iris.csv")

print("Features: ",  iris.feature_names)
print("Labels: ", iris.target_names)

#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.3, random_state = 42)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

from sklearn import metrics
print("Accuracy: ", metrics.accuracy_score(y_test,y_pred)*100)
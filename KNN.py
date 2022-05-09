import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv('iris.csv')
X = df.iloc[:,0:4].values
y = df.iloc[:,4].values

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state =0)
plt.scatter(X_train[:,0],X_train[:,1])
plt.show()

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)

from sklearn.metrics import confusion_matrix
C = confusion_matrix(y_test, y_pred)

print("Confusion Matrix\n",C)

from sklearn.metrics import accuracy_score
A = accuracy_score(y_test, y_pred)*100
print("Accuracy:",A)
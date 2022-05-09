
TASK 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
from matplotlib.colors import ListedColormap

dataset = pd.read_csv("https://raw.githubusercontent.com/stavanR/Machine-Learning-Algorithms-Dataset/master/Social_Network_Ads.csv")
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#linear kernal
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print("Linear accuracy",accuracy_score(y_test,y_pred))
print("Linear accuracy\n",cm)

#RBF kernal
classifier = SVC(kernel = 'rbf', random_state = 42)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print("RBF accuracy",accuracy_score(y_test,y_pred))
print("RBF accuracy\n",cm)

#Polynomial Kernal
classifier = SVC(kernel = 'poly',degree= 5, random_state = 0)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print("Polynomial accuracy",accuracy_score(y_test,y_pred))
print("Polynomial accuracy\n",cm)

import warnings
warnings.filterwarnings("ignore")
x_set,y_set= x_train,y_train
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop=x_set[:,0].max()+1,step=0.01), 
np.arange(start = x_set[:,1].min()-1, stop = x_set[:,1].max()+1, step= 0.01))

plt.contourf(x1,x2,classifier.predict(np.array([x1.flatten(),x2.flatten()]).T).reshape(x1.shape), alpha=0.75,cmap =ListedColormap(('green','white')))
plt.xlim(x1.min(),x1.max())
plt.xlim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
  plt.scatter(x_set[y_set == j,0],x_set[y_set==j,1], 
              c= ListedColormap(('blue','red'))(i),label=j)

plt.title('Random Forest classification(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

import warnings
warnings.filterwarnings("ignore")
x_set,y_set= x_test,y_test
x1,x2= np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop=x_set[:,0].max()+1,step=0.01), 
np.arange(start = x_set[:,1].min()-1, stop = x_set[:,1].max()+1, step= 0.01))

plt.contourf(x1,x2,classifier.predict(np.array([x1.flatten(),x2.flatten()]).T).reshape(x1.shape), alpha=0.75,cmap =ListedColormap(('green','white')))
plt.xlim(x1.min(),x1.max())
plt.xlim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
  plt.scatter(x_set[y_set == j,0],x_set[y_set==j,1], 
              c= ListedColormap(('blue','red'))(i),label=j)

plt.title('Random Forest classification(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#C values varying
c_list = np.linspace(0.01, 1000, 10)

acc1 = np.zeros(len(c_list))
acc2 = np.zeros(len(c_list))
acc3 = np.zeros(len(c_list))
acc4 = np.zeros(len(c_list))

#Linear with c
for i in range(len(c_list)):
    classifier = SVC(kernel = 'linear',C= c_list[i] ,random_state = 0)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    acc1[i]=accuracy_score(y_test,y_pred)
print("Accuracy for combination linear kernal with c values varying\n",acc1)
print("Max and Min values of the accuracy are\n{}\t{}".format(acc1.max(),acc1.min()))
print("\n")

#RBF with c
for i in range(len(c_list)):
    classifier = SVC(kernel = 'rbf',C= c_list[i] ,random_state = 0)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    acc2[i]=accuracy_score(y_test,y_pred)
print("Accuracy for combination RBF kernal with c values varying\n",acc2)
print("Max and Min values of the accuracy are\n{}\t{}".format(acc2.max(),acc2.min()))
print("\n")

##Gamma Varying values  
gamma_list = np.linspace(0.01, 1000, 10)

#rbf with gamma
for i in range(len(c_list)):
    classifier = SVC(kernel = 'rbf',gamma = gamma_list[i] ,random_state = 0)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    acc3[i]=accuracy_score(y_test,y_pred)
print("Accuracy for combination rbf kernal with gamma values varying\n",acc3)
print("Max and Min values of the accuracy are\n{}\t{}".format(acc3.max(),acc3.min()))
print("\n")

#linear with gamma
for i in range(len(c_list)):
    classifier = SVC(kernel = 'linear',gamma = gamma_list[i] ,random_state = 0)
   
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    acc4[i]=accuracy_score(y_test,y_pred)
print("Accuracy for combination linear kernal with gamma values varying\n",acc4)
print("Max and Min values of the accuracy are\n{}\t{}".format(acc4.max(),acc4.min()))
print("\n")

"""TASK 2"""

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/event-detection/Dodgers.data"
dataset = pd.read_csv(url)

print(dataset)
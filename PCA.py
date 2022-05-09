import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataset=pd.read_csv('~/desktop/wine.csv')
X=dataset.iloc[:,0:10].values
sc=StandardScaler()
X=sc.fit_transform(X)
pca=PCA(n_components=2)
X=pca.fit_transform(X)
explained_variance=pca.explained_variance_
explained_variance_ratio=pca.explained_variance_ratio_
print(explained_variance)
print(explained_variance_ratio)
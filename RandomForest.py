import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_2 = pd.read_csv('data.csv')
print(df_2.head())
X = df_2.iloc[:,3:11].values
y = df_2.iloc[:,0].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=7)
X_pca = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_
print('Information Content =', (np.array(explained_variance_ratio).sum())*100)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.20, random_state=48)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
loss_train = []
loss_test = []

for i in range(1, 50):
    rfr = RandomForestRegressor(n_estimators = i, random_state = 48)
    rf = rfr.fit(X_train, y_train)
    y_pred_1 = rf.predict(X_train)
    y_pred_2 = rf.predict(X_test)
    loss_train.append(mean_squared_error(y_train, y_pred_1))
    loss_test.append(mean_squared_error(y_test, y_pred_2))

rfr = RandomForestRegressor(n_estimators = 50, random_state = 48)
rf = rfr.fit(X_train, y_train)
y_pred = rf.predict(X_test)

from sklearn.metrics import mean_absolute_error
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))


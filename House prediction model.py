import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('kc_house_data.csv')
dataset.drop(dataset.columns[[0,1,14,15]], axis=1, inplace=True)
X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values

'''
print(house.isnull().any())
print(house.dtypes) 
'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Prediction
y_pred = regressor.predict(X_test)

plt.scatter(y_test, y_pred, color = 'red')
plt.xlabel('Actual Price')
plt.ylabel('Prediction')
plt.show()


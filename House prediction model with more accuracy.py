import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

house = pd.read_csv("kc_house_data.csv")
house.head()

'''
print(house.isnull().any())
print(house.dtypes) 
'''

house = house.drop(['id', 'date'],axis=1) #drop columns

#did it for visualising different parameters
'''
with sns.plotting_context("notebook",font_scale=2.5):
    g = sns.pairplot(house[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 
                 hue='bedrooms', palette='tab20',size=4)
g.set(xticklabels=[]);
'''

Y = house.iloc[:,0].values  #making my sets
house = house.drop(['price'], axis=1)
X = house.as_matrix()

rfe = RFE(lr, n_features_to_select=1)
rfe.fit(X,Y)

rf = RandomForestRegressor(n_jobs=-1, n_estimators=50) #random forest regression
rf.fit(X,Y)

y_pred_lr=lr.predict(X)
y_pred_rf=rf.predict(X)

#Now the results
#Note: More the plotting is conjusted to the line y=x, better are the reults
plt.scatter(Y, y_pred_rf, color = 'blue')
plt.xlabel('Actual Price')
plt.ylabel('Prediction')
plt.title('House Pricing predictions using RandomForest Regression')
plt.show()



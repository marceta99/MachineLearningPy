
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#here we will not split data set on training and test set because csv fajl that we have
#is very small so we will work with entire dataset
dataset = pd.read_csv("Position_Salaries.csv") 
X = dataset.iloc[ :, 1:-1 ].values
Y = dataset.iloc[ :, -1  ].values

#training Random Forest Regression Model
#we need to import number of trees that we want as parameter and here is 10, and random_state
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 10, random_state= 0)
regressor.fit(X, Y)
  
#prediction
#predict method accepts 2d array as input so that is why there is double brackets [[]]
regressor.predict([[6.5]])

#Random Forest Regression is not very good for data sets with one input variable, or some small number
#of input variable, but it is very good if we have a lot of input variables for our model 

#visualising the Random Forest Regression results in high resultion because we could not see results of DTR
#very nice if it is not in high resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
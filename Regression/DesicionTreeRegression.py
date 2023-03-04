
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#here we will not split data set on training and test set because csv fajl that we have
#is very small so we will work with entire dataset
dataset = pd.read_csv("Position_Salaries.csv") 
X = dataset.iloc[ :, 1:-1 ].values
Y = dataset.iloc[ :, -1  ].values

#we dont need to do feature scaling in Desicion Tree Regresion because predicion ins DTR are comming from
#spliting the data, and there are not some equations like in previous model with SVR  
#and we can split data in different categories event if data have different range without need to scale 

#training desicion tree regression model with whole dataset
#this random_state parameter is same like set.seed(0) in R
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, Y)


#prediction
#predict method accepts 2d array as input so that is why there is double brackets [[]]
regressor.predict([[6.5]])

#Desicion Tree Regression is not very good for data sets with one input variable, or some small number
#of input variable, but it is very good if we have a lot of input variables for our model 

#visualising the Desicion Tree Resultsin high resultion because we could not see results of DTR
#very nice if it is not in high resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
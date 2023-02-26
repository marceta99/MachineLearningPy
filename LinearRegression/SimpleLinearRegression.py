import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Salary_Data.csv") 
X = dataset.iloc[ :, :-1 ].values
Y = dataset.iloc[ :, -1  ].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2 , random_state=1)

 
#training simple linear regression model with training set
#for cerateing regression model we will use LInearRegression class from sklearn
#and our regression model will be instnance of that LinearRegression class
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#and with that we just get the model but wee need to train that model  with training set 
regressor.fit(X_train, Y_train)


#predictions on test set
y_pred = regressor.predict(X_test)

#visualising training set results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Training test)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()
 

#visualising test set results
#because regression line in simple regression model is from unique equation and we will always
#have same regression line that we got when we trained the model with training set we can just 
#leave this second plot which plots regression line with training model because it will be the same line always
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()
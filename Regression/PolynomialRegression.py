import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv") 
X = dataset.iloc[ :, 1:-1 ].values
Y = dataset.iloc[ :, -1  ].values

#because here we have very small data set we will not split it into data and training set
#and we will work with whole dataset


#building polynomial regression model 
#first thig is to build matrix X_poly that containes values for x1, x1^2, x^3 etc...
#and then we will integrate that matrix into linear regression model
#first we will import class that allow us to create that matrix
from sklearn.preprocessing import PolynomialFeatures 
#first we are going to build that matrix with only x1, x1^2 and that is why degree is 2
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

#and now we train linear regression model based on that matrix that containes x1, x1^2
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, Y)
 

#visualising polynomial regression results with degree of 2 
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg.predict(X_poly), color="blue")
plt.title("Polynomial model")
plt.xlabel("Position level")
plt.ylabel("Salary") 


#now we will create new polynomial model with degree of 4 which means x1,x1^2,x1^3,x1^4
#and we will see that this model is much better
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, Y)
 

#visualising polynomial regression results with degree of 4
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg.predict(X_poly), color="blue")
plt.title("Polynomial model")
plt.xlabel("Position level")
plt.ylabel("Salary") 


#predicintg salary of person with position level of 6.5
#if we had linear regression model we would predict like this lin_reg.predict([[6.5]])
#with polynomial regression model we need to input x1, x1^2, x1^3 , x1^4 
#so we will create matrix containing that values with poly_reg.fit_transform() 
lin_reg.predict(poly_reg.fit_transform( [[6.5]] ) )

#and we see that predicted salary is 158862.45

































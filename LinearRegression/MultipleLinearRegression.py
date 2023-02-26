import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv") 
X = dataset.iloc[ :, :-1 ].values
Y = dataset.iloc[ :, -1  ].values


#encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer( transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2 , random_state=1)




#creating multiple regression model 
#When we create multiple linear regression model in pyhton, class LinearRegression that we use to create that model will automatically avoid dummy variable trap, and it will automatically only use one less dummy variable when creating a model. 
#Also we will not need to use backwards elimination to choose best input variables for our model, because this class LinearRegression will automatically calculate which ones are best fitting for the model of linear regression and it will automatically only choose them to create model .
from sklearn.linear_model import LinearRegression
#with this line we build dumb model
regressor = LinearRegression()

#but we need  to train that model with train data
regressor.fit(X_train, Y_train)
 

#prediction with test set
y_pred  = regressor.predict(X_test)


#visualy printing predicted result and real values from test set
#then we will setup to be diplayed only 2 decimals for numerical values
np.set_printoptions(precision=2)

#because this is multiple linear regression model we can not just plot on x ases input variable and on y output like with simple linear regression 
#beacuse we have more then one input variable and we canot plot that on a graf, we would need 5 dimentional graf

#displaying two vectors, predicted profits and real profits from test set
#concatinate() functions thakes as parameters that vectors that we want to concatinate
#and we want to print that vectors vercitaly so we will use reshape() for that, which takes length of that array(vector) as parameter and then 1 because it is only one column that we want to print
#then second parameter of concanitate function takes 1 or 0, where 0 means that we want to do vertical concatination, and 1 means that we want to do horizontal concatination
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)), 1) )
 
    
 
    
 

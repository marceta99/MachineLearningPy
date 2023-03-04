
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#here we will not split data set on training and test set because csv fajl that we have
#is very small so we will work with entire dataset
dataset = pd.read_csv("Position_Salaries.csv") 
X = dataset.iloc[ :, 1:-1 ].values
Y = dataset.iloc[ :, -1  ].values


#feature scaling 
#in SVR we will have to apply feature scalling, because here we dont have coeficients
#like in linear, multiple and polynomial regression which would allow us not to do 
#scaling, so here we have to do it ourself

#here we will have to do feature scaling on both input and output variable, 
#because salary output variable is in much bigger range then position input variable

#also one reminder if we split data set in test and trening set, we should alwas do 
#feature scaling AFTER spliting set on training and test because test values should
#not be affected by training values, and that would be the case if we did feature scaling
#on whole dataset.


print(X)
print(Y)
#here we see that X is two dimentional array, where y is one dimentional array
#and function that we will use for scaling espexts as input values in two dimentional
#array format, so we will have to transform Y in two dimentional format first 
#for that we will use reshape function that takes number of rows that we want new transfromted object to have
#as first paramatere and then number of columns that we want that new transformed object to have
Y = Y.reshape(len(Y), 1) 

print(Y)

#we will use standardisation for feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#we are not going to use same objet of StandarScaler to scale X input matrix and output Y
#and reson for that is because when we do this fit on X matrix it will calcute and put
#mean and standard deviation values of X in sc ojbect, and we dont want to use that values
#for Y, but we want to calcute mean and standard devation of Y values 
sc_Y = StandardScaler()
Y = sc_Y.fit_transform(Y)

print(X)
print(Y)



#training SVR model on whole dataset because we didnt split data set on training and test set
#here in SVR and also in SVM with classifictaion that we will do later, we have kernels
#which can either learn some linear relathionships and that is linear kernel or non linear relathionships
#in data set which are non linear kernel, and in SVR we use Radial Basis Kernel and that is this 'rbf'
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')

#now we have empy SVR model, and now we will train that model
regressor.fit(X, np.ravel(Y))  

#predicting the results
#because we did scale Y before, when we run predict method it will predict
#values in that scale, but we want real prediction with real values not scaled
#so we will have to reverse the scaling to get predicted values in the original range

#so beacuse our model is learned based on scaled values we will have to scale input values
#so that prediction will be valid and that is why we are calling scaler sc_X.transform()
#and we will put value in double brackets [[]] because this function for scaling expexts 2d array
#and then we will apply reverse scalling to get predicted values in original scale with inverse_transform() and reshape()
sc_Y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))  


#visualising the SVR results  
#because X and Y values are scaled values, we need to reverse that scaling to original values so plot would be nicer
#and when we ploting prediction we will not need to scale values for X because they are alredy scaled so we will just insert X  
plt.scatter(sc_X.inverse_transform(X),sc_Y.inverse_transform(Y), color = "red")
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X).reshape(-1,1))  , color="blue")
plt.title("SVR  ")
plt.xlabel("Position level") 
plt.ylabel("Salary")  



















































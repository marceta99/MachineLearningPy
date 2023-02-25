#data oreprocessing

#1)import libraries
#library numpy will allow us to work with arrays
#library matplot will allow us to plot some nice charts 
#library pandas will allow us to import data set, create matrix, vectors etc...
#with dot . we can access specific module of a library and for exaple with matplotlib.pyplot
#we access pyplot module of matplotlib library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2) import dataset csv
#we will create data frame from csv file 
dataset = pd.read_csv("Data.csv") 

#then we create matrix of features and vector of depended variable
#features or independed variables are input variables and that is X
#depended variable is output variable and that is Y 
#first paramter of iloc is rows that we want and because we want all rows we will use : 
#second parameter is columns and we want all columns execept last one which is output column
X = dataset.iloc[ :, :-1 ].values
Y = dataset.iloc[ :, -1  ].values

print(X)
print(Y)

#3) taking care of missing data
#we will replace the missing values NA with with average value of that column
#we will use SimpleImputer class from library  sklearn.impute and frist we will import that class and then create instance
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#we will include only numerical colums so that is why is 1:3 and 3 is because index 3 is excluded
#and fit method will look for missing values on that columns and transform() will replace missing values
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)
 

#4) Encoding categorical data
#we will encode country categorical column with 3 values France, Germany, Spain into 3 new columns with 0 & 1
#for that we will use columnTransformer class and onehotencoder class
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#first parameter is which type of transformation(endoding) and which class is doing that encoding OneHotEncoder 
#and on which columns we want it to be done
# and second parameter is on which columns we dont want this transformation to happen with 'passthrough' and that are all other numeric columns 
ct = ColumnTransformer( transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
#here we do fiting and transforming with only one method
#because in next chapters all functions that we use to create model will demand input data to be of type np array we need to cast data into that type
X = np.array(ct.fit_transform(X))

print(X)

#5) endcoding output indipendent variable
#our output variable right now have values of 'yes' and 'no' and we want to encode at 0 and 1 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

print(Y)


#6) split dataset into training and test set
#from this we will get 4 variables X train, X test, Y train, Y test
#this random_state is like set.seed(1) in R
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2 , random_state=1)

print(X_train)
print(X_test)
print(Y_train)
print(Y_test)


#7) feature scaling 
#in order to some values not be dominated by some other bigger values we do feature scaling
#here we will apply standardisation 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#one important thing to remember is that we will not apply feature scaling to this new columns
#with values 0 and 1 that we created from factor variables because there is not need for that 
#because they are alredy in same range 
 
#first we will fit our standardScaler on trainingSet and only on age and salary columns
#and this 3: will take all indexes after 3
#and this fit method will just calculate mean and standard deviation
#and after that, transform will use formula to calculate standardization 
X_train[:, 3: ] = sc.fit_transform(X_train[:, 3: ])

#then we will scale data on test set 
#there we will need to use same fit as on train set because for our predicions to be relevant with the way model is trained
#we must use same scaler for test set, and that is why here we will just use transform method with same scaler
#because if we did use fit method we would get new scaler, and we want to use same scaler 
X_test[:, 3: ] = sc.transform(X_test[:, 3: ])


print(X_train)
print(X_test)













































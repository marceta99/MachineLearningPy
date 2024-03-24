# artificial neural network can be both used for classification and regression but here we will use it for classification 
# this is example of classic artificial neural network, meaning the fully-connected neural network with only fully-connected layers, you know, with no convolutional layers or other types of layers. Here we will just have an input vector containing different features, and we will predict an outcome which will be a binary variable

# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# here for X we will not take all columns except last one because we have few columns in our data set which are not relevant like RowNumber, CustomerId and Surname so we will take for X all columns except first 3 and last one which is output column
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Encoding categorical data

#In machine learning, dealing with categorical variables involves converting these variables into a form that could be provided to ML algorithms to do a better job in prediction. Label encoding and one-hot encoding are two widely used techniques for transforming categorical variables into numerical values.
# Label encoding converts each category into a numerical value. It assigns a unique integer to each category. For example, if you have a categorical feature like color with three categories (red, green, blue), label encoding would replace them with (0, 1, 2).
# One-hot encoding converts categorical variables into a form that could be provided to ML algorithms to do a better job in predictions. It creates a binary column for each category and returns a sparse matrix or dense array (depending on the implementation). For the same example above, one-hot encoding would create three features named 'color_red', 'color_green', and 'color_blue'. Only one of these features would have the value 1 for each sample, and the others would be 0.

# Label Encoding the "Gender" column
# now because we have categorical columns like Gender which has values for Male and Female we have to encode that values to be numbers like 1 or 0
# also we have categorical column Geography which has values like France, Spain, Germany etc and we need to encode also that categorical column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)


# One Hot Encoding the "Geography" column
# now becaues Country categorical variable has values like Germany, France, Spain and there is no order in that values like France > Spain> Germany, so we could not say France=0, Spain=1, Germany=2
# we will have to use One Hot Encoding so for each of these values will be created new column which will have values 0 or 1
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
#Feature scaling is absolutely important for deep learning. Whenever you build an artificial neural network you have to apply feature scaling. That's absolutely fundamental. And it is so fundamental that we will actually apply feature scaling to all our features regardless of whether they already have some values of zero and one like the dummy variables and same for these ones. We will just scale everything because it is so important to do it for deep learning.
# and for neural network we will scale everything, so every variable, not like for other models where we picked few variables to scale, here we will have to scale every variable
# we do feature scaling for both test set and traing set
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Building the ANN
# So the first thing we have to do is obviously to create a variable that will be nothing else than the artificial neural network itself. And guess what? This artificial neural network variable will be created as an object of a certain class.And that certain class is the sequential class which allows exactly to build an artificial neural network but as a sequence of layers as opposed to a computational graph. You saw in the intuition lectures that an artificial neural network is actually a sequence of layers,starting from the input layer and then successively, we have the fully connected layers up to the final output layer. That's what I mean by a sequence of layers. And then the other type of neural network is indeed a computational graph which are neurons connected anyway not in successive layers. And an example of this is Boltzmann machines. Restricted Boltzmann machines or deep Boltzmann machines are great examples of computational graph. this is really advanced deep learning but they are covered in our deep learning A to Z course
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
#we're gonna call our dense class, which as any class can take several arguments. And here we have to indeed enter these arguments. The most important one is this one, unit, which corresponds exactly to the number of neurons, to the number of hidden neurons you want to have in this first hidden layer, not in the input layer. We will automatically have our different features. In the input layer, the input neurons will simply be all these features(all input variables)  starting from credit score. That will be one neuron, then another input neuron will be Geography, then another one Gender etc All these will be the input neurons in the input layer. But then when we create that first hidden layer we will have some hidden neurons inside.And in this dense function now, well, we can specify of course, how many hidden neurons we want to have. And now, now comes the most frequently asked question in deep learning. The very famous question, how do we know how many neurons we want? Is there a rule of thumb, or should we just experiment? Well, unfortunately, there is no rule of thumb. It is just based on experimentation or we call it the work of an artist. You have to experiment with different hyper parameters We call them hyper parameters in the sense that these are parameters that won't be trained during the training process. So unfortunately, there is no rule of thumb and therefore we just have to pick one number here which wouldn't sound irrelevant or extravagant. And that number will be six. I actually tried several numbers, and I got more or less the same accuracy in the end.
# All right, and now the next parameter that is important among this huge list of parameters you can see many of them, but no worries we will keep the default value for all of them, except this one which corresponds of course, to the activation function. And you saw in the intuition lecturers if you will that the activation function in the hidden layers of a fully connected neural network must be the rectifier activation function. And therefore, that's exactly what we must specify here. We of course, don't want no activation function. So here we have to specify that we want rectify activation function. And the way to specify this is to enter here in our activation parameter, well, the code name for the rectifier activation function which is in quote, well ReLU.
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Congratulations. Now, you know how to build, actually a shallow neural network. And you will know in a second how to build a deep neural network. Because the way to actually add a second hidden layer here couldn't be more simple. The only thing that you have to do is just copy this line of code, and then in a new line of code here for the second hidden layer, you just need to paste it. That's what I mean by this add method can add any new layer at whatever stage of the construction process of your ANN you're into. You can use this ad method to add anything, and the way to add a second hidden layer is just the same as adding the first hidden layer. Unless of course you want to change the number of hidden neurons, but six hidden neurons in the first hidden layer and six other ones in this second hidden layer is just fine. But once again, feel free to change the hyper parameter values here. Maybe you will get a better accuracy in the end.

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
#However, now to add the output layer you have to do something special, Well, that's of course, because we are adding a new layer and this add method can add any layer you want, including of course, the output layer. So here we're still using the add method to add this final output layer, and then of course we still want our output layer to be fully connected to that second hidden layer and therefore we're using again here, the dense class. this units=6 that we had before has to be replaced by which value? Well, to get the answer, we need to have a look at our output variable Because remember, the output layer contains the dimensions of the output, the output you want to predict. And here, since we actually want to predict a binary variable, which can take the value one or zero while the dimension is actually one. Because we only need one neuron to get that final prediction zero or one. However, if we were doing classification with a non-binary dependent variable like dependent variable that has three classes let's say "A" "B" "C" well, we would actually need three dimensions, three output neurons
#For second parameter once again, remember in the intuition lectures that for the activation function of the output layer, well you don't want to have a rectifier activation function but a sigmoid activation function. Why is that? It's because having a sigmoid activation function allows to get not only ultimately the predictions but even better, it'll give you the probabilities that the binary outcome is one, so that we will not only get the predictions of whether the customers choose to leave or not the bank but we will also have for each customer the probability that the customer leaves the bank and all this thanks to that sigmoid activation function. So you definitely want that sigmoid activation function for the output layer only.
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Now we only have a brain so far, but which is totally stupid actually because it was not trained yet on the data set. So we're gonna make it smart
#And we are gonna do this in two steps. The first one is to compile the ANN with an optimizer a loss function, and a metric, which will be of course the accuracy because we're doing some classification. And then the second step will be, of course, to train the A and N on the training set over a certain number of epic.

# Part 3 - Training the ANN

#what do we have to enter as parameters inside this compiled metho We have to enter three parameters. The first one is the optimizer to choose an optimizer. Then the second one is the loss to choose a loss function. And the third one is the metrics with an S parameter. Because know that you can actually choose several metrics to evaluate your ANN at the same time, but we will only choose one and we will choose the accuracy. All right, so for the optimizer, which one would you like to get? Well, in the intuition lectures, Kirel mentioned that the best one are the optimizers that can perform stochastic gradient descent. And the best of them, you know, the one that I recommend by default is the Atom Optimizer, which is a very performance optimizer that can perform stochastic gradient descent. And by that, let me just remind what stochastic gradient descent allows to do. Well, you know, it is what will update the weights in order to reduce the loss error between your predictions And the real results. You know, when we train the A and N on the training set we will at each iteration compare the predictions in a batch to the real results in the same batch. And that optimizer here will update the weight through stochastic gradient descent because we're gonna choose the Atom optimizer to at the next iteration, hopefully reduce the loss.
#And now you have to know something very important when you are doing binary classification. You know, classification when you have to predict a binary outcome. Well, the loss function must always be the following one entered in quotes, of course which is binary underscore cross entropy, just like that. And now let me tell you what you would have to enter if you were doing non-binary classification. You know, like for example predicting three different categories while here you would have to enter a category called cross entropy loss okay? For binary classification the loss must be binary cross entropy. And for non-binary classification the loss must be category call cross entropy. And then also, you know when doing non-binary classification when predicting more than two categories while the activation should not be sigmoid, but soft max, right? I take this opportunity to also give you the other cases of classification, which you could encounter.
#And now let's enter the final parameter here. Metrics, as I said we can actually choose several metrics at the same time. Therefore, in order to enter the values of this parameter while we have to enter them in a pair of square bracket which is supposed to be, you know the list of the different metrics with which you want to evaluate your A and N during the training, but we will only choose the main one. You know, the most essential one which is the accuracy and which you have to enter in quotes. Alright, Accuracy, just like the classic spelling. And now, now congratulations. You know how to do a full compile of your A and N with an optimizer, a loss, and some metrics.
# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
# And then when training an artificial neural network we actually need to enter two more parameters which are first the batch size. Because indeed batch learning is always more efficient and more performance when training an artificial neural network. Meaning that instead of comparing your prediction to the real result, one by one, you know to compute and reduce the loss, well you are gonna do that with several predictions compared to several real results into a batch. And the batch size here, you know the batch size parameter gives exactly the number of predictions you want to have in the batch to be compared to that same number of real results. And the classic value of the batch size that is usually chosen is 32, right? If you don't want to spend too much time tuning this hyper parameter well I recommend to choose the default value 32.
# final parameter we have to enter here. That's of course the number of epochs. You know, a neural network has to be trained over a certain amount of epochs so has to improve the accuracy over time. And we will clearly see that once we execute this cell. So the name of the parameter for the number of epics is simply epochs And while you will see that, it'll go very fast. So we can just take 100 epochs. But once again, feel free to choose another number as long as it is not too small, because you know your new network needs a certain amount of epochs in order to learn properly, you know learn the correlations to get the ultimate best predictions.
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Predicting the result of a single observation

"""
Homework:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?

Solution:
"""
#It's very important to remember that any input of the predict method must be a 2D array, whether it is the test set to predict an assemble of outcomes for an assemble of observations, or whether it is a single prediction. Well anything has to be in a double pair of square brackets which therefore makes a 2D array.
# Then inside this 2D array, we're gonna enter, well, the different informations here, and that leads me to mention the second important thing which you had to pay attention to, which is about that variable here. Geography, France, according to you Geography, France, according to you do we have to enter, as you know, this first information here for the geography variable do we have to enter the string France, or do we have to enter something else? Well, of course we have to enter something else and that's something else are the values of the dummy variables, right? That's the second thing you had to do correctly. And therefore now we need to check what was the encoding for France. Well, remember if we have a look at our matrix of features X right above, you know, in part one, data processing, so this is the matrix of features X the whole matrix containing all the customers. So in order to know what France responds to, you know in terms of encoding, we just have to have a look at the first observation before one hot encoding was applied because indeed that first observation correspond to France, and so now we just need to compare France here to the one hot encoding resulting from what we did here. And well, we can see that the first row here contained 1 0 0 as the values of the dummy variables, and therefore France was encoded indeed into 1 0 0. So the dummy variable values for the France country and the geography variable is indeed 1 0 0, and that's exactly what we have to enter here as the first parameter Alright, and then all the rest is easy
# And that leads me to the third thing you had to pay attention to, which is the fact that remember the predict method should be called onto observations onto which the same scaling was applied as in the training. And since we trained our artificial neural network with the scaled values, you know the scaled values of the features, well, the predict method has to be called onto these observations to which the same scaling was applied. And that was the third thing you must have not forgotten, which is the fact that you have to apply your SE object here to scale that single observation, right? That's super important. Make sure to pay attention to this, check if some scaling was applied during the training, and yes, it is the case here in any way it is always the case for neural networks and therefore in the predict method, well we need to scale that single observation. And the way to do this is of course by calling our SE object and transform metho be careful not the fit transform because the fit transform is used to remember get the mean and the standard deviation of the values in the training set in order to fit your scaler to the training set. But then for the test set, we only need to call the transform method because if we fit it again, the scale well that would cause some information leakage. You know, I explained this in 0.1, data processing, check it out again if you need, but remember that on the test set or on new observations on which you deploy your model in production you can only apply the transform method. And that's what we're gonna call here transform, there we go
# ANN predicts exactly the predicted probability. And here we choose a threshold of 0.5 to say that if that predicted probability is larger than 0.5, well, we will consider the final result to be one, right? Because the predicted probability that the outcome is one is larger than 0.5, meaning that there is more than 50% chance that the predicted outcome is one. So we'll consider it to be one. And however, if the predictive probability that the customer leaves the bank is below 0.5, well, we will consider it to be zero. Of course, you can choose a different value of the threshold
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)


# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

















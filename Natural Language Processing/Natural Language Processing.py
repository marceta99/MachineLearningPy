# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# Here we add delimiter '\t' because this is not csv file, but tsv file tab seperated value
# also we add extra parameter quoting=3 to remove all double quotes from the file
# because here in natural language processing we need to clean text file as much as possible before using it so results would be better
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
# this is the library that will allow us to simplify the reviews from reviews dataset
import re
#the NLTK library. Very classic library in natural language processing, which will allow us to download the ensemble of stop words. So what are the stop words? These are, you know, the words we don't want to include in our reviews, you know, after cleaning the text, which you know are words that are not relevant to help the predictions of whether a review is positive or negative. And these words include, you know, the simple ones like "the", you know. All the articles like "the", "er", "end", you know. All these words which don't give any hint of whether a review is positive or negative. So we will remove all these words, you know. All the words that are not helpful to predict if a review is positive or negative.
import nltk

# now that we imported NLTK, we can call NLTK. And from which we are going to download all the stop words. And to specify this, we need to enter here in quotes inside this download function from the NLTK library "stopwords" And this will get all the stop words and you'll see later on how we use this to indeed not include these non-relevant words in our reviews.
nltk.download('stopwords')
#So basically  line of code befor downloads them and this line of code imports them into our notebook.
from nltk.corpus import stopwords

#And this is the class we'll use, of course, to apply stemming on our reviews. So now let me remind what this is about. Stemming consists of taking only the root of a word that indicates enough about what this word means. So for example. Let's say there is a review that says, "Oh, I loved this restaurant." Okay? And let's say we want to apply stemming to the word "loved". Well, what it will do is that it will transform loved into love. Just to simplify the review. Because whether we say, "Oh, I loved this restaurant." or "Oh, I love this restaurant." Well, you know, that means the same. That means that the review is positive. So we can totally remove all the conjugation of the verbs, you know. Just keeping the present tense so that we can indeed simplify the reviews. Because remember at the end, you know, after cleaning the text when creating actually the bag of words model, we will create a sparse matrix where in each column we will have all the different words of all our different reviews. And therefore, in order to optimize or you know, minimize the dimension of this sparse matrix, where the dimension is exactly the number of columns. Well, we need to simplify as much as we can the words. And if we don't apply stemming, well, you know, in the sparse matrix, we will have one column for love and one column for loved. And since that means the same thing, that would be redundant. And that would make the sparse matrix even more complex.
from nltk.stem.porter import PorterStemmer

# So the first thing we'll do is create a new list which we'll call corpus. All right. And we will initialize this list as an empty list. And what will this list be exactly? You know, what will it contain? Well, it will simply contain all our different reviews. You know, all the different reviews from our data set but all cleaned and all into this list corpus. So what we'll do actually is, you know, we will make a for loop to iterate through all the different reviews of our data set. And for each of these review, we will apply a cleaning process, you know, by putting all the letters in lower case and removing the punctuations and removing the stop words. All these things. And we will do that one review after another. And each time we clean a review. Well, we will add it to this corpus. So this corpus will only get in the end, all the cleaned reviews
corpus = []

#Now we're gonna apply different steps to clean each and every single review of our data sets.
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #And what we're gonna replace, actually, is any element that is not a letter, you know, from A to Z, by a space. So that every punctuation, like quotes, double quotes, comas or columns or anything you want will be replaced by a space, and it has to be replaced by a space. Because otherwise we can have two words that stick together. So we need to make sure we replace the punctuations by spaces so that we can indeed still separate the words. All right. And the way to do this, thanks to this subfunction is to enter here in the parameters. First, what we want to replace. And the trick to say that we want to replace anythingthat is not a letter, is to do it this way. You start with a pair of square brackets here. Just like that. So that what's inside this pair of square brackets will be what will be replaced, you know, by the spaces. And the trick to say that what we want to replace are anything but letters is to include a ^ here. And I will explain what this means. And then add A. So you have to do it like that, actually. ^^A. Okay? A-Z. So all the lowercase letters from a to z. But also then all the capital letters from A to Z. All right? And what this ^ means is exactly not, you know. This symbol in mathematics or computer science means not. Meaning not all the letters from a to z in lowercase nor the capital letters from A to Z which is exactly what we want. We want to replace anything that is not the letters from A to Z in lowercase or capitals by spaces. And the way to specify that we want to replace all these by spaces is well exactly what we have to enter here as a second parameter. And which we will enter in, you know, quotes, but inside a space, right? What's inside these quotes is exactly what we want to replace those non-letters here by. All right? So we're gonna replace everything that is not letters meaning all the punctuations by this space. And then finally, we have to enter one final argument, which is of course, where we want to do all these replacements. You know, inside the what. Inside which review, right? Inside which text. And so very simply, the third parameter we have to enter here is the review in which we want to do all these replacements.
  review = review.lower() # So that new step will be to transform all the capital letters into lowercase.
  review = review.split() # And then one final cleaning before we proceed to stemming in the next tutorial. Well, actually, what we have to do now is something to prepare for the stemming and that's something is to split the different elements of the reviews in different words.Actually, because the different elements are now words. So we're gonna split the review into its different words so that then we can apply stemming to each of these words,
  ps = PorterStemmer() # create object of class PorterStemmer
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)] # And since now we actually have a list of the words and the reviews, thanks to this step 10 here, well, we can totally apply this single-row for loop to iterate through all the words of the review and apply stemming to each of them.
  review = ' '.join(review) #But now that it's done, we will just join these words back together to get the original format of the review
  corpus.append(review) #and we need to add each review to the corpus list
print(corpus)

#So now, now it is time for the next essential step. When doing sentiment analysis, it will be to create the Bag-of-Words model, which will consist, basically, of creating this sparse matrix containing now all the words of the reviews after they were cleaned. So, we will take all the different words of all the reviews, we will put them in different columns of the sparse matrix, and then that will be actually our feature matrix of features, which we will combine to the dependent variable vector containing all the binary outcomes to train our feature machinery model which will be the Naive Bayes model to learn the text and understand whether the reviews are positive or negative.

#creating the bag-of-words model, which we are ready to do now, because all our reviews are properly cleaned. So we're gonna get them into the bag-of-words model to create, you know, this sparse matrix, which will contain in the rows, well, the different reviews, you know, the same reviews as the ones in our corpus. And in the columns, all the different words taken from all the different reviews. You know, all of them. And each cell will either get a zero or a one. It will get zero if the word of the column is not in the review of the row. And it will get a one if the word of the column is indeed part of the words in the review of the row.
#And the process of creating all these columns corresponding to each of the words taken from all the reviews is called tokenization.

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer() #It is to create an instance of this class and we're gonna call that CV as count vectorizer which will be created as, you know an instance of this count vectorizer a class, perfect which has to take as input only one important parameter. Can you actually guess what it is? Well, is actually the maximum size of sparse matrix, you know, the maximum number of columns therefore the maximum number of words you want to include in the columns of the sparse matrix. And why is this important that, because you know in our corpus of reviews now with all the simplifications, well, we actually have still some words that are not relevant or, you know not helpful to predict for review is positive or negative even if there were not part of the stop words. And these include, for example, you know, texture, you know texture doesn't really help to predict if a review is positive or negative or you know, bank you know, or holiday or Rick and even Steve. You know, Steve doesn't help at all. So we still have these words, which even if they're not part of the stop words don't help at all to predict if a review is positive or negative. And the way to get rid of them is by, you knowentering this parameter that we're about to enter. The way to get rid of them is just to take actually the most frequent words, you know the words that appear most frequently in the reviews because probably here Steve only appears once. So if we only take the most frequent words we won't include Steve in this sparse matrix, you knowin the tokenization process. So, so that's the trick.And so now we need to just choose a maximum size of the sparse matrix. However, we can't really know now how many words there are in total, you know, before we take the most frequent ones. So what we'll do in fact, is we will live this for now. You know, we won't enter this parameter now we will run this cell once we create the sparse matrix which is actually going to be the matrix of features when training our naive based model on the training set, it's gonna be the matrix of features. And therefore we will do a print in order to know the total number of columns and we will get therefore the total number of words and then we can reduce that total number of words to a lower number of the most frequent words in the sparse matrix so that we can simplify even more the bag of words model.
X = cv.fit_transform(corpus).toarray() #The fit method will just take all the words and the transform method will put all these words into the columns. And then we just need to add here a two array. Because actually, you know, remember that the matrix of features must be a 2D array It has to be a 2D array, because then you know we will train the naive based model on the training set. And this expect, of course, an array as the format of its input, you know, the matrix of features.
y = dataset.iloc[:, -1].values #Our final step here is to create the dependent variable vector Y and we will just take last column of data set which already has values 1 or 0 depening of it review is positive or negative

# now we want to get total number of columns in X which is acctualy total number of words that we got from tekokenization and we will get it by only acessing the length of the first row. So  this will give us the number of columns in X which is total number of words that we taken from all reviews and tokenization and for each review which is row there are values 0 or 1 for each word(column) depending on is that review containing that word or not
print(len(X[0])) # 1500 columns(words)

# and now we now the number 1500 is the number of most frequent words so we can add this as paramter when we create object of class CountVectorizer 
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# now we have bag of words model with only relevant words like "Good", "Bad" and without non relevant words like "Steph", "Controlization" and other...

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training the Naive Bayes classification model on the Training set, we can use also other classification models not only Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results which is 1 or 0 and that will represnt if review is positive or not
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)  
print(cm)
print(accuracy_score(y_test, y_pred)) # this number of correct prediction devied by total number of observation in test set









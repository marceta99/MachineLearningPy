# Apriori

# Run the following command in the terminal to install the apyori package: pip install apyori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
# U ovom data setu nemamo imena kolona kao kod ostalih data setova jer ovde je za svaku kupovinu u prodavnici koja je 
# svaki novi red, samo po kolonama izlistano proizvodi koji su kupljeni u toj kupovini
# tako da nemamo imena kolona kao kod ostalih datasetova i zato stavljamo header=none
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

#this apyori model from apyori package does not accepts dataframe and accepts list as input
# so because this dataset is dataframe, we will have to create list from that dataframe and we will name it transactions
# ovaj transactions je niz nizova, odnosno matrica
# also accepted format of values has to be strings so we use this str() function
transactions = []
for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
  
# in apriori we dont split data set on training and test set
#but we train model on full data set

# this apriori functions not only trains aproiry model but at the same time 
# returns rules, which are outcomes of this model

# we will set some minimum support values for rules, because we dont want rules
# that have very small support, so we will set some custom support minimum value 
# and that is min_support and that depends of your example but in this one we want
# basket elements that are bought at least 3 a day, and product that sells less than
# that a day are not important to me. Because this dataset show data on seven days
# purchases that happened we will mutiplay that by 7 and devide by 7501 because that
# is total number of purchases in our data set and we will get 3*7/7501=0.003 which is our min support

# next thing is min confidence, here he just tried few different min_confidence values from 0 to 1
# and saw that 0.2 is okay for this example, so not some logic, just tried different values

# next thing is min lift and generaly the lifts bellow 3 are not that relevant

# so in this example we want to have one product on left side of the rule and one
# product on the right side, because the goal is to get one product and pair it with
# some other product which will give for free if customer buy first product so we have
# to find with this rules which product are usually bought together 
# and that is why we set min_lenght=2 and max_length=2 because we only want exactly two products
# one on the left and one on the right side that will pair with first one and give it for free
# if we want for example to make if you bought two products you will get third one for free
# then we would set min_length=3 and max_length=3

#Well, you know, we're done with this apriori function which will return the rules respecting all these values we set for the parameters. A minimum support of 0.003 which means that the products in the rules appear at least oh 0.3% of the time. Then the minimum confidence, which means that for each product A in the left hand side of the rules well we will have product B in the right hand side of the rule at least 20% of the time. And then we have a minimum lift of three and we have only two products in our rules. Thanks to this min length equals two and max length equals two.

# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Visualising the results

## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
print(results)

#so we see here for example that if people bought light cream confidence is 0.29
# that they will buy chicken, so there is 29% chance that if somebody bought light cream that they will but chicken
# and support is 0.0045 that means that this rule happens in 0.45% of purchases

# so because lift is the most importan metric we will display this rules sorted so that
# we can see the ones with the biggest lift and that is the most important ones

#So, as we can see it is a function that takes as input, the results, meaning these results, the rules as they are right now, you know, organized this way. And then it'll take, separately, the left hand side of the rule, meaning the product at the left hand side of the rule, then the product at the right hand side of the rule, then the support of all the rules, then the confidences, the lift of these rules. And then it'll return all the rules with the left hand side and the right hand side and their supports, confidences and lift inside the list, right? That's why we have this list function here again, alright? So that's what this 'inspect' function will do. And then at the end, we create the final Pandas data frame, which takes as input the output of this inspect function. And besides we add the column names, you know, the first column will be the left hand side of the rule. The second column will be the right hand side of the rule. The third column will be the support, force, the confidence, and finally the lift. So we will have a super nice table. With these columns and giving all the important information for each of the rules
## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

print(resultsinDataFrame)

## Displaying the results sorted by descending lifts
print(resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))















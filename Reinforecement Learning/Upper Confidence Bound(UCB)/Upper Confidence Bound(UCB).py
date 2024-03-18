#Okay? So what happened exactly is that the advertising team prepared 10 different ads, you know with 10 different designs. For example, on one ad we will see the SUV in a beautiful mountain. On the other Ad, we will see the SUV in a futuristic city, on another ad, we'll see the SUV in a charming city, you know, like a charming city in a south of France or Italy on another ad we'll see the car on the moon, you know, why not, on another ad we'll see the car on a beautiful countryside, cornfield you know, something like that. So basically all the ads have different designs. And advertising team is wondering, well which Ad will convert the most? You know, which ad will attract the most people to click the ad and then potentially buy the SUV? So we have these 10 different ads and what we're gonna do, and that is the process of online learning, we're gonna show these Ads to different users online. You know, once they connect to a certain website or to a search engine, you know, it can be ads that appear at the top of a page when you type or research on Google. We're gonna show one of these ads each time the user connects to the webpage, and we're gonna record the result whether this user clicked yes or no on the ad. Okay? So just to recap, there is a first user that connects to let's say a webpage or algorithm, which will be here first. UCB will select an ad to show to this user, and then the user will decide to click yes or no on the ad. If the user clicks on the ad, we will record it as one. And if the user doesn't click on the ad we will record it as zero. Okay? And then a new user connects the webpage, and same the algorithm selects an ad to show to this new user. And if this new user clicks the ad then it's a one, and if not, it's a zero. Okay? And we're gonna do this for lots of users, actually 10,000 users. And that's what this data set is about. However, now there is something you must absolutely understand and that is very, very important. Make sure to understand it and make sure to rewind if this is not understood. Okay? So I'm going to explain this, please listen carefully. So you know, in reality what happens is that users connect one by one to the webpage, and for each of them we successively show them the ad. Right? So everything happens in real time, you know, it's a dynamic process. It's not a static process with a static dataset, which was recorded over a certain period of time. It's a real time process. And therefore the only way to simulate this would be either that I, you know make 10 real ads right now, you know, 10 real ads of a car. Then I open a Google AdWords account, and then I show the Ads for real to some users, you know, real persons connecting to the website. Of course, I'm not gonna do this because first of all, this is costly. And then, you know, this would deceive the users. Well, you know, I would have to really sell a car somehow. So of course this is not an option and therefore I have to make a simulation, okay? I have to make a simulation. And this simulation is exactly given by this dataset. Because in this dataset what happens is that each row corresponds to the different users connecting to the webpage and to whom we're gonna show the ads. And then each column of this dataset corresponds to the different ads. Okay? From Ad one to Ad 10. And this dataset is a simulation in the sense that each time a user connects to the webpage, well this dataset tells us, even if we wouldn't know, in reality this dataset tells us on which Ad the user of the row would click on. You know. So for example, this first user you know, this corresponds to the first user to whom we're gonna show the Ad. And what these cells mean is that this user would click on Ad one, if we show this user Ad one, then it wouldn't click Ad two, if we show Ad two because there is a zero here, then the user wouldn't click Ad three, if we show Ad three. It wouldn't click Ad four, if we show Ad four, but then it would click at five if we showed Ad five, and et cetera.
# Right? The Ad on which the users click the most. So I know that we could do it, for example, with a naive strategy, you know, a naive algorithm like a simple one where we collect some simple statistics to see which Ad is most frequently clicked on. But remember, as kiral explained in the intuition lectures each time we impress an Ad, you know, on the website or the Google search engine, well this incurs a cost. Right? It has a cost to impress Ad. Therefore, we need to figure out as fast as possible, you know, in the minimum number of rounds, because you know, the users here are represented as rounds because we show the ads to the users one by one, as in one round after the other. So we need to figure out in a minimum number of rounds which Ad converts the most, meaning, which is the best ad to which the users are most attracted to. And that's why we need a stronger algorithm than a simple statistics algorithm. And that's stronger algorithm will be first UCB and then Thompson sampling.

# Upper Confidence Bound (UCB)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
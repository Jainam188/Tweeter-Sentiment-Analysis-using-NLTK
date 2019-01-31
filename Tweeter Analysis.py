import tweepy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


consumerKey = ''
consumerSecret = ''
accessToken = ''
accessTokenSecret = ''

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)

api = tweepy.API(auth)

tweets = api.search("Artificial Intelligence", count=200)

data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

print(data.head(10))

print(tweets[0].created_at)

import nltk

nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

list = []

for index, row in data.iterrows():
    ss = sid.polarity_scores(row["Tweets"])
    list.append(ss)

se = pd.Series(list)

data["Polarity"] = se.values

print(data.head(100))

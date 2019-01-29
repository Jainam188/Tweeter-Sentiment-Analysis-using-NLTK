import tweepy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


consumerKey = 'xeKYme7fBVQwagoU0OEI4yonI'
consumerSecret = 'caq7LfbYewR3UokDLwqhDM3NWw1gaqUsk40vaaDf2ypGnxQYyn'
accessToken = '1874616464-wGJuRGp6eEne71zSGNP49xyMgqh3Dfkb0CH6HQ7'
accessTokenSecret = '3JaGJD0mcLJLTRZgpdAScFikgyDwT0g7rRVgORM4bUGxi'

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
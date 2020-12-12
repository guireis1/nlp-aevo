# from tweepy import *
import tweepy
import pandas as pd
import csv
import numpy as np
import re
import string
import pandas as pd
# import preprocessor as p
consumer_key = 'gqPbGlNrhKXedgPSZWiYAMXTL'
consumer_secret = '40CwT8qXD2XojrUrcHl7JDfhJBIEtlOlTnRRJSroXWMmMu4YzI'
access_key= '310550023-qDcttLKeSH2AnnJR6kgheJjzLEJ1elbYsxfUzbwN'
access_secret = '2z9uyePwzvIL6OkNi4qSFtYbQEf1szcLoFBm2OlfK7GM7'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
date_since = "2020-08-16"
search_words = "#inovação"      #enter your words
new_search = search_words + " -filter:retweets"

# start_date = datetime.datetime(2018, 1, 19, 12, 00, 00)
# end_date = datetime.datetime(2018, 1, 19, 13, 00, 00)

tweets = tweepy.Cursor(api.search,
              q=new_search,
              lang="pt").items(10)

save_hash = []
save_location = []
save_data = []
save_text = []

for tweet in tweets:
    print(tweet.full_text)
    # print ("Screen-name:", tweet.author.screen_name.encode('utf8'))

    # print ("Tweet:", tweet.text.encode('utf8'))
    # print ("Retweeted:", tweet.retweeted)
    # print ("Favourited:", tweet.favorited)
    # print ("Location:", tweet.user.location.encode('utf8'))
    # print ("Time-zone:", tweet.user.time_zone)
    # print ("Geo:", tweet.geo)
    # print("#: ", tweet.entities.get('hashtags'))
    # print("Tweet created:", tweet.created_at)
    # print("//////////////////")
    if tweet.entities.get('hashtags'):
        # save_location.append(tweet.user.location)
        # print("Tweet created:", tweet.created_at)
        for i in tweet.entities.get('hashtags'):
            # print(i)
            save_hash_local.append(i['text'])
        if tweet.user.location:
            save_location.append(tweet.user.location)
        else:
            save_location.append('')

        save_date.append(tweet.created_at)

        save_hash.append(save_hash_local)
        save_hash_local = []
        save_text.append(tweet.full_text)

df = pd.DataFrame({'hash':save_hash, 'location':save_location,'date':save_date,'text':save_text})
# df.to_csv('data/inovacao5.csv',header=True,index = False)

#Sentiment Analysis----------------------------------------------------------------------------------------------

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../2020005_Final"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
tweets=pd.read_csv("chatgpt1.csv",encoding = "ISO-8859-1")
tweets.head(50000)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *

from nltk import tokenize

sid = SentimentIntensityAnalyzer()

tweets['sentiment_compound_polarity']=tweets.Text.apply(lambda x:sid.polarity_scores(x)['compound'])
tweets['sentiment_neutral']=tweets.Text.apply(lambda x:sid.polarity_scores(x)['neu'])
tweets['sentiment_negative']=tweets.Text.apply(lambda x:sid.polarity_scores(x)['neg'])
tweets['sentiment_pos']=tweets.Text.apply(lambda x:sid.polarity_scores(x)['pos'])
tweets['sentiment_type']=''
tweets.loc[tweets.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
tweets.loc[tweets.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
tweets.loc[tweets.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'
tweets.head(50000)

tweets.sentiment_type.value_counts().plot(kind='bar',title="sentiment analysis")

#Hashtag Analysis--------------------------------------------------------------------------------------------------

#pip install spacy
#pip install en_core_web_sm-3.1.0-py3-none-any.whl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from string import punctuation
import collections
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('chatgpt1.xlsx')

from sklearn.cluster import KMeans

# Read the XLSX file
data = pd.read_csv('chatgpt1.csv')

# Extract hashtags from the 'hashtag' column
hashtags = data['hashtag'].dropna().tolist()

# Count the frequency of each hashtag
hashtag_counts = {}
for hashtag in hashtags:
    if isinstance(hashtag, str):
        if hashtag in hashtag_counts:
            hashtag_counts[hashtag] += 1
        else:
            hashtag_counts[hashtag] = 1

# Sort the hashtags by frequency in descending order
sorted_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)

# Get the top 10 hashtags and their frequencies
top_hashtags = sorted_hashtags[:10]
hashtags, counts = zip(*top_hashtags)

# Create a bar graph for the top hashtags
plt.bar(hashtags, counts)
plt.xlabel('Hashtags')
plt.ylabel('Frequency')
plt.title('Top 10 Hashtags')
plt.xticks(rotation=45)

# Display the graph
plt.tight_layout()
plt.show()

#User Classification----------------------------------------------------------------------------------------------

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load the tweets from the CSV file
df = pd.read_csv('chatgpt1.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Tweet Id'], test_size=0.2, random_state=42)

# Vectorize the tweets
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# Predict the user based on the tweets
y_pred = classifier.predict(X_test_vec)

# Print the predicted users for the test tweets
for tweet, user in zip(X_test, y_pred):
    print(f'Tweet: {tweet}\nPredicted User: {user}\n')

# Evaluate the classifier
accuracy = classifier.score(X_test_vec, y_test)
print(f'Accuracy: {accuracy}')

#Clustering--------------------------------------------------------------------------------------------------------

import pandas as pd
from sklearn.cluster import KMeans

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('chatgpt1.csv')

# Extract the column 'LikeCount' as the feature for clustering
features = data['LikeCount'].values.reshape(-1, 1)

# Define the number of clusters
n_clusters = 3

# Initialize the k-means clustering algorithm
kmeans = KMeans(n_clusters=n_clusters)

# Perform clustering and retrieve the cluster labels assigned to each data point
cluster_labels = kmeans.fit_predict(features)

# Add the cluster labels as a new column to the DataFrame
data['Cluster'] = cluster_labels

# Print the resulting clusters
print(data[['LikeCount', 'Cluster']])
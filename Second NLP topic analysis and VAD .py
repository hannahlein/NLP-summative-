#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk import stem
stemmer = stem.PorterStemmer()
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
nltk.download('punkt')
import string
punct = list(string.punctuation)
from collections import Counter
import requests
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
get_ipython().system('pip install PRAW')
import numpy as np
import praw
import datetime
from urllib import request


# In[2]:


#get the dataset from Project Gutenberg through its URL 

from urllib import request
url = "https://www.gutenberg.org/cache/epub/1581/pg1581.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')


# In[3]:


#create datasets for the individual Gospels 

def get_book(chapter_start, chapter_end): 
    start_pos = raw.index(chapter_start)
    end_pos = raw.index(chapter_end)
    return raw[start_pos:end_pos]


#get Matthew 
HG_S_Matthew = get_book(
    "THE HOLY GOSPEL OF JESUS CHRIST ACCORDING TO SAINT MATTHEW", 
    "THE HOLY GOSPEL OF JESUS CHRIST ACCORDING TO ST. MARK"
    )

#get Mark 
HG_S_Mark = get_book(
    "THE HOLY GOSPEL OF JESUS CHRIST ACCORDING TO ST. MARK", 
    "THE HOLY GOSPEL OF JESUS CHRIST ACCORDING TO ST. LUKE"
    )


#get Luke 

HG_S_Luke = get_book(
    "THE HOLY GOSPEL OF JESUS CHRIST ACCORDING TO ST. LUKE", 
    "THE HOLY GOSPEL OF JESUS CHRIST ACCORDING TO ST. JOHN"
    )


#get John 

HG_S_John = get_book(
    "THE HOLY GOSPEL OF JESUS CHRIST ACCORDING TO ST. JOHN", 
    "THE ACTS OF THE APOSTLES"
    )


# In[4]:


#topic analysis: install necessary libraries 

get_ipython().system('pip install plotly')
import pandas as pd
import spacy
import re
import seaborn as sns
sns.set()
# !pip install TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from IPython.display import IFrame
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from scipy.spatial import distance


# In[5]:


get_ipython().system('pip install nltk')
from nltk.tokenize import sent_tokenize


# In[6]:


#tokenise Matthew per sentence 

clean_Matthew = ' '.join([i for i in HG_S_Matthew.splitlines() if i != ''])
clean_Matthew = re.sub(r'\d+', '', clean_Matthew)  # removes all digits
matthew_tokenised = sent_tokenize(clean_Matthew)


# In[7]:


matthew_tokenised


# In[8]:


#vectors Matthew 

data = matthew_tokenised

vectorizer = TfidfVectorizer(input = "content", strip_accents = "ascii", stop_words = "english")

vectors = vectorizer.fit_transform(matthew_tokenised)

vectors = vectors.todense().tolist()

df = pd.DataFrame(vectors,columns=vectorizer.get_feature_names_out())
df


# In[9]:


#3D plot Matthew 

pca_1 = PCA(n_components = 3)
comps_1 = pca_1.fit_transform(df)
pc_df_1 = pd.DataFrame(data = comps_1, columns = ['PC'+str(i) for i in range(1, comps_1.shape[1]+1)])

clustering = AgglomerativeClustering(n_clusters = 3, metric = 'euclidean', linkage = 'ward').fit(df)

kmeans = KMeans(n_clusters = 5, random_state=0, n_init="auto").fit(df)

df_all = pd.concat([df, pc_df_1], axis = 1)
df_all['clusters_ag'] = [str(i) for i in clustering.labels_]
df_all['clusters_knn'] = [str(i) for i in kmeans.labels_]
df_all['text'] = [str(i) for i in matthew_tokenised]


fig = px.scatter_3d(df_all, x='PC1', y='PC2', z='PC3',
              color='clusters_ag', hover_data = ['text'])

fig.update_traces(marker=dict(size = 5, line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig.show('notebook')


# In[10]:


#observing clusters in Matthew

#Jesus said/ answered 

#shall 

#sub-cluster of shall kill (And whosever shall kill shall be in danger of judgement, And they shall kill him)

#Father and Son 

#marital questions (Is it lawful for a man to put away his wife for every cause?)

#body parts (hand, face, eye, foot)

#disciples 


# In[11]:


#tokenise Mark per sentence 

clean_Mark = ' '.join([i for i in HG_S_Mark.splitlines() if i != ''])
clean_Mark = re.sub(r'\d+', '', clean_Mark)  # removes all digits
mark_tokenised = sent_tokenize(clean_Mark)


#vectors Mark 

data = mark_tokenised

vectorizer = TfidfVectorizer(input = "content", strip_accents = "ascii", stop_words = "english")

vectors = vectorizer.fit_transform(mark_tokenised)

vectors = vectors.todense().tolist()

df = pd.DataFrame(vectors,columns=vectorizer.get_feature_names_out())
df


#3D plot Mark 

pca_1 = PCA(n_components = 3)
comps_1 = pca_1.fit_transform(df)
pc_df_1 = pd.DataFrame(data = comps_1, columns = ['PC'+str(i) for i in range(1, comps_1.shape[1]+1)])

clustering = AgglomerativeClustering(n_clusters = 3, metric = 'euclidean', linkage = 'ward').fit(df)

kmeans = KMeans(n_clusters = 5, random_state=0, n_init="auto").fit(df)

df_all = pd.concat([df, pc_df_1], axis = 1)
df_all['clusters_ag'] = [str(i) for i in clustering.labels_]
df_all['clusters_knn'] = [str(i) for i in kmeans.labels_]
df_all['text'] = [str(i) for i in mark_tokenised]

fig = px.scatter_3d(df_all, x='PC1', y='PC2', z='PC3',
              color='clusters_ag', hover_data = ['text'])

fig.update_traces(marker=dict(size = 5, line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig.show('notebook')


# In[12]:


#observation of clusters in Mark 

#shall 

#Jesus said/ Jesus answered 

#questions (Do you not yet know nor understand? And having ears hear you not? 
#Couldst thou not watch one hour? Have you not faith yet)

#Jesus performing miracles/ healing people 
#(And he stretched it forth and his hand was restored under him, the devil is gone out of thy daughter)


# In[13]:


#tokenise Luke per sentence 

clean_Luke = ' '.join([i for i in HG_S_Luke.splitlines() if i != ''])
clean_Luke = re.sub(r'\d+', '', clean_Luke)  # removes all digits
luke_tokenised = sent_tokenize(clean_Luke)


#vectors Luke

data = luke_tokenised

vectorizer = TfidfVectorizer(input = "content", strip_accents = "ascii", stop_words = "english")

vectors = vectorizer.fit_transform(luke_tokenised)

vectors = vectors.todense().tolist()

df = pd.DataFrame(vectors,columns=vectorizer.get_feature_names_out())
df


#3D plot Luke

pca_1 = PCA(n_components = 3)
comps_1 = pca_1.fit_transform(df)
pc_df_1 = pd.DataFrame(data = comps_1, columns = ['PC'+str(i) for i in range(1, comps_1.shape[1]+1)])

clustering = AgglomerativeClustering(n_clusters = 3, metric = 'euclidean', linkage = 'ward').fit(df)

kmeans = KMeans(n_clusters = 5, random_state=0, n_init="auto").fit(df)

df_all = pd.concat([df, pc_df_1], axis = 1)
df_all['clusters_ag'] = [str(i) for i in clustering.labels_]
df_all['clusters_knn'] = [str(i) for i in kmeans.labels_]
df_all['text'] = [str(i) for i in luke_tokenised]


fig = px.scatter_3d(df_all, x='PC1', y='PC2', z='PC3',
              color='clusters_ag', hover_data = ['text'])

fig.update_traces(marker=dict(size = 5, line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig.show('notebook')


# In[14]:


#observation of clusters in Luke 

#shall (Seek and you shall find, Shall call me blessed)

#Jesus said/ Jesus answered 


# In[15]:


#tokenise John per sentence 

clean_John = ' '.join([i for i in HG_S_John.splitlines() if i != ''])
clean_John = re.sub(r'\d+', '', clean_John)  # removes all digits
john_tokenised = sent_tokenize(clean_John)


#vectors John 

data = john_tokenised

vectorizer = TfidfVectorizer(input = "content", strip_accents = "ascii", stop_words = "english")

vectors = vectorizer.fit_transform(john_tokenised)

vectors = vectors.todense().tolist()

df = pd.DataFrame(vectors,columns=vectorizer.get_feature_names_out())
df


#3D plot John 

pca_1 = PCA(n_components = 3)
comps_1 = pca_1.fit_transform(df)
pc_df_1 = pd.DataFrame(data = comps_1, columns = ['PC'+str(i) for i in range(1, comps_1.shape[1]+1)])

clustering = AgglomerativeClustering(n_clusters = 3, metric = 'euclidean', linkage = 'ward').fit(df)

kmeans = KMeans(n_clusters = 5, random_state=0, n_init="auto").fit(df)

df_all = pd.concat([df, pc_df_1], axis = 1)
df_all['clusters_ag'] = [str(i) for i in clustering.labels_]
df_all['clusters_knn'] = [str(i) for i in kmeans.labels_]
df_all['text'] = [str(i) for i in john_tokenised]


fig = px.scatter_3d(df_all, x='PC1', y='PC2', z='PC3',
              color='clusters_ag', hover_data = ['text'])

fig.update_traces(marker=dict(size = 5, line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig.show('notebook')


# In[16]:


#observation of clusters in John 

#father 

#shall 

#Jesus answered/ Jesus said 

#questions (Whom seekest thou, Whom doust thou make thyself, Believest thou this?)


# In[30]:


import nltk
import numpy as np
import plotly.graph_objs as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

#load the Bible text data
with open('lemmatised_Matthew.csv', 'r') as file:
    text = file.read()

#tokenise the text into sentences
sentences = nltk.sent_tokenize(text)

#initialise the VADER sentiment analyser
analyzer = SentimentIntensityAnalyzer()

#calculate the valence, arousal, and dominance for each sentence
vad_scores = []
for sentence in sentences:
    scores = analyzer.polarity_scores(sentence)
    vad_scores.append([scores['pos'] - scores['neg'], scores['pos'] + scores['neg'], scores['pos'] - scores['neg'] + scores['neu']])

#convert the scores to a numpy array for plotting
vad_scores = np.array(vad_scores)

#create the interactive 3D scatter plot using plotly
fig = go.Figure(data=[go.Scatter3d(
    x=vad_scores[:,0],
    y=vad_scores[:,1],
    z=vad_scores[:,2],
    mode='markers',
    marker=dict(
        size=5,
        color=vad_scores[:,2],     # Use dominance for color scale
        colorscale='Viridis',      # Set color scale to 'Viridis'
        opacity=0.8
    )
)])

#set the axis labels and ranges
fig.update_layout(scene=dict(
    xaxis_title='Valence',
    yaxis_title='Arousal',
    zaxis_title='Dominance',
    xaxis_range=[-1, 1],
    yaxis_range=[-1, 1],
    zaxis_range=[-1, 1],
))

#show the plot
fig.show()


# In[29]:


import nltk
import numpy as np
import plotly.graph_objs as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

#load the Bible text data
with open('lemmatised_Mark.csv', 'r') as file:
    text = file.read()

#tokenise the text into sentences
sentences = nltk.sent_tokenize(text)

#initialise the VADER sentiment analyser
analyzer = SentimentIntensityAnalyzer()

#calculate the valence, arousal, and dominance for each sentence
vad_scores = []
for sentence in sentences:
    scores = analyzer.polarity_scores(sentence)
    vad_scores.append([scores['pos'] - scores['neg'], scores['pos'] + scores['neg'], scores['pos'] - scores['neg'] + scores['neu']])

#convert the scores to a numpy array for plotting
vad_scores = np.array(vad_scores)

#create the interactive 3D scatter plot using plotly
fig = go.Figure(data=[go.Scatter3d(
    x=vad_scores[:,0],
    y=vad_scores[:,1],
    z=vad_scores[:,2],
    mode='markers',
    marker=dict(
        size=5,
        color=vad_scores[:,2],     # Use dominance for color scale
        colorscale='Viridis',      # Set color scale to 'Viridis'
        opacity=0.8
    )
)])

#set the axis labels and ranges
fig.update_layout(scene=dict(
    xaxis_title='Valence',
    yaxis_title='Arousal',
    zaxis_title='Dominance',
    xaxis_range=[-1, 1],
    yaxis_range=[-1, 1],
    zaxis_range=[-1, 1],
))

#show the plot
fig.show()


# In[28]:


import nltk
import numpy as np
import plotly.graph_objs as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

#load the Bible text data
with open('lemmatised_Luke.csv', 'r') as file:
    text = file.read()

#tokenise the text into sentences
sentences = nltk.sent_tokenize(text)

#initialise the VADER sentiment analyser
analyzer = SentimentIntensityAnalyzer()

#calculate the valence, arousal, and dominance for each sentence
vad_scores = []
for sentence in sentences:
    scores = analyzer.polarity_scores(sentence)
    vad_scores.append([scores['pos'] - scores['neg'], scores['pos'] + scores['neg'], scores['pos'] - scores['neg'] + scores['neu']])

#convert the scores to a numpy array for plotting
vad_scores = np.array(vad_scores)

#create the interactive 3D scatter plot using plotly
fig = go.Figure(data=[go.Scatter3d(
    x=vad_scores[:,0],
    y=vad_scores[:,1],
    z=vad_scores[:,2],
    mode='markers',
    marker=dict(
        size=5,
        color=vad_scores[:,2],     # Use dominance for color scale
        colorscale='Viridis',      # Set color scale to 'Viridis'
        opacity=0.8
    )
)])

#set the axis labels and ranges
fig.update_layout(scene=dict(
    xaxis_title='Valence',
    yaxis_title='Arousal',
    zaxis_title='Dominance',
    xaxis_range=[-1, 1],
    yaxis_range=[-1, 1],
    zaxis_range=[-1, 1],
))

#show the plot
fig.show()


# In[27]:


import nltk
import numpy as np
import plotly.graph_objs as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

#load the Bible text data
with open('lemmatised_John.csv', 'r') as file:
    text = file.read()

#tokenise the text into sentences
sentences = nltk.sent_tokenize(text)

#initialise the VADER sentiment analyser
analyzer = SentimentIntensityAnalyzer()

#calculate the valence, arousal, and dominance for each sentence
vad_scores = []
for sentence in sentences:
    scores = analyzer.polarity_scores(sentence)
    vad_scores.append([scores['pos'] - scores['neg'], scores['pos'] + scores['neg'], scores['pos'] - scores['neg'] + scores['neu']])

#convert the scores to a numpy array for plotting
vad_scores = np.array(vad_scores)

#create the interactive 3D scatter plot using plotly
fig = go.Figure(data=[go.Scatter3d(
    x=vad_scores[:,0],
    y=vad_scores[:,1],
    z=vad_scores[:,2],
    mode='markers',
    marker=dict(
        size=5,
        color=vad_scores[:,2],     # Use dominance for color scale
        colorscale='Viridis',      # Set color scale to 'Viridis'
        opacity=0.8
    )
)])

#set the axis labels and ranges
fig.update_layout(scene=dict(
    xaxis_title='Valence',
    yaxis_title='Arousal',
    zaxis_title='Dominance',
    xaxis_range=[-1, 1],
    yaxis_range=[-1, 1],
    zaxis_range=[-1, 1],
))

#show the plot
fig.show()


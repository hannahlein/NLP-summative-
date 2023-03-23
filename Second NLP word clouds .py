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


print(HG_S_Matthew)


# In[5]:


#tokenise Matthew 

HG_S_Matthew = word_tokenize(HG_S_Matthew)

#stop words 

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

words = HG_S_Matthew

filtered_words = [word for word in words if word.lower() not in stop_words]

print(filtered_words)
HG_S_Matthew_new = filtered_words


# In[6]:


#lemmatise Matthew 

lemmas = [lemmatizer.lemmatize(i) for i in HG_S_Matthew_new]

#create a dataframe that contains the lemmas 

lemmas_df = pd.DataFrame()
lemmas_df['lemmatised_Matthew'] = lemmas

#save the dataframe in a csv file 

lemmas_df.to_csv('lemmatised_Matthew.csv', index=True)


# In[7]:


#create a word cloud for Matthew 

get_ipython().system('pip install wordcloud')

from wordcloud import WordCloud
import matplotlib.pyplot as plt

#convert the list to string
text = ' '.join(lemmas)

#create the word cloud object
wordcloud = WordCloud(width=800, height=800, background_color='white').generate(text)

#plot the word cloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

#show the plot
plt.show()


# In[8]:


#tokenise Mark 

HG_S_Mark = word_tokenize(HG_S_Mark)

#stop words 

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

words = HG_S_Mark

filtered_words = [word for word in words if word.lower() not in stop_words]

print(filtered_words)
HG_S_Mark_new = filtered_words


# In[9]:


#lemmatise Mark 

lemmas = [lemmatizer.lemmatize(i) for i in HG_S_Mark_new]

#create a dataframe that contains the lemmas 

lemmas_df = pd.DataFrame()
lemmas_df['lemmatised_Mark'] = lemmas

#save the dataframe in a csv file 

lemmas_df.to_csv('lemmatised_Mark.csv', index=True)


# In[19]:


#create a word cloud for Mark 

get_ipython().system('pip install wordcloud')

from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = ' '.join(lemmas)

wordcloud = WordCloud(width=800, height=800, background_color='white').generate(text)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()


# In[11]:


#tokenise Luke 

HG_S_Luke = word_tokenize(HG_S_Luke)

#stop words 

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

words = HG_S_Luke

filtered_words = [word for word in words if word.lower() not in stop_words]

print(filtered_words)
HG_S_Luke_new = filtered_words


# In[12]:


#lemmatise Luke 

lemmas = [lemmatizer.lemmatize(i) for i in HG_S_Luke_new]

#create a dataframe that contains the lemmas 

lemmas_df = pd.DataFrame()
lemmas_df['lemmatised_Luke'] = lemmas

#save the dataframe in a csv file 

lemmas_df.to_csv('lemmatised_Luke.csv', index=True)


# In[13]:


#create a word cloud for Luke 

get_ipython().system('pip install wordcloud')

from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = ' '.join(lemmas)

wordcloud = WordCloud(width=800, height=800, background_color='white').generate(text)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()


# In[14]:


#tokenise John 

HG_S_John = word_tokenize(HG_S_John)

#stop words 

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

words = HG_S_John 

filtered_words = [word for word in words if word.lower() not in stop_words]

print(filtered_words)
HG_S_John_new = filtered_words


# In[15]:


#lemmatise John  

lemmas = [lemmatizer.lemmatize(i) for i in HG_S_John_new]

#create a dataframe that contains the lemmas 

lemmas_df = pd.DataFrame()
lemmas_df['lemmatised_John'] = lemmas

#save the dataframe in a csv file 

lemmas_df.to_csv('lemmatised_John.csv', index=True)


# In[16]:


#create a word cloud for John 

get_ipython().system('pip install wordcloud')

from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = ' '.join(lemmas)

wordcloud = WordCloud(width=800, height=800, background_color='white').generate(text)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()


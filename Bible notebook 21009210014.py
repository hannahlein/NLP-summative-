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


# In[40]:


from urllib import request
url = "https://www.gutenberg.org/cache/epub/1581/pg1581.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
raw[:750000]


# In[4]:


#clean the data (get rid of special characters, punctuation, numbers etc.)


# In[5]:


raw = raw.encode('ascii', 'ignore')
raw = raw.decode()


# In[6]:


raw


# In[7]:


raw = ' '.join(raw.splitlines())
raw = raw.lower()


# In[8]:


raw


# In[9]:


raw = raw.translate(str.maketrans("", "", string.punctuation))


# In[10]:


raw


# In[11]:


raw = ''.join(c for c in raw if c.isalpha() or c.isspace())


# In[12]:


raw


# In[13]:


raw = word_tokenize(raw)


# In[14]:


raw


# In[15]:


#remove most frequent words that don't carry meaning (e. g. "the") and put them in raw_filtered 


# In[16]:


import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

words = raw

filtered_words = [word for word in words if word.lower() not in stop_words]

print(filtered_words)
raw_filtered = filtered_words


# In[17]:


raw_filtered


# In[18]:


lemmas = [lemmatizer.lemmatize(i) for i in raw_filtered]


# In[19]:


lemmas 


# In[26]:


lemmas_df = pd.DataFrame()
lemmas_df['lemmas_dataset'] = lemmas


# In[27]:


lemmas_df


# In[28]:


lemmas_df.to_csv('lemmas_dataset.csv', index=True)


# In[20]:


counts = pd.Series(Counter(lemmas))


# In[21]:


counts 


# In[22]:


sns.displot(counts, kind = 'kde')
plt.show()


# In[23]:


sns.lineplot(x = [len(i) for i in counts.index], y = [i for i in counts.values])

